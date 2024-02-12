# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset
from dpt_wrapper import DPTWrapper
import torch.nn.functional as F
import numpy as np
import glob
import cv2
import utils.imutils as imutils
import utils.utils as utils


def get_valid_frames(scan_root: str, scan_name: str):

    print(f"Processing valid frames for {scan_name}...")
    
    # get scannet directories
    scan_dir = os.path.join(scan_root)
    sensor_data_dir = os.path.join(scan_dir, "sensor_data")
    meta_file_path = os.path.join(scan_dir, scan_name + ".txt")

    with open(meta_file_path, "r") as f:
        meta_info_lines = f.readlines()
        meta_info_lines = [line.split(" = ") for line in meta_info_lines]
        meta_data = {key: val for key, val in meta_info_lines}

    # fetch total number of color files
    color_file_count = int(meta_data["numColorFrames"].strip())

    dist_to_last_valid_frame = 0
    bad_file_count = 0
    valid_frames = []
    for frame_id in range(color_file_count):
        # for a frame to be valid, we need a valid pose and a valid
        # color frame.

        color_filename = os.path.join(sensor_data_dir, f"frame-{frame_id:06d}.color.jpg")
        depth_filename = color_filename.replace(f"color.jpg", f"depth.png")
        pose_path = os.path.join(sensor_data_dir, f"frame-{frame_id:06d}.pose.txt")

        # check if an image file exists.
        if not os.path.isfile(color_filename):
            dist_to_last_valid_frame += 1
            bad_file_count += 1
            continue

        # check if a depth file exists.
        if not os.path.isfile(depth_filename):
            dist_to_last_valid_frame += 1
            bad_file_count += 1
            continue

        world_T_cam_44 = np.genfromtxt(pose_path).astype(np.float32)
        # check if the pose is valid.
        if (
            np.isnan(np.sum(world_T_cam_44))
            or np.isinf(np.sum(world_T_cam_44))
            or np.isneginf(np.sum(world_T_cam_44))
        ):
            dist_to_last_valid_frame += 1
            bad_file_count += 1
            continue

        valid_frames.append(f"{scan_name} {frame_id:06d} {dist_to_last_valid_frame}")
        dist_to_last_valid_frame = 0

    print(
        f"Scene {scan_name} has {bad_file_count} bad frame files out of " f"{color_file_count}."
    )
    
    valid_frames = [int(frame_line.split(" ")[1]) for frame_line in valid_frames]
    
    return valid_frames

class ScanNetDataset(Dataset):
    def __init__(self, rootdir, scan, depth_estimator, output_height=484, output_width=684):
        super(ScanNetDataset, self).__init__()

        self.scan_path = os.path.join(rootdir, scan)
        self.ho, self.wo = output_height, output_width
        
        self.depth_estimator = depth_estimator

        # Replace the following line with any desired monocular depth estimator.
        # Our method has been tested with DPT, and uses it by default. 
        
        self.valid_frames_ids = get_valid_frames(self.scan_path, scan)
        
        self.image_paths = glob.glob(self.scan_path + "/sensor_data/frame-*.color.jpg")

        h, w = imutils.png2np( self.image_paths[0] ).shape[:2]
        sh, sw = self.ho / h, self.wo / w

        fin_intrinsics = os.path.join(self.scan_path, 'intrinsic/intrinsic_color.txt')
        with open(fin_intrinsics, 'r') as freader:
            self.K = torch.tensor([float(i) for i in freader.read().replace('\n', ' ').split()]).view(4, 4)
        self.K[0, :] *= sw
        self.K[1, :] *= sh
        
    def __len__(self):
        return len(self.valid_frames_ids)

    def __getitem__(self, idx):

        frame_id =  self.valid_frames_ids[idx]
        
        rgb_numpy = imutils.png2np( os.path.join( self.scan_path, f"sensor_data/frame-{frame_id:06d}.color.jpg") )
        
        rgb = F.interpolate( torch.from_numpy(rgb_numpy).permute(2, 0, 1).unsqueeze(0).float(),
                             (self.ho, self.wo),
                             mode="bilinear",
                             align_corners=True).squeeze(0)
        depth_scaling = cv2.imread( os.path.join( self.scan_path, f"sensor_data/frame-{frame_id:06d}.depth.png"), cv2.IMREAD_UNCHANGED ) / 1000.
        
        depth_scaling = F.interpolate(torch.tensor(depth_scaling).unsqueeze(0).unsqueeze(0), (self.ho, self.wo), mode='nearest').squeeze(0).float()

        # Estimate a monocular depth map
        inv_depth_mono = self.depth_estimator( rgb_numpy ) 
        inv_depth_mono = F.interpolate( torch.from_numpy(inv_depth_mono).unsqueeze(0).unsqueeze(0).float(),
                                        (self.ho, self.wo),
                                        mode='nearest').squeeze(0)
        #
        # Scale monocular depth map
        mask_scaling = depth_scaling > 1e-5
        inv_depth_scaling = 1 / depth_scaling
        inv_depth_scaling[~mask_scaling] = 0
        inv_depth_scaled, _, _ = utils.scale_depth( inv_depth_mono,
                                                    inv_depth_scaling,
                                                    mask_scaling )
        depth_scaled = (1 / inv_depth_scaled) 
        depth_cleaned = utils.clean_depth_edges(depth_scaled.squeeze(0)).unsqueeze(0)
        
        # Load the world-to-camera pose as a 4x4 matrix
        pose_file = os.path.join(self.scan_path, os.path.join( self.scan_path, f"sensor_data/frame-{frame_id:06d}.pose.txt"))
        with open(pose_file, 'r') as freader:
            pose_c2w = np.array([float(i) for i in freader.read().replace('\n', ' ').split()]).reshape(4, 4)
        pose_w2c = np.linalg.inv(pose_c2w)

        # We use OpenGL's coordinate system: +ve x is to the right, +ve y is up, +ve z is out of the screen
        # ScanNet poses are in OpenCV's system so we perform a change of bases here.
        M = np.eye(4)
        M[:, 1:3] *= -1
        pose_w2c = torch.from_numpy(np.matmul(M, pose_w2c))
                
        return { "rgb": rgb,
                 "depth": depth_cleaned,
                 "pose_w2c": pose_w2c,
                 "K": self.K,
                 "frame_id": frame_id}
    

