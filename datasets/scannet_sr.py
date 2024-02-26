# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
import sys
import pickle
import torch
from torch.utils.data import Dataset
from datasets.scannet import get_valid_frames
import torch.nn.functional as F
import numpy as np
import glob
import cv2
import utils.imutils as imutils
import utils.utils as utils


class ScanNetSRDataset(Dataset):
    def __init__(self, rootdir, scan, scannet_sr_path, output_height=484, output_width=684):
        super(ScanNetSRDataset, self).__init__()

        self.scan_path = os.path.join(rootdir, scan)
        self.ho, self.wo = output_height, output_width
        self.scannet_sr_path = os.path.join(scannet_sr_path, 'scannet', 'dense', 'depths', scan)


        # Replace the following line with any desired monocular depth estimator.
        # Our method has been tested with DPT, and uses it by default. 

        self.valid_frames_ids = get_valid_frames(self.scan_path, scan)

        self.image_paths = glob.glob(self.scan_path + "/sensor_data/frame-*.color.jpg")

        h, w = imutils.png2np(self.image_paths[0]).shape[:2]
        sh, sw = self.ho / h, self.wo / w

        fin_intrinsics = os.path.join(self.scan_path, 'intrinsic/intrinsic_color.txt')
        with open(fin_intrinsics, 'r') as freader:
            self.K = torch.tensor([float(i) for i in freader.read().replace('\n', ' ').split()]).view(4, 4)
        self.K[0, :] *= sw
        self.K[1, :] *= sh

    def __len__(self):
        return len(self.valid_frames_ids)

    def __getitem__(self, idx):
        frame_id = self.valid_frames_ids[idx]

        rgb_numpy = imutils.png2np(os.path.join(self.scan_path, f"sensor_data/frame-{frame_id:06d}.color.jpg"))

        rgb = F.interpolate(torch.from_numpy(rgb_numpy).permute(2, 0, 1).unsqueeze(0).float(),
                            (self.ho, self.wo),
                            mode="bilinear",
                            align_corners=True).squeeze(0)

        sr_frame_path = os.path.join(self.scannet_sr_path, f"{frame_id:06d}.pickle")
        objects_sr = []

        with (open(sr_frame_path, "rb")) as openfile:
            while True:
                try:
                    objects_sr.append(pickle.load(openfile))
                except EOFError:
                    break

        depth_scaling = cv2.imread(os.path.join(self.scan_path, f"sensor_data/frame-{frame_id:06d}.depth.png"),
                                   cv2.IMREAD_UNCHANGED) / 1000.

        depth_scaling = F.interpolate(torch.tensor(depth_scaling).unsqueeze(0).unsqueeze(0), (self.ho, self.wo),
                                      mode='nearest').squeeze(0).float()

        # Load SR frames

        depth_mono_sr = objects_sr[0]['depth_pred_s0_b1hw'].cpu()
        depth_mono_sr = F.interpolate(depth_mono_sr,
                                      (self.ho, self.wo),
                                      mode='nearest').squeeze(0)
        inv_depth_mono_sr = 1 / depth_mono_sr

        # Scale monocular depth map
        mask_scaling = depth_scaling > 1e-5
        inv_depth_scaling = 1 / depth_scaling
        inv_depth_scaling[~mask_scaling] = 0
        inv_depth_scaled_sr, _, _ = utils.scale_depth(inv_depth_mono_sr,
                                                   inv_depth_scaling,
                                                   mask_scaling)
        depth_scaled_sr = (1 / inv_depth_scaled_sr)
        depth_cleaned_sr = utils.clean_depth_edges(depth_scaled_sr.squeeze(0)).unsqueeze(0)

        # Load the world-to-camera pose as a 4x4 matrix
        pose_file = os.path.join(self.scan_path,
                                 os.path.join(self.scan_path, f"sensor_data/frame-{frame_id:06d}.pose.txt"))
        with open(pose_file, 'r') as freader:
            pose_c2w = np.array([float(i) for i in freader.read().replace('\n', ' ').split()]).reshape(4, 4)
        pose_w2c = np.linalg.inv(pose_c2w)

        # We use OpenGL's coordinate system: +ve x is to the right, +ve y is up, +ve z is out of the screen
        # ScanNet poses are in OpenCV's system so we perform a change of bases here.
        M = np.eye(4)
        M[:, 1:3] *= -1
        pose_w2c = torch.from_numpy(np.matmul(M, pose_w2c))

        return {"rgb": rgb,
                "depth": depth_cleaned_sr,
                "pose_w2c": pose_w2c,
                "K": self.K,
                "frame_id": frame_id}
