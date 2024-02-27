# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
import torch
import numpy as np
import argparse
from datasets.scannet_sr import ScanNetSRDataset

from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip
from dpt_wrapper import DPTWrapper
from pcupdate import PCUpdate
from utils import imutils
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_path", help="Base directory of ScanNet test scans.")
    parser.add_argument("--scannet_sr_path", help="Base directory of ScanNet frames processed with SR")
    parser.add_argument("--scans_list_path", help="Scans list.")
    parser.add_argument("--tuple_file_path", help="path to test tuple indices to save.")
    parser.add_argument("--outdir", default="output", help="Directory for saving output depth in.")
    parser.add_argument("--save_numpy", action="store_true", help="Save the processed depthmaps as Numpy files.")
    parser.add_argument("--save_viz", action="store_true", help="Save viz.")

    
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    if (torch.cuda.is_available()):
        device = "cuda:0"
    else:
        print("No CUDA device found; Exiting.")
        exit()

    h, w = 484, 648
    

    # read scan names from file
    with open(args.scans_list_path, "r") as f:
        scans = f.readlines()
        scans = [scan.strip() for scan in scans]

    with open(args.tuple_file_path, "r") as f:
        tuples = f.readlines()

    for scan in scans:
        scan_keyframes = [int(tuple.split(" ")[1]) for tuple in tuples if tuple.split(" ")[0].strip() == scan]
        
        dataset = ScanNetSRDataset(
            rootdir=args.scannet_path, 
            scan=scan,
            scannet_sr_path=args.scannet_sr_path,
            output_height=h, 
            output_width=w
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            # num_workers=1,
        )
        
        pcupdater = PCUpdate(h, w, device)
        
        output_frames = []
        np_list = {}
        method = "sr_tocd"
        output_dir = (Path(args.outdir) / f"{method}")
        output_dir.mkdir(exist_ok=True)
        for batch, sample in enumerate(tqdm(dataloader)):

            
            torch.cuda.empty_cache()

            pcupdater.update(sample['rgb'].to(device),
                            sample['depth'].to(device),
                            sample['pose_w2c'].to(device),
                            sample['K'].to(device) )

            depth_out = pcupdater.fused_depth.cpu()
            
            if batch == 0:
                mn, mx = torch.quantile(depth_out[depth_out > 0], 0.05), torch.quantile(depth_out[depth_out > 0], 0.95)

            outpath_frames = os.path.join(output_dir, f"{scan}")
            if not os.path.exists(outpath_frames):
                # Create a new directory because it does not exist
                os.makedirs(outpath_frames)
                print("The new directory is created!")

            if args.save_viz:
                outpath_frames = os.path.join(output_dir, f"{scan}")

                depth_comparison_rgb = imutils.np2png_d([sample['depth'].view(h, w).cpu().numpy(),
                                                         depth_out.view(h, w).cpu().numpy()],
                                                        fname=None,
                                                        vmin=mn,
                                                        vmax=mx)

                output = np.concatenate((sample['rgb'].squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                         depth_comparison_rgb), 1)


                # output_frames.append((output * 255).astype(np.uint8))
                imutils.np2png([output], os.path.join(outpath_frames, '%.04d.png' % batch))

            #

            # if args.save_numpy:
            #     if sample["frame_id"].item() in scan_keyframes:
            #         np_list[sample["frame_id"].item()] = depth_out.numpy()
            filename = os.path.join(output_dir, scan, f"{sample['frame_id'].item()}.npy")
            np.save(filename, depth_out.numpy())

        # if args.save_numpy:
        #     np.save(os.path.join(output_dir, f"{scan}"), np_list)

        if args.save_viz:
            video_clip = ImageSequenceClip(output_frames, fps=15)
            video_clip.write_videofile(os.path.join(output_dir, f"{scan}.mp4"), verbose=False, codec='mpeg4',
                                       logger=None, bitrate='2000k')
