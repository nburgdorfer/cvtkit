import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys

from cvt.io import read_pfm
from cvt.camera import compute_baselines, _intrinsic_pyramid, scale_cam, crop_cam
from cvt.common import _build_depth_pyramid, _normalize_image, crop_image

def build_dataset(cfg, mode, scenes):
    if cfg["dataset"] == 'DTU':
        from cvt.datasets.DTU import DTU as Dataset
    else:
        raise Exception(f"Unknown Dataset {self.cfg['dataset']}")

    return Dataset(cfg, mode, scenes)

class BaseDataset(data.Dataset):
    def __init__(self, cfg, mode, scenes):
        self.cfg = cfg
        self.mode = mode
        self.data_path = self.cfg["data_path"]
        self.device = self.cfg["device"]
        self.scenes = scenes
        self.crop_h = self.cfg["camera"]["crop_h"]
        self.crop_w = self.cfg["camera"]["crop_w"]

        try:
            self.resolution_levels = len(self.cfg["model"]["feature_channels"])
        except:
            self.resolution_levels = 1

        if self.mode=="inference":
            self.num_frame = self.cfg["inference"]["num_frame"]
            self.frame_spacing = self.cfg["inference"]["frame_spacing"]
            self.scale = self.cfg["inference"]["scale"]
            self.sample_mode = self.cfg["inference"]["sample_mode"]
            self.random_crop = False
        else:
            self.num_frame = self.cfg["training"]["num_frame"]
            self.frame_spacing = self.cfg["training"]["frame_spacing"]
            self.scale = self.cfg["training"]["scale"]
            self.sample_mode = self.cfg["training"]["sample_mode"]
            self.random_crop = self.cfg["training"]["random_crop"]
        self.random_clusters = (mode=="training")

        self.K = {}
        self.build_samples()

        if self.mode == "inference":
            # use all samples during inference
            self.samples = self.total_samples
        else:
            # shuffle and sub-sample during training
            if self.mode=="training":
                self.max_samples = self.cfg["training"]["max_training_samples"]
            elif self.mode=="validation":
                self.max_samples = self.cfg["training"]["max_val_samples"]
            self.shuffle_and_subsample()

    def build_samples(self, frame_spacing):
        raise NotImplementedError()

    def load_intrinsics(self):
        raise NotImplementedError()

    def get_pose(self, frame_id):
        raise NotImplementedError()

    def get_image(self, image_file, scale=True):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop and resize image
        h,w,_ = image.shape
        image = image[(self.crop_h//2):h-(self.crop_h//2),(self.crop_w//2):w-(self.crop_w//2), :]
        if scale:
            image = cv2.resize(image, (self.W,self.H), interpolation=cv2.INTER_LINEAR)
        image = _normalize_image(image, mean=0.5, std=0.5)
        image = np.moveaxis(image, [0,1,2], [1,2,0])
        return image.astype(np.float32)

    def get_depth(self, depth_file, scale=True):
        if (depth_file[-3:] == "pfm"):
            depth = read_pfm(depth_file)
        elif (depth_file[-3:] == "png"):
            depth = cv2.imread(depth_file, 2) / self.png_depth_scale

        # crop and resize depth
        h,w = depth.shape
        depth = depth[(self.crop_h//2):h-(self.crop_h//2),(self.crop_w//2):w-(self.crop_w//2)]
        if scale:
            depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_LINEAR)
        depth = depth.reshape(1, depth.shape[0], depth.shape[1])
        return depth.astype(np.float32)

    def shuffle_and_subsample(self):
        np.random.shuffle(self.total_samples)
        num_samples = min(len(self.total_samples), self.max_samples)
        self.samples = self.total_samples[:num_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        sample = self.samples[idx]
        scene = sample["scene"]

        # load and compute intrinsics
        K = np.copy(self.K[scene])

        # random crop scaled image patches or scale entire image
        if self.random_crop:
            crop_row = np.random.randint(0, (self.cfg["camera"]["height"] - self.crop_h) - self.H)
            crop_col = np.random.randint(0, (self.cfg["camera"]["width"]- self.crop_w) - self.W)
            K = crop_cam(K, crop_row, crop_col)
        else:
            K = scale_cam(K, scale=self.scale)

        images = [None]*self.num_frame
        poses = [None]*self.num_frame
        target_depths = [None]*self.num_frame
        filenames = []
        for i, fid in enumerate(sample["frame_inds"]):
            images[i] = self.get_image(sample["image_files"][i], scale=(not self.random_crop))
            poses[i] = self.get_pose(sample["pose_files"][i], fid)
            target_depths[i] = self.get_depth(sample["depth_files"][i], scale=(not self.random_crop))
            filenames.append(scene + '-' + '_'.join('%04d' % x for x in sample["frame_inds"]))

            if self.random_crop:
                images[i] = crop_image(images[i], crop_row, crop_col, self.scale)
                target_depths[i] = crop_image(target_depths[i], crop_row, crop_col, self.scale)
        images = np.asarray(images, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        target_depths = np.asarray(target_depths, dtype=np.float32)

        # compute min and max camera baselines
        min_baseline, max_baseline = compute_baselines(poses)

        # load data dict
        data = {}
        data["ref_id"] = int(sample["frame_inds"][0])
        data["K"] = K
        data["images"] = images
        data["poses"] = poses
        data["target_depth"] = target_depths[0]
        if self.cfg["camera"]["baseline_mode"] == "min":
            data["baseline"] = min_baseline
        elif self.cfg["camera"]["baseline_mode"] == "max":
            data["baseline"] = max_baseline

        ## Scaling intrinsics for the feature pyramid
        if self.resolution_levels > 1:
            data["target_depths"] = _build_depth_pyramid(data["target_depth"][0], levels=self.resolution_levels)

            multires_intrinsics = []
            for i in range(self.num_frame):
                multires_intrinsics.append(_intrinsic_pyramid(K, self.resolution_levels)[::-1])
            
            data["multires_intrinsics"] = np.stack(multires_intrinsics).astype(np.float32)

        return data
