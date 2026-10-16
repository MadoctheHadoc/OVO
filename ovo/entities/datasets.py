""" Slightly modified code based on Gaussian-SLAM's datasets
"""
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio

from ..utils.geometry_utils import rasterize_lidar_to_depth


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        resize_ratio = dataset_config.get("resize_ratio", 1.0)
        self.height = int(dataset_config["H"]*resize_ratio)
        self.width = int(dataset_config["W"]*resize_ratio)
        self.fx = dataset_config["fx"]*resize_ratio
        self.fy = dataset_config["fy"]*resize_ratio
        self.cx = dataset_config["cx"]*resize_ratio
        self.cy = dataset_config["cy"]*resize_ratio

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)# added
        color_data = color_data.astype(np.uint8)# added
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        return index, color_data, depth_data, self.poses[index]

class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "color").glob("*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))
        depth_dir = self.dataset_path / "depth"
        # A scene recorded LiDAR-only (e.g. irl_downstairs) has no depth
        # sensor at all: has_depth is False, depth_paths stays empty, and
        # __getitem__ synthesizes a depth image from the LiDAR sweep instead
        # of reading one from disk -- see _decode_frame.
        self.has_depth = depth_dir.is_dir() and any(depth_dir.glob("*.png"))
        self.depth_paths = sorted(list(
            depth_dir.glob("*.png")), key=lambda x: int(os.path.basename(x)[:-4])) if self.has_depth else []
        self.load_poses(self.dataset_path / "pose")
        # Optional per-keyframe LiDAR sweep. When a depth sensor is also
        # present it replaces the depth projection (see VanillaMapper); when
        # it's not, it's the only source of depth. Coverage can be sparse
        # (keyframe_scannet_writer only writes a sweep when the keyframe
        # carries a cloud), so lidar is looked up by exact per-index filename
        # in __getitem__, not by position in a glob list, to keep sparse
        # frames aligned.
        lidar_dir = self.dataset_path / "lidar"
        self.lidar_dir = lidar_dir if lidar_dir.is_dir() else None
        if not self.has_depth and self.lidar_dir is None:
            raise FileNotFoundError(
                f"{self.dataset_path} has neither a depth/ directory nor a lidar/ directory")
        depth_th = dataset_config.get("depth_th",0)
        if depth_th >0:
            self.depth_th = depth_th
        else:
            self.depth_th = None

    @staticmethod
    def _read_pose_txt(path):
        with open(path, "r") as f:
            lines = f.readlines()
        ls = [list(map(float, line.split(' '))) for line in lines]
        return np.array(ls).reshape(4, 4).astype(np.float32)

    def load_poses(self, path):
        pose_paths = sorted(path.glob('*.txt'),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        self.poses = [self._read_pose_txt(p) for p in pose_paths]

    def _decode_frame(self, index, color_path, depth_path, c2w, lidar_path=None):
        """ Shared by `ScanNet` and `ScanNetOnline`: decode a color/depth pair
        (plus optional LiDAR sweep) at already-resolved paths into a frame
        tuple. `depth_path` may be None (scene has no depth sensor), in which
        case `lidar_path` is required and depth is synthesized by rasterizing
        the sweep onto the camera plane -- the same projection
        `VanillaMapper._rasterize_lidar` uses, so a lidar-only scene gets a
        depth image consistent with what the mapper would have built, and
        every consumer of frame depth (semantic association, frame-validity
        gating, mapping) sees the same LiDAR-derived values instead of
        depth-camera data. """
        color_data = cv2.imread(str(color_path))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        lr_color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        lidar_data = None
        if lidar_path is not None and lidar_path.exists():
            lidar_data = np.load(lidar_path).astype(np.float32)

        if depth_path is not None:
            depth_data = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_data = depth_data.astype(np.float32) / self.depth_scale
            if self.depth_th is not None:
                depth_data[depth_data > self.depth_th] = 0
        elif lidar_data is not None:
            depth_data = rasterize_lidar_to_depth(
                lidar_data, self.dataset_config["H"], self.dataset_config["W"],
                self.intrinsics, crop_edge=0, depth_th=self.depth_th)
        else:
            # No depth sensor and no sweep for this keyframe: an all-zero
            # depth makes `missing_depth` (ovomapping.py) skip the frame,
            # same as a dropped depth-camera frame would today.
            depth_data = np.zeros((self.dataset_config["H"], self.dataset_config["W"]), dtype=np.float32)

        edge = self.crop_edge
        if edge > 0:
            lr_color_data = lr_color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        if lidar_data is not None:
            return index, lr_color_data, depth_data, c2w, color_data, lidar_data
        return index, lr_color_data, depth_data, c2w, color_data

    def __getitem__(self, index):
        lidar_path = self.lidar_dir / f"{index:06d}.npy" if self.lidar_dir is not None else None
        depth_path = self.depth_paths[index] if self.has_depth else None
        return self._decode_frame(
            index, self.color_paths[index], depth_path, self.poses[index], lidar_path)


class ScanNetOnline(ScanNet):
    """Reads a ScanNet-style directory while `keyframe_scannet_writer` (or
    equivalent) is still filling it in, instead of globbing a fixed frame
    list once at start-up.

    Frames are located by index on demand: `__getitem__(index)` blocks,
    polling every `poll_interval` seconds, until frame `index`'s color/depth/
    pose files exist. Touch a `STOP` file inside `input_path` to end an
    online run cleanly -- while waiting for a frame, its presence raises
    `StopIteration` instead of blocking forever.
    """

    def __init__(self, dataset_config: dict):
        BaseDataset.__init__(self, dataset_config)
        self.color_dir = self.dataset_path / "color"
        self.depth_dir = self.dataset_path / "depth"
        self.pose_dir = self.dataset_path / "pose"
        self.lidar_dir = self.dataset_path / "lidar"
        self.stop_path = self.dataset_path / "STOP"
        self.poll_interval = dataset_config.get("poll_interval", 0.2)
        # LiDAR-only recording (no depth sensor): the writer never creates
        # depth/, so don't wait on it -- see __getitem__.
        self.has_depth = dataset_config.get("has_depth", True)
        depth_th = dataset_config.get("depth_th", 0)
        self.depth_th = depth_th if depth_th > 0 else None

    def __len__(self):
        highest = -1
        if self.color_dir.is_dir():
            for name in os.listdir(self.color_dir):
                stem, ext = os.path.splitext(name)
                if ext not in (".jpg", ".png") or not stem.isdigit():
                    continue
                highest = max(highest, int(stem))
        return highest + 1

    def __getitem__(self, index):
        color_path = self.color_dir / f"{index:06d}.jpg"
        depth_path = self.depth_dir / f"{index:06d}.png" if self.has_depth else None
        pose_path = self.pose_dir / f"{index:06d}.txt"
        lidar_path = self.lidar_dir / f"{index:06d}.npy"
        required = [color_path, pose_path] + ([depth_path] if self.has_depth else [lidar_path])
        while not all(p.exists() for p in required):
            if self.stop_path.exists():
                raise StopIteration(f"Stop marker found while waiting for frame {index}")
            time.sleep(self.poll_interval)

        c2w = self._read_pose_txt(pose_path)
        return self._decode_frame(index, color_path, depth_path, c2w, lidar_path)


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_projected_depth" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):
        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class Matterport(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "rgb").glob("*.png")))
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.png")))
        self.load_poses(self.dataset_path / "pose")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        poses = os.listdir(path)
        poses.sort()
        self.poses = []
        C = np.eye(4)
        C[1, 1] = -1
        C[2, 2] = -1
        for pose_file in poses:
            c2w = np.loadtxt(str(path / pose_file)).reshape(4, 4).astype(np.float32)

            c2w = np.matmul(c2w, C)
            self.poses.append(c2w.astype(np.float32))
          
    def _load_depth_intrinsics(self, H, W):
        """
        Load the depth camera intrinsics.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """        
        H, W = 720, 1080
        hfov = 90 * np.pi / 180
        vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
        fx = W / (2.0 * np.tan(hfov / 2.0))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        cx = W / 2
        cy = H / 2
        depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return depth_camera_matrix
      
    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)# added
        color_data = color_data.astype(np.uint8)# added
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        return index, color_data, depth_data, self.poses[index]


def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "scannet" :
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == 'matterport':
        return Matterport
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")


def get_online_dataset(dataset_name: str):
    if dataset_name.lower() == "scannet":
        return ScanNetOnline
    raise NotImplementedError(f"Online dataset {dataset_name} not implemented")
