from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random
from einops import rearrange
import json
import os, sys
import math

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb


class THumanDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 prompt_embeds_path: str,
                 num_views: int,
                 bg_color: Any,
                 img_wh: Tuple[int, int],
                 groups_num: int = 1,
                 validation: bool = False,
                 data_view_num: int = 6,
                 num_validation_samples: int = 4,
                 num_samples: Optional[int] = None,
                 invalid_list: Optional[str] = None,
                 trans_norm_system: bool = True,  # if True, transform all normals map into the cam system of front view
                 augment_data: bool = False,
                 read_normal: bool = True,
                 read_color: bool = False,
                 read_depth: bool = False,
                 read_mask: bool = False,
                 mix_color_normal: bool = False,
                 suffix: str = 'png',
                 subscene_tag: int = 3,
                 backup_scene: str = "0033",
                 view_set: int = 0,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.bg_color = bg_color
        self.validation = validation
        self.num_samples = num_samples
        self.trans_norm_system = trans_norm_system
        self.augment_data = augment_data
        self.invalid_list = invalid_list
        self.groups_num = groups_num
        print("augment data: ", self.augment_data)
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_color = read_color
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.mix_color_normal = mix_color_normal  # mix load color and normal maps
        self.suffix = suffix
        self.subscene_tag = subscene_tag

        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses_thuman"
        # default to front view input
        self.view_set = view_set

        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        # subjects = np.loadtxt(os.path.join(self.root_dir, "all.txt"), dtype=str)
        #
        # self.all_objects = []
        # for subject in subjects:
        #     self.all_objects.append(subject)

        self.all_objects = [name for name in os.listdir(self.root_dir) if
                            os.path.isdir(os.path.join(self.root_dir, name))]
        self.all_objects = sorted(self.all_objects)

        self.all_objects = self.all_objects[:666]

        if not validation:
            self.all_objects = self.all_objects[:-num_validation_samples]  # training set
        else:
            self.all_objects = self.all_objects[-num_validation_samples:]

        print("loading ", len(self.all_objects), " objects in the dataset")
        # directly load text prompt embeddings
        # (Nv, 77, 1024)
        self.normal_text_embeds = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
        self.color_text_embeds = torch.load(f'{prompt_embeds_path}/clr_embeds.pt')

    def load_fixed_poses(self):
        poses = {}
        for view in self.view_types:
            RT = np.loadtxt(os.path.join(self.fix_cam_pose_dir, f'{0:03d}_{view}_RT.txt'))
            poses[view] = RT

        return poses

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T  # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_mask(self, img_path, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = np.float32(img > 0)

        assert len(np.shape(img)) == 2

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_image(self, img_path, bg_color, alpha, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.  # [0, 1]
        assert img.shape[-1] == 3 or img.shape[-1] == 4  # RGB or RGBA

        if alpha is None and img.shape[-1] == 4:
            alpha = img[:, :, 3:]
            img = img[:, :, :3]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_depth(self, img_path, bg_color, alpha, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 65535.  # [0, 1]

        img[img > 0.4] = 0
        img = img / 0.4

        assert img.ndim == 2  # depth
        img = np.stack([img] * 3, axis=-1)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        # print(np.max(img[:, :, 0]))

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_normal(self, img_path, bg_color, alpha, RT_w2c=None, RT_w2c_cond=None, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(Image.open(img_path).resize(self.img_wh))

        assert normal.shape[-1] == 3 or normal.shape[-1] == 4, "Format not implemented, normal map must be RGB or RGBA"

        if alpha is None and normal.shape[-1] == 4:
            alpha = normal[:, :, 3:] / 255.
            normal = normal[:, :, :3]

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)

        img = (normal * 0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def __len__(self):
        return len(self.all_objects)

    def __getitem__(self, index):
        object_name = self.all_objects[index % len(self.all_objects)]
        set_idx = 0
        set_idx = self.view_set

        if self.augment_data:
            cond_view = random.sample(self.view_types, k=1)[0]
        else:
            cond_view = 'front'

        view_types = self.view_types

        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in view_types]

        elevations = []
        azimuths = []

        # get the bg color
        bg_color = self.get_bg_color()
        assert (self.read_color and (self.read_normal or self.read_depth)) is True, \
            "Must read normal or depth other than color"
        # input(front view) rgb+mask
        if self.read_mask:
            cond_alpha = self.load_mask(os.path.join(
                self.root_dir, f"{object_name}/model", f"mask_f{set_idx:03d}_{cond_view}.{self.suffix}"
            ), return_type='np')
        else:
            cond_alpha = None
        # front view img
        img_tensors_in = [self.load_image(os.path.join(
            self.root_dir, f"{object_name}/model", f"rgb_{set_idx:03d}_{cond_view}.{self.suffix}"
        ), bg_color, cond_alpha, return_type='pt').permute(2, 0, 1)] * self.num_views

        img_tensors_out = []
        normal_tensors_out = []
        normal_tensors_out_smplx = []

        for view, tgt_w2c in zip(view_types, tgt_w2cs):
            img_path = os.path.join(self.root_dir, f"{object_name}/model", f"rgb_{set_idx:03d}_{view}.{self.suffix}")
            mask_path = os.path.join(self.root_dir, f"{object_name}/model", f"mask_{set_idx:03d}_{view}.{self.suffix}")
            normal_path = os.path.join(self.root_dir, f"{object_name}/model",
                                       f"normals_{set_idx:03d}_{view}.{self.suffix}")
            depth_path = os.path.join(self.root_dir, f"{object_name}/model",
                                      f"depth_{set_idx:03d}_{view}.{self.suffix}")

            normal_smplx_path = os.path.join(self.root_dir,
                                             f"{object_name}/smplx/normals_{set_idx:03d}_{view}.{self.suffix}")

            if self.read_mask:
                alpha = self.load_mask(mask_path, return_type='np')
            else:
                alpha = None

            if self.read_color:
                img_tensor = self.load_image(img_path, bg_color, alpha, return_type="pt")
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensors_out.append(img_tensor)

            if self.read_normal:
                # object mv normals
                normal_tensor = self.load_normal(normal_path, bg_color, alpha, RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c,
                                                 return_type="pt").permute(2, 0, 1)
                normal_tensors_out.append(normal_tensor)
                # smplx obj mv normals
                normal_tensor_smplx = self.load_normal(normal_smplx_path, bg_color, alpha=None, RT_w2c=tgt_w2c,
                                                       RT_w2c_cond=cond_w2c,
                                                       return_type="pt").permute(2, 0, 1)
                normal_tensors_out_smplx.append(normal_tensor_smplx)

            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        if self.read_color:
            img_tensors_out = torch.stack(img_tensors_out, dim=0).float()  # (Nv, 3, H, W)
        if self.read_normal:
            normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float()  # (Nv, 3, H, W)
            normal_tensors_out_smplx = torch.stack(normal_tensors_out_smplx, dim=0).float()  # (Nv, 3, H, W)

        elevations = torch.as_tensor(np.asarray(elevations)).float().squeeze(1)
        azimuths = torch.as_tensor(np.asarray(azimuths)).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()

        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1)  # (Nv, 3)

        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class] * self.num_views, dim=0)  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_views, dim=0)  # (Nv, 2)

        return {
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'normals_out': normal_tensors_out,
            'normal_text_embeds': self.normal_text_embeds,
            'color_text_embeds': self.color_text_embeds
        }


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.num_datasets = len(datasets)

    def __getitem__(self, i):
        chosen = random.choices(self.datasets, self.weights, k=1)[0]
        return chosen[i]

    def __len__(self):
        return max(len(d) for d in self.datasets)


if __name__ == "__main__":

    train_dataset = THumanDataset(
        # root_dir="/media/pawting/SN640/Datasets/wonder3d_dev/thuman2/ortho135_res1024",
        root_dir="/media/pawting/SN640/Datasets/wonder3d_dev/thuman2/ortho1_res1024_viewstep5",
        num_views=6,
        bg_color="white",
        img_wh=(512, 512),
        read_color=True,
        read_normal=True,
        read_mask=False,
        read_depth=False,
        view_set=30
    )

    data_0 = train_dataset[0]
    for k, v in data_0.items():
        print(k, v.shape)

    # img_out = rearrange(np.asarray(data_0["imgs_out"][0]), "c h w -> h w c")
    # normals_smplx = rearrange(np.asarray(data_0["normals_smplx"][0]), "c h w -> h w c")
    for view in range(6):
        cv2.imwrite(f"/media/pawting/SN640/hello_worlds/Era3D/tmp/img_out_{view}_.jpg",
                    rearrange(np.asarray(data_0["normals_out"][view]), "c h w -> h w c") * 255)
    print(data_0["imgs_out"].size())
    # print(data_0["imgs_out"])
    # print(data_0["imgs_in"])

"""
# output per subject
# elevations
elevations_cond tensor([0., 0., 0., 0., 0., 0.])
elevations_cond_deg tensor([0., 0., 0., 0., 0., 0.])
elevations tensor([ 0.0000, -0.2362, -0.1686,  0.5220,  0.6907,  0.3733])
elevations_deg tensor([  0.0000, -13.5356,  -9.6611,  29.9105,  39.5716,  21.3893])

# azimuths_deg
azimuths_deg tensor([  0.0000,  46.5458,  97.0298, 180.0000, 277.0298, 320.0548])
azimuths tensor([0.0000, 0.8124, 1.6935, 3.1416, 4.8351, 5.5860])

# imgs_in 
shape(num_views, 3, h, w)

# imgs_out 
    # mix mode
        # 每次组成一组多视角normal或color
        shape(num_views, 3, h, w)
    # joint mode
        shape(num_views, 3, h, w)
        
# normal_out
# joint mode only
shape(num_views, 3, h, w)

# task
    # mix mode
        task_embeddings 
        tensor([[1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.]])
    # joint mode        
        normal_task_embeddings 
        tensor([[1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.]])
        color_task_embeddings 
        tensor([[0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.]])

# camera
camera_embeddings 
tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.0000, -0.2362,  0.8124],
        [ 0.0000, -0.1686,  1.6935],
        [ 0.0000,  0.5220,  3.1416],
        [ 0.0000,  0.6907,  4.8351],
        [ 0.0000,  0.3733,  5.5860]])

"""

"""
# shapes
elevations_cond         torch.Size([6])
elevations              torch.Size([6])
azimuths                torch.Size([6])
elevations_cond_deg     torch.Size([6])
elevations_deg          torch.Size([6])
azimuths_deg            torch.Size([6])

imgs_in                 torch.Size([6, 3, 256, 256])
imgs_out                torch.Size([6, 3, 256, 256])
normals_out             torch.Size([6, 3, 256, 256])
normals_smplx           torch.Size([6, 3, 256, 256])

camera_embeddings       torch.Size([6, 3])
normal_task_embeddings  torch.Size([6, 2])
color_task_embeddings   torch.Size([6, 2])
"""
