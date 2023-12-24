import os
import os.path as osp
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from itertools import compress
import pykitti
from abc import abstractmethod

import re
def microsoft_sorted(input_list):
    return sorted(input_list, key=lambda s: [int(s) if s.isdigit() else s for s in sum(re.findall(r'(\D+)(\d+)', 'a'+s+'0'), ())])

def pose_7dof_to_matrix(poses_7dof):
    poses_t = poses_7dof[:, :3, None]
    poses_quat = R.from_quat(poses_7dof[:, 3:]).as_matrix()
    poses_matrix = np.concatenate((poses_quat, poses_t), axis=2)
    poses_one = np.repeat(np.array([[[0, 0, 0, 1]]]), poses_matrix.shape[0], axis=0)
    poses_matrix = np.concatenate((poses_matrix, poses_one), axis=1)
    return poses_matrix

class BaseLoader:
    def __init__(self, root, return_full=True):
        """
        The BaseLoader supposes the numbers of poses and intrinsics are the same as that of rgb's.
        """
        self.root = root
        self.return_full = return_full
        self.img_file_type = ['.png', '.jpg']
        self.np_file_type = ['.npy']
        self.init_data_info()

    @abstractmethod
    def init_data_info(self):
        self.rgb_suffix = ...
        self.depth_suffix = ...
        self.depth_scale = ...
        self.scene_names = ...

    ################ main function ################

    @abstractmethod
    def get_root_info(self, scene_name):
        self.rgb_root = osp.join(self.root, scene_name, 'rgb')
        self.depth_root = osp.join(self.root, scene_name, 'depth')

    def obtain_rgb_depth_intrinsic_pose(self, scene_name, sample_indexes=None):
        """
        sample_indexes: The sampled index of all data before valid filtering.
        """
        self.scene_name = scene_name
        self.get_root_info(scene_name)
    
        self.info_dict = {}
        self.info_dict.update(self.get_image_ids())
        # 1. read data and filter invalid pairs
        self.info_dict.update(self.get_data_filtered_full()) 
        # 2. sample partial image pairs
        filtered_image_ids = self.info_dict['filtered_image_ids']
        sampled_ids = self.get_sample_ids(sample_indexes)
        sampled_filtered_ids = microsoft_sorted(list(set(filtered_image_ids).intersection(set(sampled_ids))))
        self.info_dict['sampled_ids'] = sampled_ids
        self.info_dict['sampled_filtered_ids'] = sampled_filtered_ids
        sample_filtered_indexes = [filtered_image_ids.index(id) for id in sampled_filtered_ids]
        self.info_dict.update(self.sample_data_from_filtered(sample_filtered_indexes))
        # 3. record the valid_mask_full of the sampled prediction depth without filtering invalid gt_depth
        self.info_dict['valid_mask_sampled_pred'] = [sampled_ids.index(id) for id in sampled_filtered_ids]
        
        if not self.return_full:
            del self.info_dict['images_full']
            del self.info_dict['depths_full']
            del self.info_dict['intrinsic_full']
            del self.info_dict['poses_full']
        
        return self.info_dict

    def get_sample_ids(self, sample_indexes):
        if sample_indexes is None:
            sampled_ids = self.info_dict['image_ids']
        else:
            sampled_ids = [self.info_dict['image_ids'][i] for i in sample_indexes]
        return sampled_ids

    def get_image_ids(self):
        # RGB image_ids
        image_ids = []
        for file_name in microsoft_sorted(os.listdir(self.rgb_root)):
            if file_name.endswith(self.rgb_suffix):
                img_id = osp.splitext(file_name)[0]
                image_ids.append(img_id)
        return dict(
            image_ids=image_ids,
            image_ids_all=image_ids,
        )
    
    ####### get_data_filtered_full begin #########
    def get_data_filtered_full(self):
        # paths of rgbs and depths
        rgb_paths = [osp.join(self.rgb_root, image_id + self.rgb_suffix) for image_id in self.info_dict['image_ids']]
        depth_paths = [osp.join(self.depth_root, image_id + self.depth_suffix) for image_id in self.info_dict['image_ids']]
        # poses and intrinsics
        poses = self.get_poses_full()
        intrinsics = self.get_intrinsics_full()

        # valid_mask
        valid_mask_rgb = np.array([osp.exists(path) for path in rgb_paths])
        valid_mask_depth = np.array([osp.exists(path) for path in depth_paths])
        valid_mask_pose = (~np.isnan(poses.reshape(-1, 16).sum(axis=1)))
        valid_mask_intrinsic = (~np.isnan(intrinsics.reshape(-1, 9).sum(axis=1)))

        assert valid_mask_rgb.shape == valid_mask_depth.shape == valid_mask_pose.shape == valid_mask_intrinsic.shape
        valid_mask = valid_mask_rgb & valid_mask_depth & valid_mask_pose & valid_mask_intrinsic # [n]

        # read data and filter
        images = np.stack([self.load_data(path, is_rgb_img=True) for (i, path) in enumerate(rgb_paths) if valid_mask[i]], axis=0)
        depths = np.stack([self.load_data(path) for (i, path) in enumerate(depth_paths) if valid_mask[i]], axis=0)
        depths = self.mask_depth_invalid_values(depths)
        depths = depths / self.depth_scale
        poses = poses[valid_mask]
        intrinsics = intrinsics[valid_mask]

        return dict(
            images_full=images,
            depths_full=depths,
            poses_full=poses,
            intrinsics_full=intrinsics,
            valid_mask_full=valid_mask,
            filtered_image_ids=[id for i, id in enumerate(self.info_dict['image_ids']) if valid_mask[i]]
        )

    def mask_depth_invalid_values(self, depths):
        return depths
   
    @abstractmethod
    def get_poses_full(self):
        '''
        return np.array poses of camera_to_world, with the shape of [n, 4, 4]
        '''
        poses_full = ... 
        return poses_full
    
    @abstractmethod
    def get_intrinsics_full(self):
        '''
        return np.array intrinsics, with the shape of [n, 3, 3]
        '''
        intrinsics_full = ... # array, [n, 3, 3]
        return intrinsics_full
    
    ####### get_data_filtered_full end #########

    def sample_data_from_filtered(self, indexes):
        return dict(
            images=self.info_dict['images_full'][indexes],
            depths=self.info_dict['depths_full'][indexes],
            poses=self.info_dict['poses_full'][indexes],
            intrinsics=self.info_dict['intrinsics_full'][indexes],
        )
        
    ########### tools begin ############
    
    def get_demo_depth(self):
        self.get_root_info(self.scene_names[0])
        for file_name in os.listdir(self.depth_root):
            if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.JPG'):
                demo_depth_path = osp.join(self.depth_root, file_name)
                try:
                    demo_depth = self.load_data(demo_depth_path) / self.depth_scale
                    break
                except:
                    continue
        return demo_depth

    def load_data(self, path: str, is_rgb_img: bool=False):
        if not osp.exists(path):
            raise RuntimeError(f'{path} does not exist.')

        data_type = osp.splitext(path)[-1]
        if data_type in self.img_file_type:
            if is_rgb_img:
                data = cv2.cvtColor(
                    cv2.imread(path),
                    cv2.COLOR_BGR2RGB
                    )
            else:
                data = cv2.imread(path, -1)
        elif data_type in self.np_file_type:
            data = np.load(path)
        else:
            raise RuntimeError(f'{data_type} is not supported in current version.')
        
        return data

    ########### tools end ############



class NYUDepthVideo_Loader(BaseLoader):
    def __init__(self, root, return_full=True):
        super().__init__(root, return_full)
    
    def init_data_info(self):
        self.rgb_suffix = '.jpg'
        self.depth_suffix = '.png'
        self.depth_scale = 5000

        scene_names = []
        for scene_name in microsoft_sorted(os.listdir(self.root)):
            scene_root = osp.join(self.root, scene_name)
            if not osp.isdir(scene_root):
                continue
            if scene_name == 'annotations':
                continue
            scene_names.append(scene_name)
        self.scene_names = scene_names

    def get_root_info(self, scene_name):
        self.rgb_root = osp.join(self.root, scene_name, 'rgb')
        self.depth_root = osp.join(self.root, scene_name, 'depth')
    
    def get_poses_full(self):
        try:
            poses_full = np.loadtxt(osp.join(self.root, self.scene_name, 'poses.txt')).reshape(-1, 3, 4)
        except:
            poses_full = np.loadtxt(osp.join(self.root, self.scene_name, 'rgb/poses.txt')).reshape(-1, 3, 4)
        poses_ones_full = np.repeat(np.array([[[0, 0, 0, 1]]]), poses_full.shape[0], axis=0)
        poses_full = np.concatenate((poses_full, poses_ones_full), axis=1)
        return poses_full
    
    def get_intrinsics_full(self):
        try:
            intrinsic_full = np.loadtxt(osp.join(self.root, self.scene_name, 'cam.txt'))[:3, :3]
        except:
            intrinsic_full = np.loadtxt(osp.join(self.root, self.scene_name, 'rgb/cam.txt'))[:3, :3]
        intrinsic_full = np.repeat(intrinsic_full[None], len(self.info_dict['image_ids']), axis=0)
        return intrinsic_full
    

class Scannet_test_Loader(BaseLoader):
    def __init__(self, root, return_full=True):
        super().__init__(root, return_full)
    
    def init_data_info(self):
        self.rgb_suffix = '.jpg'
        self.depth_suffix = '.png'
        self.depth_scale = 1000

        scene_names = []
        for scene_name in microsoft_sorted(os.listdir(self.root)):
            scene_root = osp.join(self.root, scene_name)
            if not osp.isdir(scene_root):
                continue
            scene_names.append(scene_name)
        self.scene_names = scene_names

    def get_root_info(self, scene_name):
        self.rgb_root = osp.join(self.root, scene_name, 'color')
        self.depth_root = osp.join(self.root, scene_name, 'depth')
    
    def get_poses_full(self):
        pose_root = osp.join(self.root, self.scene_name, 'pose')

        poses_full = []
        image_ids = self.info_dict['image_ids']
        for i in range(len(image_ids)):
            pose_i = np.loadtxt(osp.join(pose_root, '%s.txt' %image_ids[i]))
            poses_full.append(pose_i)
        poses_full = np.stack(poses_full, axis=0)

        return poses_full
    
    def get_intrinsics_full(self):
        intrinsic_full = np.loadtxt(osp.join(self.root, self.scene_name, 'intrinsic/intrinsic_color.txt'))[:3, :3]
        intrinsic_full = np.repeat(intrinsic_full[None], len(self.info_dict['image_ids']), axis=0)
        return intrinsic_full


class SevenScenes_Loader(BaseLoader):
    def __init__(self, root, return_full=True):
        super().__init__(root, return_full)
    
    def init_data_info(self):
        self.rgb_suffix = '.png'
        self.depth_suffix = '.png'
        self.depth_scale = 1000

        scene_names = []
        for scene_name in microsoft_sorted(os.listdir(self.root)):
            scene_root = osp.join(self.root, scene_name)
            if not osp.isdir(scene_root):
                continue
            scene_names.append(scene_name)
        self.scene_names = scene_names

    def get_root_info(self, scene_name):
        self.rgb_root = osp.join(self.root, scene_name, 'rgb')
        self.depth_root = osp.join(self.root, scene_name, 'depth')
    
    def mask_depth_invalid_values(self, depths):
        depths[depths == 65535] = 0
        return depths
    
    def get_poses_full(self):
        pose_root = osp.join(self.root, self.scene_name, 'pose')
        image_ids = self.info_dict['image_ids']

        poses_full = []
        for i in range(len(image_ids)):
            pose_i = np.loadtxt(osp.join(pose_root, '%s.txt' %image_ids[i]))
            poses_full.append(pose_i)
        poses_full = np.stack(poses_full, axis=0)

        return poses_full
    
    def get_intrinsics_full(self):
        image_ids = self.info_dict['image_ids']
        fx = 585; fy = 585; cx = 320; cy = 240
        intrinsic_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
        intrinsic_full = np.repeat(intrinsic_full[None], len(image_ids), axis=0)
        return intrinsic_full


class TUM_Loader(BaseLoader): 
    def __init__(self, root, return_full=True):
        super().__init__(root, return_full)
    
    def init_data_info(self):
        self.rgb_suffix = '.png'
        self.depth_suffix = '.png'
        self.depth_scale = 5000

        scene_names = []
        for scene_name in microsoft_sorted(os.listdir(self.root)):
            scene_root = osp.join(self.root, scene_name)
            if not osp.isdir(scene_root):
                continue
            scene_names.append(scene_name)
        self.scene_names = scene_names

    def get_root_info(self, scene_name):
        self.rgb_root = osp.join(self.root, scene_name, 'rgb')
        self.depth_root = osp.join(self.root, scene_name, 'depth')
    
    def get_sample_ids(self, sample_indexes):
        if sample_indexes is None:
            sampled_ids = self.info_dict['image_ids']
        else:
            sampled_ids = [self.info_dict['image_ids_all'][i] for i in sample_indexes]
        return sampled_ids

    def get_image_ids(self):
        rgb_depth_pose_match_path = osp.join(self.root, self.scene_name, 'rgb_depth_pose.txt')
        with open(rgb_depth_pose_match_path, 'r') as f:
            rgb_depth_pose_list = f.readlines()

        # RGB image_ids
        rgb_image_ids_all = []
        for file_name in microsoft_sorted(os.listdir(self.rgb_root)):
            if file_name.endswith(self.rgb_suffix):
                img_id = osp.splitext(file_name)[0]
                rgb_image_ids_all.append(img_id)

        rgb_image_ids = []
        depth_image_ids = []
        poses_full = []
        for line in rgb_depth_pose_list:
            line = line.split(' ')
            rgb_id = osp.splitext(line[1].split('/')[-1])[0]
            depth_id = osp.splitext(line[3].split('/')[-1])[0] 
            pose = line[-7:]
            pose = pose_7dof_to_matrix(np.array(pose, np.float32)[None, ...])[0]
            rgb_image_ids.append(rgb_id)
            depth_image_ids.append(depth_id)
            poses_full.append(pose)
        poses_full = np.stack(poses_full, axis=0)
        
        return dict(
            image_ids_all=rgb_image_ids_all,
            image_ids=rgb_image_ids,
            depth_image_ids=depth_image_ids,
            poses_full=poses_full,
        )

    def get_data_filtered_full(self):
        # paths of rgbs and depths
        rgb_paths = [osp.join(self.rgb_root, image_id + self.rgb_suffix) for image_id in self.info_dict['image_ids']]
        depth_paths = [osp.join(self.depth_root, image_id + self.depth_suffix) for image_id in self.info_dict['depth_image_ids']]
        # poses and intrinsics
        poses = self.get_poses_full()
        intrinsics = self.get_intrinsics_full()

        # valid_mask
        valid_mask_rgb = np.array([osp.exists(path) for path in rgb_paths])
        valid_mask_depth = np.array([osp.exists(path) for path in depth_paths])
        valid_mask_pose = (~np.isnan(poses.reshape(-1, 16).sum(axis=1)))
        valid_mask_intrinsic = (~np.isnan(intrinsics.reshape(-1, 9).sum(axis=1)))
        assert valid_mask_rgb.shape == valid_mask_depth.shape == valid_mask_pose.shape == valid_mask_intrinsic.shape
        valid_mask = valid_mask_rgb & valid_mask_depth & valid_mask_pose & valid_mask_intrinsic # [n]

        # read data and filter
        images = np.stack([self.load_data(path, is_rgb_img=True) for (i, path) in enumerate(rgb_paths) if valid_mask[i]], axis=0)
        depths = np.stack([self.load_data(path) for (i, path) in enumerate(depth_paths) if valid_mask[i]], axis=0) / self.depth_scale
        poses = poses[valid_mask]
        intrinsics = intrinsics[valid_mask]

        return dict(
            images_full=images,
            depths_full=depths,
            poses_full=poses,
            intrinsics_full=intrinsics,
            valid_mask_full=valid_mask,
            filtered_image_ids=[id for i, id in enumerate(self.info_dict['image_ids']) if valid_mask[i]],
        )
    
    def get_poses_full(self):
        poses = self.info_dict['poses_full']
        return poses
    
    def get_intrinsics_full(self):
        image_ids = self.info_dict['image_ids']
        fx = 525; fy = 525; cx = 319.5; cy = 239.5
        intrinsic_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
        intrinsic_full = np.repeat(intrinsic_full[None], len(image_ids), axis=0)
        return intrinsic_full
    

class KITTIDepthVideo_Loader(BaseLoader):
    def __init__(self, root, scene_names_list, return_full=True):
        super().__init__(root, return_full)
        self.scene_names = scene_names_list
    
    def init_data_info(self):
        self.rgb_suffix = '.png'
        self.depth_suffix = '.png'
        self.depth_scale = 256

    def get_root_info(self, scene_name):
        scene_date = scene_name.split('_drive')[0]
        self.rgb_root = osp.join(self.root, 'raw_data', scene_date, scene_name, "image_02/data")
        self.depth_root = osp.join(self.root, 'depth_annotated', scene_name, "proj_depth/groundtruth/image_02/")
    
    def get_poses_full(self):
        image_ids = self.info_dict['image_ids']
        scene_date = self.scene_name.split('_drive')[0]
        scene_drive = self.scene_name.split('_drive_')[-1].split('_sync')[0]
        self.pykitti_loader = pykitti.raw(osp.join(self.root, 'raw_data'), scene_date, scene_drive, frame_range=range(len(image_ids)))

        T_cam2_imu = self.pykitti_loader.calib.T_cam2_imu
        oxts = self.pykitti_loader.oxts
        poses_full = []
        for i, oxt in enumerate(oxts):
            T_w_imu = oxt.T_w_imu
            poses_full.append(T_w_imu @ np.linalg.inv(T_cam2_imu))
        poses_full = np.stack(poses_full, axis=0)

        return poses_full
    
    def get_intrinsics_full(self):
        image_ids = self.info_dict['image_ids']
        intrinsic_full = self.pykitti_loader.calib.K_cam2
        intrinsic_full = np.repeat(intrinsic_full[None], len(image_ids), axis=0)
        return intrinsic_full
    