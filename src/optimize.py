import numpy as np
import torch
import torch.nn.functional as F

import os
import os.path as osp
import time
import cv2
import re
import shutil
import time
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from rgbd_fusion.tsdf_fusion import TSDFFusion
from utils.lwlr_function_batch import sparse_depth_lwlr_batch, fill_grid_sparse_depth_torch_batch
from utils.utils import extract_frames, Excel_Editor, FeatureSimComputer, image_indexes_from_sim_matrix, torch_det_2x2, \
                        torch_inverse_3x3, img_numpy_to_torch, microsoft_sorted
from dataset_loader.gt_dataset_loader import NYUDepthVideo_Loader, Scannet_test_Loader, SevenScenes_Loader, TUM_Loader, KITTIDepthVideo_Loader
from utils.utils import pose_vec2mat

try:
    from mmseg.apis import inference_segmentor, init_segmentor
    mmseg_import_flag = True
except:
    mmseg_import_flag = False

try:
    from lietorch import SE3
    import lietorch
    lietorch_flag = True
except:
    lietorch_flag = False

print('mmseg_import_flag :', mmseg_import_flag)
print('lietorch_flag :', lietorch_flag)

def depth_project(coords_x, coords_y, tgt_depth, intrinsics, poses, coords_mask, shape_new):
    coords_ones = torch.ones_like(coords_x)
    cam_coords = torch.stack([coords_x, coords_y, coords_ones], dim=1) # [n, 3, sample_num]
    intrinsics_inv = torch_inverse_3x3(intrinsics)
    cam_coords = (intrinsics_inv @ cam_coords.double()) # [n, 3, sample_num]
    cam_coords *= tgt_depth[coords_mask[:, None, :]].view(cam_coords.shape[0], 1, -1)
    
    proj_cam_to_src_pixel = intrinsics @ poses[:, :3].double()  # [n, 3, 4]
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    pcoords = rot @ cam_coords + tr

    depth_computed_points = pcoords[:, 2].clamp(min=1e-3) # [B]

    projected_x = pcoords[:, 0]
    projected_x = 2 * (projected_x / depth_computed_points.clone())/(shape_new[1]-1) - 1
    projected_y = pcoords[:, 1]
    projected_y = 2 * (projected_y / depth_computed_points.clone())/(shape_new[0]-1) - 1
    
    projected_x[depth_computed_points == tr[:, 2]] = 2
    projected_y[depth_computed_points == tr[:, 2]] = 2

    return depth_computed_points, projected_x, projected_y

def get_pose_prob_torch(angle, thr):
    angle[angle != angle] = 0 # deal with nan
    angle[angle > thr] = 0
    angle = (thr/2 - (angle - thr/2).abs())
    return angle

def get_ref_ids(tgt_ids, num_imgs, near_frames_num, angle=None, epoch=None, sample_ref_num=1, angle_thr=None):
    length = tgt_ids.shape[0]
    if epoch == 0:
        p = torch.ones((length, (2 * near_frames_num + 1)), dtype=torch.float)
        p[:, near_frames_num] = 0 # do not sample tgt_id
        ref_ids_idx = torch.multinomial(p, sample_ref_num, replacement=False) # [n, sample_num]
        ref_ids_near = torch.stack([tgt_ids + i for i in range(-near_frames_num, (near_frames_num+1))], dim=1) # [n, near_num]
        ref_ids = ref_ids_near[torch.arange(length)[:, None].repeat(1, sample_ref_num), ref_ids_idx]
    else:
        ref_ids_all = torch.arange(num_imgs)[None, :].repeat(length, 1).cuda()
        tgt_ids_repeat = tgt_ids[:, None].repeat(1, num_imgs)

        if angle is None:
            p = torch.ones((length, ref_ids_all.shape[0]), dtype=torch.float) # average sampling
        else:
            p = get_pose_prob_torch(angle[tgt_ids_repeat, ref_ids_all], thr=angle_thr)
            p = torch.sqrt(p)
        
        p[torch.arange(length), tgt_ids] = 0
        p = p.float()
        p_distance = (ref_ids_all - tgt_ids_repeat).abs()
        near_mask = (p_distance <= near_frames_num)

        for i in range(length):
            if p[i].sum() == 0 or p[i, ~near_mask[i]].sum() == 0:
                p[i, near_mask[i]] = (1 / (near_mask[i].sum())).float()
            else:
                p[i, ~near_mask[i]] = (p[i, ~near_mask[i]] / p[i, ~near_mask[i]].sum()).float() / 2
                p[i, near_mask[i]] = (1 / (near_mask[i].sum())).float() / 2

        p = p / p.sum()
        ref_ids = torch.multinomial(p, sample_ref_num, replacement=False) # [n, sample_num], omit the index step

    return ref_ids



class FrozenRecon:
    def __init__(self, args) -> None:

        # 1. define parameters and file paths
        self.args = args
        ## optimize from video or rgb images.
        self.scene_name = self.args.scene_name
        ## file paths
        self.suffix = self.args.save_suffix
        self.output_root = osp.join(self.args.outputs, self.scene_name + self.suffix)
        self.data_root = osp.join(self.args.data_root, self.scene_name)
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)
        self.rgb_root = osp.join(self.data_root, 'rgb')
        self.save_path = osp.join(self.output_root, 'optimized_params.npy')
        self.outputs = {'times':{}}
        ## path of excel for saving results
        excel_title = ['scene_name', 'extract_time', 'inference_time', 'optimize_time', 'total_time']
        self.excel = Excel_Editor('optimize_time', excel_title)
        self.workbook_save_path = osp.join(self.output_root, 'optimize_time.xls')
        
        # 2. preprocess, including extract RGB frames, recording image_ids and rgb_paths, and generating LeReS depth.
        self.preprocess()

    def preprocess(self):
        time1 = time.time()

        shutil.copyfile(__file__, osp.join(self.output_root, __file__.split('/')[-1]))

        # 1. extract frames from a pose-free video
        extracted_flag = osp.join(self.data_root, 'extracted_flag.txt')
        if not os.path.exists(extracted_flag):
            data_root_rgb = osp.join(self.data_root, 'rgb')
            output_root_rgb = osp.join(self.output_root, 'rgb')
            if self.args.video_path is not None:
                print('extracting video frames ...')
                extract_frames(self.args.video_path, self.rgb_root, self.args.frames_downsample_ratio)
                shutil.copytree(data_root_rgb, output_root_rgb)
            elif self.args.img_root is not None:
                assert self.args.scene_name is not None
                if osp.exists(data_root_rgb):
                    shutil.rmtree(data_root_rgb)
                if osp.exists(output_root_rgb):
                    shutil.rmtree(output_root_rgb)
                print('copying RGB frames ...')
                shutil.copytree(self.args.img_root, data_root_rgb)
                shutil.copytree(self.args.img_root, output_root_rgb)
            else:
                raise ValueError
            with open(extracted_flag, 'w') as f:
                f.writelines('extract RGB frames successed~')
        else:
            print('frames have been extracted ...')
        time2 = time.time()

        # 2. recording image_ids and rgb_paths
        self.image_ids = []
        self.rgb_paths = []
        for file_name in microsoft_sorted(os.listdir(self.rgb_root)):
            suffix = osp.splitext(file_name)[-1]
            if suffix not in ['.jpg', '.png', '.JPG']:
                continue
            img_id = osp.splitext(file_name)[0]
            self.image_ids.append(img_id)
            self.rgb_paths.append(osp.join(self.rgb_root, file_name))
        
        self.image_ids_all = self.image_ids.copy()
        
        # 3. generate LeReS depth
        pred_flag = osp.join(self.data_root, 'predicted_affine_flag.txt')
        if not (osp.exists(pred_flag) and osp.exists(osp.join(self.data_root, 'pred_feature'))):
            cmd = 'export PYTHONPATH=./LeReS/ && python ./LeReS/tools/test_depth.py --load_ckpt ./LeReS/res101.pth --backbone resnext101 --rgb_root %s --outputs %s' %(self.rgb_root, self.data_root)
            print('generating affine-invariant depth...')
            os.system(cmd)
            with open(pred_flag, 'w') as f:
                f.writelines('prediction of affine-invariant depth successed~')
        time3 = time.time()

        self.outputs['times']['extract_time'] = time2 - time1
        self.outputs['times']['inference_time'] = time3 - time2
        self.time3 = time3
    
    def copy_gt_depth(self, depth_root):
        copy_flag_gt_depth = osp.join(self.data_root, 'copy_flag_gt_depth.txt')
        if not os.path.exists(copy_flag_gt_depth):
            if self.args.img_root is not None:
                assert self.args.scene_name is not None
                print('copying RGB frames ...')
                shutil.copytree(depth_root, osp.join(self.data_root, 'gt_depth'))
                shutil.copytree(osp.join(self.data_root, 'gt_depth'), osp.join(self.output_root, 'gt_depth'))
            else:
                raise ValueError
            with open(copy_flag_gt_depth, 'w') as f:
                f.writelines('copy gt depth successed~')
        else:
            print('gt depths have been copied ...')

    def load_and_init(self):

        self.num_imgs = len(self.image_ids)

        demo_img = cv2.imread(self.rgb_paths[0]).astype(np.float32)
        h, w, _ = demo_img.shape
        self.h = h; self.w = w

        resize_ratio = max(self.args.resize_shape / self.h, self.args.resize_shape / self.w)
        self.h_optim = int(resize_ratio * self.h)
        self.w_optim = int(resize_ratio * self.w)

        # 1. read LeReS features and downsample frames according to the customized similarity
        if not self.args.outdoor_scenes:
            print('down sampling images...')
            leres_features = torch.stack([
                torch.from_numpy(
                        np.load(osp.join(self.data_root, 'pred_feature', '%s-feature.npy' % image_id)).astype(np.float32)
                        ).cuda() for image_id in self.image_ids
                ], dim=0).float()
            feature_sim_cal = FeatureSimComputer(leres_features, max_width=200)
            similarity_matrix = feature_sim_cal.calculate_similarity() # [n, max_width+1]
            similarity_matrix_bool = (similarity_matrix > self.args.sim_thr) * 1.
            image_ids_indexes = image_indexes_from_sim_matrix(similarity_matrix_bool)
            del leres_features, feature_sim_cal, similarity_matrix, similarity_matrix_bool
        else:
            image_ids_indexes = np.arange(len(self.image_ids))
            
        self.image_ids = [self.image_ids[index] for index in image_ids_indexes]
        self.rgb_paths = [self.rgb_paths[index] for index in image_ids_indexes]

        self.num_imgs = len(self.image_ids)
        self.image_ids_indexes = np.array(image_ids_indexes)

        rgb_selected_root = osp.join(self.data_root, 'rgb_selected')
        os.makedirs(rgb_selected_root, exist_ok=True)
        for img_id in range(self.num_imgs):
            shutil.copyfile(self.rgb_paths[img_id], self.rgb_paths[img_id].replace('/rgb/', '/rgb_selected/'))
        
        # 2. load rgb images
        self.rgb_imgs_wo_resize_cpu = torch.stack([
            img_numpy_to_torch(
                cv2.imread(path)[:, :, ::-1].astype(np.float32),
            ) for path in self.rgb_paths
        ], dim=0) # [n, 3, h, w]

        self.recon_ratio = min(self.w_optim / args.recon_shape, self.h_optim / args.recon_shape)
        self.w_recon = int(self.w_optim / self.recon_ratio);  self.h_recon = int(self.h_optim / self.recon_ratio)

        self.rgb_imgs_recon_cpu = F.interpolate(self.rgb_imgs_wo_resize_cpu, (self.h_recon, self.w_recon), mode='bilinear')
        self.rgb_imgs = F.interpolate(self.rgb_imgs_wo_resize_cpu, (self.h_optim, self.w_optim), mode='bilinear').cuda()
        del self.rgb_imgs_wo_resize_cpu

        ## imgs guassian noise with guass_sigma
        if self.args.guass_sigma != 0:
            print('adding guassian noise with sigma =', self.args.guass_sigma)
            imgs_noise = torch.randn_like(self.rgb_imgs).cuda() * self.args.guass_sigma / 255.
            self.rgb_imgs += imgs_noise
            self.rgb_imgs[self.rgb_imgs > 1] = 1
            self.rgb_imgs[self.rgb_imgs < 0] = 0

            # demo noise img
            noise_img_demo = self.rgb_imgs[0].detach().cpu().numpy() * 255.
            noise_img_demo = np.transpose(noise_img_demo, (1, 2, 0))
            cv2.imwrite(osp.join(self.output_root, 'noise_img_demo.png'), noise_img_demo.astype(np.uint8))
            
        # 3. obtain gt_loader of dataset if necessary
        if self.args.gt_pose_flag or self.args.gt_intrinsic_flag or self.args.gt_depth_flag:
            if self.args.dataset_name == 'NYUDepthVideo':
                gt_loader = NYUDepthVideo_Loader(osp.join(args.gt_root, args.dataset_name))
            elif self.args.dataset_name == 'scannet_test_div_5':
                gt_loader = Scannet_test_Loader(osp.join(args.gt_root, args.dataset_name))
            elif self.args.dataset_name == '7scenes_new_seq1' or self.args.dataset_name == '7scenes_new_seq1_div5':
                gt_loader = SevenScenes_Loader(osp.join(args.gt_root, args.dataset_name))
            elif self.args.dataset_name == 'TUM':
                gt_loader = TUM_Loader(osp.join(args.gt_root, args.dataset_name))
            elif self.args.dataset_name == 'KITTI':
                gt_loader = KITTIDepthVideo_Loader(osp.join(args.gt_root, args.dataset_name), scene_names_list=[args.scene_name])
            else:
                raise ValueError
            
            gt_data_info = gt_loader.obtain_rgb_depth_intrinsic_pose(self.scene_name, sample_indexes=self.image_ids_indexes)
                
            if self.args.gt_pose_flag:
                gt_poses = gt_data_info['poses']
                gt_trans = gt_poses[:, :3, 3]
                gt_rot_quat = R.from_matrix(gt_poses[:, :3, :3]).as_quat()
                self.gt_poses_7dof = torch.from_numpy(np.concatenate((gt_trans, gt_rot_quat), axis=1)).cuda()
                self.gt_poses = torch.from_numpy(gt_poses).cuda()
                print('loading gt poses...')
            if self.args.gt_intrinsic_flag:
                self.gt_intrinsic = torch.from_numpy(gt_data_info['intrinsics']).cuda().double()
                print('loading gt intrinsic...')
            if self.args.gt_depth_flag:
                self.gt_depths_wo_resize_cpu = torch.from_numpy(gt_data_info['depths'])[:, None]
                self.gt_depths_recon_cpu = F.interpolate(self.gt_depths_wo_resize_cpu, (self.h_recon, self.w_recon), mode='nearest')
                self.gt_depths = F.interpolate(self.gt_depths_wo_resize_cpu, (self.h_optim, self.w_optim), mode='nearest').cuda()
                del self.gt_depths_wo_resize_cpu
                print('loading gt depth...')
            
            ## copy ground-truth depth to FrozenRecon/datasets if provided
            self.gt_depth_scale = self.args.gt_depth_scale
            assert self.gt_depth_scale is not None
            self.copy_gt_depth(gt_loader.depth_root)
        
        # 4. load mono_depths if do not use gt_depth
        if not self.args.gt_depth_flag:
            ## load mono depths
            print('loading mono_depths...')
            self.mono_depths_wo_resize = torch.stack([
                torch.from_numpy(
                    np.load(osp.join(self.data_root, 'pred_depth_npy', image_id + '-depth.npy')).astype(np.float32),
                    )[None, ...] for image_id in self.image_ids
                ], dim=0).float().cuda()
            self.mono_depths_recon = F.interpolate(self.mono_depths_wo_resize, (self.h_recon, self.w_recon), mode='bilinear')
            self.mono_depths = F.interpolate(self.mono_depths_wo_resize, (self.h_optim, self.w_optim), mode='bilinear')
            del self.mono_depths_wo_resize

        # 5. load invalid masks if outdoor_scenes
        if self.args.outdoor_scenes and mmseg_import_flag:
            seg_check_path = self.rgb_paths[-1].replace('/rgb/', '/seg_invalid_masks/').replace('.jpg', '.npy').replace('.png', '.npy')
            if not osp.exists(seg_check_path):
                print('generating invalid_masks')
                seg_root = osp.join(osp.dirname(osp.dirname(__file__)), 'SegFormer')
                seg_cfg_path = osp.join(seg_root, 'local_configs/segformer/B3/segformer.b3.512x512.ade.160k.py')
                seg_ckpt_path = osp.join(seg_root, 'segformer.b3.512x512.ade.160k.pth')
                seg_model = init_segmentor(seg_cfg_path, seg_ckpt_path, device=torch.device('cuda'))
                for i, img_path in enumerate(self.rgb_paths):
                    seg_result = inference_segmentor(seg_model, img_path)[0]
                    invalid_mask = ((seg_result == 2) | (seg_result == 12) | (seg_result == 20))  * 1.
                    save_path = img_path.replace('/rgb/', '/seg_invalid_masks/').replace('.jpg', '.npy').replace('.png', '.npy')
                    os.makedirs(osp.dirname(save_path), exist_ok=True)
                    np.save(save_path, invalid_mask)

            print('loading invalid_masks...')
            self.invalid_masks = torch.stack([
                torch.from_numpy(
                    cv2.resize(
                        np.load(path.replace('/rgb/', '/seg_invalid_masks/').replace('.jpg', '.npy').replace('.png', '.npy')).astype(np.float32),
                        (self.w_optim, self.h_optim),
                        cv2.INTER_NEAREST
                    ))[None, ...].cuda() for path in self.rgb_paths
                ], dim=0).float()
        else:
            pass

        pred_valid_mask = [True for id in self.image_ids]
        
        if self.args.gt_pose_flag or self.args.gt_intrinsic_flag or self.args.gt_depth_flag:
            # 6. mask these to the intersection between the sampled frames and gt valid frames
            sampled_filtered_ids = gt_data_info['sampled_filtered_ids']
            pred_valid_mask = [True if id in sampled_filtered_ids else False for id in self.image_ids]
            pred_valid_mask = torch.tensor(pred_valid_mask).bool()
            self.image_ids = [id for i, id in enumerate(self.image_ids) if pred_valid_mask[i]]
            assert self.image_ids == sampled_filtered_ids
            self.rgb_paths = [path for i, path in enumerate(self.rgb_paths) if pred_valid_mask[i]]
            self.rgb_imgs_recon_cpu = self.rgb_imgs_recon_cpu[pred_valid_mask]
            self.rgb_imgs = self.rgb_imgs[pred_valid_mask]
            self.image_ids_indexes = np.array([self.image_ids_all.index(id) for id in sampled_filtered_ids])
            if not self.args.gt_depth_flag:
                self.mono_depths_recon = self.mono_depths_recon[pred_valid_mask]
                self.mono_depths = self.mono_depths[pred_valid_mask]
                
        self.num_imgs = len(self.image_ids)

        if self.args.outdoor_scenes and mmseg_import_flag:
            self.invalid_masks = self.invalid_masks[pred_valid_mask]

        # 7. optimizer 
        self.optimize_params = {}
        params = []

        if not self.args.gt_pose_flag:
            poses_6dof_t = torch.zeros((self.num_imgs-1, 3)).double().cuda()
            poses_6dof_r = torch.zeros((self.num_imgs-1, 3)).double().cuda()

            self.optimize_params['poses_6dof_t'] = poses_6dof_t
            self.optimize_params['poses_6dof_r'] = poses_6dof_r
            params.append({'params': self.optimize_params['poses_6dof_t'], 'lr': 0})
            params.append({'params': self.optimize_params['poses_6dof_r'], 'lr': 0})
        
        if not self.args.gt_intrinsic_flag:
            self.optimize_params['focal_length_ratio'] = torch.tensor([1.2]).cuda().double()
            params.append({'params': [self.optimize_params['focal_length_ratio']], 'lr': 0})
        
        if not self.args.gt_depth_flag:
            self.optimize_params['scale_map'] = torch.tensor([1.]).cuda().repeat(self.num_imgs)
            self.optimize_params['shift_map'] = torch.tensor([0.]).cuda().repeat(self.num_imgs)
            
            params.append({'params': [self.optimize_params['scale_map'], self.optimize_params['shift_map']], 'lr': 0})
            
            self.optimize_params['sparse_guided_points'] = torch.ones((self.num_imgs, 5, 5)).float().cuda()
            params.append({'params': [self.optimize_params['sparse_guided_points']], 'lr': 0})


        for key in self.optimize_params:
            self.optimize_params[key].requires_grad = True
        
        if params != []:
            self.args.pseudo_training = False
            self.optimizer = torch.optim.AdamW(params, 
                                        lr=0,
                                        betas=(0.9, 0.999),
                                        weight_decay=0)
        else:
            self.args.pseudo_training = True
            self.args.iters_per_epoch = 50
            self.args.epochs = 1
            param_pseudo = torch.tensor([1.]).cuda().repeat(self.num_imgs)
            param_pseudo.requires_grad = True
            self.optimize_params['param_pseudo'] = param_pseudo
            params.append({'params': [
                        self.optimize_params['param_pseudo']
                    ], 'lr': 0})
            self.optimizer = torch.optim.AdamW(params, 
                                        lr=0,
                                        betas=(0.9, 0.999),
                                        weight_decay=0)
    
    def optimize(self):
        
        # total 3 epochs
        for epoch in range(self.args.epochs):
            # 1. set learning rate
            pose_lr_t = self.args.pose_lr_t[epoch]
            pose_lr_r = self.args.pose_lr_r[epoch]
            focal_lr = self.args.focal_lr[epoch]
            depth_lr = self.args.depth_lr[epoch]
            sparse_points_lr = self.args.sparse_points_lr[epoch]
            
            lr_set_cnt = 0
            if not self.args.gt_pose_flag:
                self.optimizer.param_groups[lr_set_cnt]['lr'] = pose_lr_t
                lr_set_cnt += 1
                self.optimizer.param_groups[lr_set_cnt]['lr'] = pose_lr_r
                lr_set_cnt += 1
                print('Poses lr set...')

            if not self.args.gt_intrinsic_flag:
                self.optimizer.param_groups[lr_set_cnt]['lr'] = focal_lr
                lr_set_cnt += 1
                print('Intrinsic lr set...')

            if not self.args.gt_depth_flag:
                self.optimizer.param_groups[lr_set_cnt]['lr'] = depth_lr
                lr_set_cnt += 1
                self.optimizer.param_groups[lr_set_cnt]['lr'] = sparse_points_lr
                lr_set_cnt += 1
                print('Depth lr set...')
                
            h_optim = self.h_optim; w_optim = self.w_optim

            # 2. optimization
            progress_bar = tqdm(range(0, self.args.iters_per_epoch), desc="Training progress of epoch %s" %epoch)
            for t in range(self.args.iters_per_epoch):
                time1 = time.time()
                
                # 2.1 compute camera intrinsic
                w_ratio = self.w_optim / self.w
                h_ratio = self.h_optim / self.h
                if not self.args.gt_intrinsic_flag:
                    fx = self.optimize_params['focal_length_ratio'] * (self.w_optim / self.w) * ((self.h + self.w) / 2)
                    fy = self.optimize_params['focal_length_ratio'] * (self.h_optim / self.h) * ((self.h + self.w) / 2)
                    intrinsics = torch.tensor([[[0, 0, w_optim/2], [0, 0, h_optim/2], [0, 0, 1]]]).cuda().double()
                    intrinsics[0, 0:1, 0:1] = fx
                    intrinsics[0, 1:2, 1:2] = fy
                else:
                    intrinsics = self.gt_intrinsic.clone()
                    ## scale camera intrinsic to the shape of gt_depth
                    intrinsics[:, 0, 0] *= w_ratio
                    intrinsics[:, 0, 2] *= w_ratio
                    intrinsics[:, 1, 1] *= h_ratio
                    intrinsics[:, 1, 2] *= h_ratio

                # 2.2 compute global camera poses
                if not self.args.gt_pose_flag:
                    poses_6dof = torch.cat((self.optimize_params['poses_6dof_t'], self.optimize_params['poses_6dof_r']), dim=1)
                    if lietorch_flag:
                        poses_rel_SE3_near = SE3.exp(poses_6dof)

                        pose_init = torch.zeros(1, 6).double().cuda()
                        pose_init_SE3 = SE3.exp(pose_init)
                        poses_global_SE3 = [pose_init_SE3]
                        # NOTE: it is very time-spending
                        for i in range(self.num_imgs-1):
                            pose_init_SE3 = poses_rel_SE3_near[i:(i+1)] * pose_init_SE3
                            poses_global_SE3.append(pose_init_SE3)
                        poses_global_SE3 = lietorch.cat(poses_global_SE3, dim=0)
                    else:
                        pose_init = torch.eye(4)[None, ...].double().cuda()
                        pose = torch.eye(4).double().cuda()
                        poses_computed_global = [pose_init]
                        for i in range(self.num_imgs-1):
                            pose_temp = pose_vec2mat(poses_6dof[i][None, ...])
                            pose_ones = torch.tensor([[[0, 0, 0, 1]]]).double().cuda()
                            pose_temp = torch.cat((pose_temp, pose_ones), dim=1)
                            pose_init = pose_temp @ pose_init
                            poses_computed_global.append(pose_init)
                        poses_computed_global = torch.cat(poses_computed_global, dim=0)
                else:
                    if lietorch_flag:
                        poses_global_SE3 = SE3.InitFromVec(self.gt_poses_7dof).inv()
                    else:
                        poses_computed_global = self.gt_poses.inverse()
                    

                # 2.3 compute relative camera poses of each two frames
                time2 = time.time()
                if epoch == 0:
                    rel_angle = None
                else:
                    if lietorch_flag:
                        poses_rel_SE3_all = poses_global_SE3[None, ...] * poses_global_SE3[:, None].inv()
                        rel_angle = poses_rel_SE3_all.log()[:, :, 3:].norm(dim=-1)
                    else:
                        poses_computed_global_i = poses_computed_global[:, None] # [n, 1, 4, 4]
                        poses_computed_global_j = poses_computed_global[None, ...]  # [1, n, 4, 4]
                        poses_computed_relative = poses_computed_global_j.double() @ poses_computed_global_i.double().inverse()

                        R_rel_trace = poses_computed_relative[:, :, 0, 0] + poses_computed_relative[:, :, 1, 1] + poses_computed_relative[:, :, 2, 2] # R_rel_trace=2cosw+1
                        eps = 1e-7
                        rel_angle = torch.acos(torch.clamp((R_rel_trace - 1) / 2, -1 + eps, 1 - eps)) # relative angle
                        rel_angle[(rel_angle - rel_angle.T) > eps] = 3.14

                time3 = time.time()

                # 2.4 sample tgt_ids and ref_ids for warping and computing losses 
                tgt_ids = []
                ref_ids = []
                sample_imgs = self.args.samples_per_iter
                sample_imgs = min(sample_imgs, self.num_imgs)
                angle = None if epoch == 0 else rel_angle.clone()
                sample_ref_num = 1

                tgt_ids = torch.randperm(self.num_imgs)[:sample_imgs].cuda()
                ref_ids = get_ref_ids(tgt_ids, self.num_imgs, self.args.near_frames_num, angle=angle, epoch=epoch, sample_ref_num=sample_ref_num, angle_thr=self.args.angle_thr)
                tgt_ids = tgt_ids[:, None].repeat(1, sample_ref_num).view(-1)
                ref_ids = ref_ids.view(-1)
                assert tgt_ids.shape == ref_ids.shape
                time4 = time.time()

                sample_num = int(w_optim * h_optim * self.args.sample_pix_ratio) 
                if lietorch_flag:
                    valid_mask = (ref_ids >= 0) & (ref_ids <= self.num_imgs-1) & \
                        ((torch.isnan(poses_global_SE3.data[tgt_ids.clamp(min=0, max=self.num_imgs-1)]) + torch.isnan(poses_global_SE3.data[ref_ids.clamp(min=0, max=self.num_imgs-1)])).sum(dim=1) == 0)
                else:
                    valid_mask = (ref_ids >= 0) & (ref_ids <= self.num_imgs-1) & \
                        ((torch.isnan(poses_computed_global[tgt_ids.clamp(min=0, max=self.num_imgs-1)]) + torch.isnan(poses_computed_global[ref_ids.clamp(min=0, max=self.num_imgs-1)])).sum(dim=(1, 2)) == 0)
                tgt_ids = tgt_ids[valid_mask]
                ref_ids = ref_ids[valid_mask]
                n = ref_ids.shape[0]

                # 2.5 sample pixels from coords
                coords_x = torch.arange(0, w_optim).float().cuda()[None, None, :].repeat(n, h_optim, 1).view(n, -1)
                coords_y = torch.arange(0, h_optim).float().cuda()[None, :, None].repeat(n, 1, w_optim).view(n, -1)

                valid_imgs_num = coords_x.shape[0]
                coords_mask_index = torch.randperm(h_optim * w_optim)[:sample_num].repeat(valid_imgs_num, 1).cuda()
                coords_mask = (coords_x != coords_x)
                coords_mask.scatter_(1, coords_mask_index, 1)

                coords_x = coords_x[coords_mask].view(valid_imgs_num, -1)
                coords_y = coords_y[coords_mask].view(valid_imgs_num, -1)

                # 2.6 computing lwlr of mono depth
                if not self.args.gt_depth_flag:
                    metric_depths_global = self.mono_depths * self.optimize_params['scale_map'][:, None, None, None].abs() + self.optimize_params['shift_map'][:, None, None, None]
                    metric_depths_global = metric_depths_global.squeeze()
                    metric_depths_global += 1e-6
                    
                    # speed up version of lwlr
                    sparse_guided_depth = fill_grid_sparse_depth_torch_batch(metric_depths_global, self.optimize_params['sparse_guided_points'].abs(), fill_coords=None, device=torch.device('cuda'))
                    sparse_guided_depth = sparse_guided_depth * metric_depths_global
                    k_para = self.args.k_para
                    metric_depths = sparse_depth_lwlr_batch(metric_depths_global, sparse_guided_depth, down_sample_scale=32, k_para=k_para, sample_num=sparse_guided_depth[0][sparse_guided_depth[0]>0].numel(), device=torch.device('cuda'))[:, None, ...] # lwlr
                else:
                    metric_depths = self.gt_depths
                metric_depths[metric_depths < 0] = 0

                # 2.7 computing relative poses
                if epoch == 0:
                    if lietorch_flag:
                        poses_SE3 = poses_global_SE3[ref_ids] * poses_global_SE3[tgt_ids].inv()
                    else:
                        poses = poses_computed_global[ref_ids] @ poses_computed_global[tgt_ids].inverse()
                else:
                    if lietorch_flag:
                        poses_SE3 = poses_rel_SE3_all[tgt_ids, ref_ids]
                    else:
                        poses = poses_computed_relative[tgt_ids, ref_ids]

                if lietorch_flag:
                    poses = poses_SE3.matrix()
                assert (poses.sum(dim=1).sum(dim=1) == 0).sum() == 0
                time5 = time.time()

                # 2.8 warp from reference images to target images and compute losses
                tgt_sample_coords = torch.stack([coords_x, coords_y], dim=2)[:, :, None, :] # [n(filtered), sampled_num, 1, 2]
                tgt_sample_coords[..., 0] = 2 * tgt_sample_coords[..., 0] / (w_optim - 1) - 1
                tgt_sample_coords[..., 1] = 2 * tgt_sample_coords[..., 1] / (h_optim - 1) - 1
                
                if self.args.gt_intrinsic_flag:
                    intrinsics_tgt = intrinsics[tgt_ids]
                else:
                    intrinsics_tgt = intrinsics
                depth_computed_points, projected_x, projected_y = depth_project(coords_x, coords_y, metric_depths[tgt_ids].view(tgt_ids.numel(), 1, -1), intrinsics_tgt, poses, coords_mask, (h_optim, w_optim))
                ## valid_mask of projection
                valid_mask = (projected_x >= -0.95) & (projected_x <= 0.95) & (projected_y >= -0.95) & (projected_y <= 0.95)
                ## tgt color
                color_tgt = F.grid_sample(self.rgb_imgs[tgt_ids].float(), tgt_sample_coords.float(), padding_mode='zeros', align_corners=False)
                ## projected ref color
                sample_coords = torch.stack([projected_x, projected_y], dim=2)[:, :, None, :] # [n(filtered), sample_num, 1, 2]
                color_projected = F.grid_sample(self.rgb_imgs[ref_ids].float(), sample_coords.float(), padding_mode='zeros', align_corners=False)
                ## projected ref depth
                depth_projected_points = F.grid_sample(metric_depths[ref_ids].float(), sample_coords.float(), padding_mode='zeros', align_corners=False)
                depth_projected_points = depth_projected_points[:, 0, :, 0] # [n, sample_num]

                if self.args.outdoor_scenes and mmseg_import_flag:
                    ## invalid_mask
                    invalid_mask_tgt = F.grid_sample(self.invalid_masks[tgt_ids].float(), tgt_sample_coords.float(), padding_mode='zeros', align_corners=False, mode='nearest')
                    invalid_mask_ref2tgt = F.grid_sample(self.invalid_masks[ref_ids].float(), sample_coords.float(), padding_mode='zeros', align_corners=False, mode='nearest')

                ## update valid_mask
                valid_mask[depth_projected_points == 0] = 0
                valid_mask[depth_computed_points == 0] = 0
                valid_mask[(color_projected.sum(dim=1) == 0)[..., 0]] = 0

                if self.args.outdoor_scenes and mmseg_import_flag:
                    valid_mask[invalid_mask_tgt[:, 0, :, 0] == 1] = 0
                    valid_mask[invalid_mask_ref2tgt[:, 0, :, 0] == 1] = 0

                ## filter out valid depth
                depth_projected_points = depth_projected_points[valid_mask]
                depth_computed_points = depth_computed_points[valid_mask]
                assert depth_projected_points.shape == depth_computed_points.shape
                ## filter out valid colors
                color_tgt[~valid_mask[:, None, :, None].repeat(1, 3, 1, 1)] = 0
                color_projected[~valid_mask[:, None, :, None].repeat(1, 3, 1, 1)] = 0
                color_tgt = color_tgt.permute(0, 2, 1, 3)
                color_projected = color_projected.permute(0, 2, 1, 3)
                color_tgt = color_tgt[valid_mask]
                color_projected = color_projected[valid_mask]
                ## compute losses
                loss_photometric = (color_tgt - color_projected).abs()
                loss_photometric = loss_photometric.mean(dim=1).squeeze()
                loss_geometric = ((depth_projected_points - depth_computed_points).abs() / (depth_projected_points + depth_computed_points))
                time6 = time.time()

                valid_percent = self.args.max_valid_percent_loss
                if epoch < 2 and valid_percent < 1:
                    photo_thr = torch.quantile(loss_photometric, valid_percent)
                    loss_photometric = loss_photometric[loss_photometric < photo_thr]

                    geo_thr = torch.quantile(loss_geometric, valid_percent)
                    loss_geometric = loss_geometric[loss_geometric < geo_thr]

                loss_photometric = loss_photometric.mean()
                loss_geometric = loss_geometric.mean()
                if not self.args.gt_depth_flag:
                    loss_scale_norm = ((self.optimize_params['sparse_guided_points'][tgt_ids] - 1).abs().sum() + (self.optimize_params['sparse_guided_points'][ref_ids] - 1).abs().sum()) / (tgt_ids.numel() + ref_ids.numel())
                if self.args.pseudo_training:
                    loss_pseudo = self.optimize_params['param_pseudo'].sum() * 0.

                if t % self.args.print_iters == 0:
                    print_info = {}
                    print_info["Epoch"] = epoch
                    if not self.args.pseudo_training:
                        print_info["loss_photometric"] = loss_photometric.tolist()
                        print_info["loss_geometric"] = loss_geometric.tolist()
                        if not self.args.gt_depth_flag:
                            print_info["loss_scale_norm"] = loss_scale_norm.tolist()
                    else:
                        print_info["pseudo_training_loss"] = loss_pseudo.tolist()
                    progress_bar.set_postfix(print_info)
                    progress_bar.update(self.args.print_iters)
                if t == (self.args.iters_per_epoch - 1):
                    progress_bar.close()

                pc_weight = self.args.loss_pc_weight
                gc_weight = self.args.loss_gc_weight 
                scale_norm_weight = self.args.loss_norm_weight

                loss = pc_weight[epoch] * loss_photometric + gc_weight[epoch] * loss_geometric
                if not self.args.gt_depth_flag:
                    loss += (scale_norm_weight[epoch] * loss_scale_norm)
                if self.args.pseudo_training:
                    loss += loss_pseudo
                    
                time7 = time.time()

                # 2.9 save for the latest iteration
                if t == (self.args.iters_per_epoch - 1) and (epoch == self.args.epochs - 1):
                    resized_height_width = np.array([self.h_optim, self.w_optim])
                    if lietorch_flag:
                        poses_computed_global = poses_global_SE3.matrix()
                    
                    rgb_imgs = self.rgb_imgs_recon_cpu.permute(0, 2, 3, 1) * 255. # [n, h, w, 3]

                    if intrinsics.shape[0] == 1:
                        intrinsics = intrinsics.repeat(rgb_imgs.shape[0], 1, 1)
                    intrinsics[:, 0, 0] /= self.recon_ratio; intrinsics[:, 0, 2] /= self.recon_ratio; intrinsics[:, 1, 1] /= self.recon_ratio; intrinsics[:, 1, 2] /= self.recon_ratio
                    
                    if self.args.gt_depth_flag:
                        optimized_depth = self.gt_depths_recon_cpu.squeeze()
                    else:
                        
                        metric_depths_global = self.mono_depths_recon * self.optimize_params['scale_map'][:, None, None, None].abs() + self.optimize_params['shift_map'][:, None, None, None]
                        metric_depths_global = metric_depths_global.squeeze()
                        metric_depths_global += 1e-6
                        
                        # speed up version of lwlr
                        sparse_guided_depth = fill_grid_sparse_depth_torch_batch(metric_depths_global, self.optimize_params['sparse_guided_points'].abs(), fill_coords=None, device=torch.device('cuda'))
                        sparse_guided_depth = sparse_guided_depth * metric_depths_global

                        k_para = self.args.k_para
                        optimized_depth = sparse_depth_lwlr_batch(metric_depths_global, sparse_guided_depth, down_sample_scale=32, k_para=k_para, sample_num=sparse_guided_depth[0][sparse_guided_depth[0]>0].numel(), device=torch.device('cuda')).squeeze() # lwlr
                    
                    optimized_params = dict(
                        rgb_imgs = rgb_imgs.numpy().astype(np.uint8),
                        optimized_depth = optimized_depth.detach().cpu().numpy(),
                        poses_computed_global = poses_computed_global.detach().cpu().numpy(),
                        optimized_intrinsic = intrinsics.detach().cpu().numpy(),
                        optimized_image_indexes = self.image_ids_indexes,
                        resized_height_width = resized_height_width,
                    )
                    if not self.args.gt_depth_flag:
                        optimized_params.update(dict(
                            optimized_scale_map = self.optimize_params['scale_map'].detach().cpu().numpy(),
                            optimized_shift_map = self.optimize_params['shift_map'].detach().cpu().numpy(),
                            optimized_sparse_guided_points = self.optimize_params['sparse_guided_points'].detach().cpu().numpy(),
                        ))
                    
                    if args.outdoor_scenes and mmseg_import_flag:
                        optimized_params.update(dict(
                            invalid_masks = self.invalid_masks.detach().cpu().numpy(),
                        ))
                    
                    self.outputs['optimized_params'] = optimized_params
                    np.save(self.save_path, optimized_params)
                    print('saved to :', self.save_path)
                    break

                time8 = time.time()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                time9 = time.time()

                # print('time1 :', time2 - time1)
                # print('time2 :', time3 - time2)
                # print('time3 :', time4 - time3)
                # print('time4 :', time5 - time4)
                # print('time5 :', time6 - time5)
                # print('time6 :', time7 - time6)
                # print('time7 :', time8 - time7)
                # print('time8 :', time9 - time8)
                # print('total time :', time9 - time1)
                # # assert False
        
        time4 = time.time()
        self.outputs['times']['optimize_time'] = time4 - self.time3
        self.outputs['times']['total_time'] = self.outputs['times']['extract_time'] + self.outputs['times']['inference_time'] + self.outputs['times']['optimize_time']

    def save_excel(self):
        column_data = [
                self.scene_name, 
                self.outputs['times']['extract_time'], 
                self.outputs['times']['inference_time'], 
                self.outputs['times']['optimize_time'], 
                self.outputs['times']['total_time']
        ]
        self.excel.add_data(column_data)
        self.excel.save_excel(self.workbook_save_path)

    def run(self):
        self.load_and_init()
        self.optimize()
        self.save_excel()
        print('finished optimization~')

        return self.outputs


def recon(args, optimized_params, voxel_size=0.1):
    save_path_pred = osp.join(osp.dirname(save_params_path), 'optimized_mesh_voxel_%s.ply' %str(voxel_size))
    if osp.exists(save_path_pred) or osp.exists(save_path_pred.replace('.ply', '_mesh.ply')):
        print('pred pcd exists, continue...')
        return None
    
    images = optimized_params['rgb_imgs'] # [b, h, w, 3]
    depths = optimized_params['optimized_depth']
    pred_intrs = optimized_params['optimized_intrinsic']
    pred_poses = optimized_params['poses_computed_global']

    bs, h, w = depths.shape
    depths = depths.squeeze()
    if args.outdoor_scenes and mmseg_import_flag:
        invalid_masks = optimized_params['invalid_masks']
        invalid_masks = F.interpolate(invalid_masks, (images.shape[1], images.shape[2]), mode='nearest')
        invalid_masks = invalid_masks.detach().cpu().numpy()
        assert len(invalid_masks.shape) == 3
        depths[invalid_masks[optimized_params['optimized_image_indexes']] == 1] = 0

    tsdf = TSDFFusion()
    print('tsdf fusing the pred pcd...')
    save_pcd_path_optimized = tsdf.fusion(
        images.astype(np.uint8), 
        depths, 
        pred_intrs, 
        np.linalg.inv(pred_poses), 
        save_path=save_path_pred, edge_mask=True, save_mesh=True, voxel_size=voxel_size
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_path', help='The path to in_the_wild video for optimization.', type=str, default=None)
    parser.add_argument('--img_root', help='The path to in_the_wild images for optimization.', type=str, default=None)

    parser.add_argument('--recon_voxel_size', help='Voxel size of reconstruction.', type=float, default=0.1)
    parser.add_argument('--resize_shape', help='The shape of shortest side for optimization.', type=int, default=150)
    parser.add_argument('--recon_shape', help='The shape of shortest side for reconstruction, e.g., 480', type=int, default=480) 
    parser.add_argument('--angle_thr', help='Angle threshold of keyframe selection strategy.', type=float, default=np.pi/4)
    parser.add_argument('--k_para', help='The k parameter of lwlr function.', type=int, default=50) # seems does not matter so much
    parser.add_argument('--iters_per_epoch', help='Iterations for each epoch. 2000 is usually more than enough, and should not be too small.', type=int, default=2000)

    parser.add_argument('--pose_lr_t', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-2, 1e-3, 1e-3])
    parser.add_argument('--pose_lr_r', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-3, 1e-4, 1e-4])
    parser.add_argument('--focal_lr', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-2, 1e-2, 1e-2])
    parser.add_argument('--depth_lr', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-1, 1e-2, 1e-2])
    parser.add_argument('--sparse_points_lr', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-1, 1e-2, 1e-2])

    parser.add_argument('--loss_pc_weight', help="hyperparameter of losses.", type=float, nargs='+', default=[2, 2, 2])
    parser.add_argument('--loss_gc_weight', help="hyperparameter of losses.", type=float, nargs='+', default=[0.5, 1, 0.1])
    parser.add_argument('--loss_norm_weight', help="hyperparameter of losses.", type=float, nargs='+', default=[0.01, 0.1, 0.1])

    parser.add_argument('--dataset_name', help='Dataset name. Ignore it when input images or videos in the wild.', type=str, choices=['NYUDepthVideo', 'scannet_test_div_5', '7scenes_new_seq1', 'TUM', 'KITTI'], default=None)
    parser.add_argument('--scene_name', help='If None, optimize all scenes. If set, only optimize one scene.', type=str, default=None)
    parser.add_argument('--outdoor_scenes', help='Whether to optimize outdoor scenes. (Used for filtering out sky regions and car (dynamic objects).)', action='store_true')

    # gt args
    parser.add_argument('--gt_root', help='Path to ground-truth data root, change it.', type=str, default='/mnt/nas/share/home/xugk/data/')
    parser.add_argument('--gt_depth_flag', help='Whether to use gt depth straightforwardly.', action='store_true')
    parser.add_argument('--gt_pose_flag', help='Whether to use gt poses straightforwardly.', action='store_true')
    parser.add_argument('--gt_intrinsic_flag', help='Whether to use gt intrinsic straightforwardly.', action='store_true')
    
    parser.add_argument('--save_suffix', help='Save folder suffix, it will be useful for different settings of one scene.', type=str, default='')
    parser.add_argument('--save_mesh', help='Whether to reconstruct mesh for visualization.', type=bool, default=True)

    # usually do not change them
    parser.add_argument('--sample_pix_ratio', help='sample partial pixels for reducing computation cost.', type=float, default=0.25) # the ratio does not matter so much
    parser.add_argument('--epochs', help='Total optimization epochs', type=int, default=3)
    parser.add_argument('--frames_downsample_ratio', help='The downsample ratio of extracting frames from the input video. (Only when users input a video.)', type=int, default=10)
    parser.add_argument('--outputs', help='Output folders', type=str, default='./outputs')
    parser.add_argument('--data_root', help='Copy data, place it here, and generate temporary variables here.', type=str, default='./datasets')
    parser.add_argument('--samples_per_iter', help='Number of sample images for computing losses per iteration.', type=int, default=50)
    parser.add_argument('--sim_thr', help='Similarity threshold, used for down sampling frames. Usually, it is not so necessary, but it will help when the frames are too many.', type=float, default=0.85)
    parser.add_argument('--max_valid_percent_loss', help='Filter out big loss values larger than this percentage. (0.9 means 90%)', type=float, default=0.9) # slightly improved
    parser.add_argument('--guass_sigma', help='images guassian noise sigma value, only used for testing robustness', type=float, default=0) 
    parser.add_argument('--near_frames_num', help='Near frames for the first local optimization stage.', type=int, default=3)
    parser.add_argument('--print_iters', help='The frequence of printing.', type=int, default=1)
    args = parser.parse_args()

    if (args.save_suffix != '') and (not args.save_suffix.startswith('_')):
        args.save_suffix = '_' + args.save_suffix

    scene_names = []
    img_roots = []

    # NYU
    if args.dataset_name == 'NYUDepthVideo':
        args.gt_depth_scale = 5000.
        base_root = osp.join(args.gt_root, args.dataset_name)
        for scene_name in os.listdir(base_root):
            if scene_name == 'annotations':
                continue
            scene_names.append(scene_name)
            img_roots.append(osp.join(base_root + scene_name + '/rgb'))
    
    # ScanNet
    elif args.dataset_name == 'scannet_test_div_5':
        args.gt_depth_scale = 1000.
        base_root = osp.join(args.gt_root, args.dataset_name)
        for scene_id in range(7, 21):
            scene_names.append('scene07%02d_00' %scene_id)
            img_roots.append(osp.join(base_root, 'scene07%02d_00/color' %scene_id))
    
    # 7scenes
    elif args.dataset_name == '7scenes_new_seq1':
        args.gt_depth_scale = 1000.
        base_root = osp.join(args.gt_root, args.dataset_name)
        for scene_name in os.listdir(base_root):
            scene_names.append(scene_name)
            img_roots.append(osp.join(base_root + scene_name + '/rgb'))

    # TUM
    elif args.dataset_name == 'TUM':
        args.gt_depth_scale = 5000.
        base_root = osp.join(args.gt_root, args.dataset_name)
        for scene_name in os.listdir(base_root):
            if not osp.isdir(osp.join(base_root, scene_name)):
                continue
            if scene_name == 'annotations':
                continue
            scene_names.append(scene_name)
            img_roots.append(osp.join(base_root + scene_name + '/rgb'))
    
    # KITTI
    elif args.dataset_name == 'KITTI':
        args.gt_depth_scale = 256.
        base_root = osp.join(args.gt_root, args.dataset_name)
        scene_names = [
            "2011_09_26_drive_0001_sync",
            "2011_09_26_drive_0009_sync",
            "2011_09_26_drive_0091_sync",
            "2011_09_28_drive_0001_sync",
            "2011_09_29_drive_0004_sync",
            "2011_09_29_drive_0071_sync",
        ]
        for scene_name in scene_names:
            date = scene_name.split('_drive')[0]
            if not scene_name.endswith('_sync'):
                scene_name = scene_name + '_sync'
            img_root = osp.join(base_root, 'raw_data', date, scene_name, 'image_02/data/')
            img_roots.append(img_root)
    
    # self-imgs
    elif args.img_root is not None:
        args.dataset_name = 'self-imgs'
        if args.scene_name:
            scene_name = args.scene_name
        else:
            scene_name = osp.basename(args.img_root)
        scene_names.append(scene_name)
        img_roots.append(args.img_root)
    
    # self-video
    elif args.video_path is not None:
        args.dataset_name = 'self-video'
        if args.scene_name:
            scene_name = args.scene_name
        else:
            scene_name = osp.splitext(osp.basename(args.video_path))[0]
        scene_names.append(scene_name)
        img_roots.append('') # pseudo parameters
    else:
        raise ValueError('Error of args.dataset_name')
    
    for (scene_name, img_root) in zip(scene_names, img_roots):
        if (args.scene_name is not None) and (scene_name != args.scene_name):
            continue

        args.scene_name = scene_name
        args.img_root = img_root
        print('scene_name :', args.scene_name)

        save_params_path = osp.join(args.outputs, args.scene_name + args.save_suffix, 'optimized_params.npy')
        if not os.path.exists(save_params_path):
            recon_optimizer = FrozenRecon(args)
            outputs = recon_optimizer.run()
            optimized_params = outputs['optimized_params']
            del recon_optimizer
        else:
            print('Loading optimized FrozenRecon params...')
            outputs = np.load(save_params_path, allow_pickle=True).item()
            optimized_params = outputs

        if args.save_mesh:
            print('reconstructing ...')
            recon(args, optimized_params, voxel_size=args.recon_voxel_size)
