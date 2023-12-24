from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import os.path as osp
import argparse
import numpy as np
import torch
from lwlr_numpy_origin import *

print(torch.__version__)
print(torch.version.cuda)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')
    parser.add_argument('--rgb_root', default='', help='rgb root')
    parser.add_argument('--outputs', default='', help='rgb root')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


if __name__ == '__main__':

    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()

    # base_root = '/home/xugk/code/dataset/scannet_test/'
    # base_root = '/home/xugk/code/dataset/7scenes_new_seq1/'
    # base_root = '/home/xugk/code/dataset/NYUDepthVideo/'
    # base_root = '/home/xugk/code/dataset/ETH3D/'
    # base_root = '/data/scannetv2_unpack/scans_test/'
    # base_root = '/home/xugk/code/dataset/self-video/img/'
    # base_root = '/home/xugk/code/dataset/TUM/'
    # base_root = '/home/xugk/code/dataset/Replica/'
    # for scene_name in sorted(os.listdir(base_root)):
    # for scene_name in sorted(os.listdir(base_root)):

    scene_name = osp.basename(osp.dirname(args.rgb_root))

    if ('annotation' in scene_name) or ('useless' in scene_name) or (not osp.isdir(args.rgb_root)):
        pass
    else:
        # if 'jpg' not in scene_name:
        #     continue

        # if 'scene' not in scene_name:
        #     continue
        
        # # scene_name = scene_name.split('_dslr_jpg')[0]
        # scene_id = int(scene_name.split('_')[0].split('scene')[-1])
        # if scene_id > 720:
        #     continue

        image_dir = args.rgb_root
        # image_dir = base_root + scene_name + '/color/'
        # image_dir = base_root + '%s_dslr_jpg/%s/images/dslr_images/' %(scene_name, scene_name)
        imgs_list = os.listdir(image_dir)
        imgs_list.sort()
        imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
        if args.outputs is None:
            image_dir_out = os.path.dirname(os.path.dirname(__file__)) + '/outputs_leres/outputs_' + scene_name
        else:
            image_dir_out = osp.join(args.outputs)
        os.makedirs(image_dir_out, exist_ok=True)

        for i, v in enumerate(imgs_path):
            print('processing (%04d)-th image... %s' % (i, v))
            if not (v.endswith('.jpg') or v.endswith('.png') or v.endswith('.JPG')):
                continue
            
            # # for neural rgbd reconstruction
            # lwlr_depth_metric_path = os.path.join(os.path.join(image_dir_out, 'lwlr'), v.split('/')[-1].replace('/img', '/depth'))
            # if os.path.exists(lwlr_depth_metric_path) and :
            #     print('passing :', lwlr_depth_metric_path)
            #     continue

            rgb = cv2.imread(v)

            # # Resize for scannet
            # rgb = cv2.resize(rgb, (640, 480))

            rgb_c = rgb[:, :, ::-1].copy()
            gt_depth = None
            A_resize = cv2.resize(rgb_c, (448, 448))
            rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

            img_torch = scale_torch(A_resize)[None, :, :, :]
            pred_depth, lateral_out = depth_model.inference(img_torch)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            feature_save = lateral_out[3]
            feature_save = feature_save.cpu().numpy().squeeze()
            pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
            print(pred_depth_ori.max())
            print(pred_depth_ori.min())

            # print(feature_save.shape)
            # assert False
            

            # if GT depth is available, uncomment the following part to recover the metric depth
            #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

            # img_name = v.split('/')[-1]
            # cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
            # # save depth
            # plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
            # cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
            # np.save(os.path.join(image_dir_out, img_name[:-4]+'-depth.npy'), pred_depth_ori)

            
            # inference for neural rgbd reconstruction
            img_name = v.split('/')[-1]
            temp_dir = image_dir_out
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(osp.join(temp_dir, 'pred_depth_viz'), exist_ok=True)
            os.makedirs(osp.join(temp_dir, 'pred_depth_npy'), exist_ok=True)
            os.makedirs(osp.join(temp_dir, 'pred_feature'), exist_ok=True)
            
            os.makedirs(temp_dir, exist_ok=True)
            # cv2.imwrite(os.path.join(temp_dir, img_name), rgb)
            # save depth
            plt.imsave(os.path.join(temp_dir, 'pred_depth_viz', img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
            # cv2.imwrite(os.path.join(temp_dir, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
            np.save(os.path.join(temp_dir, 'pred_depth_npy', img_name[:-4]+'-depth.npy'), pred_depth_ori)

            np.save(os.path.join(temp_dir, 'pred_feature', img_name[:-4]+'-feature.npy'), feature_save)


            # gt_depth_path = v.replace('/test_images/', '/depth_filtered/').replace('/img', '/depth')
            # gt_depth = cv2.imread(gt_depth_path, -1) / 1000.

            # pred_depth_global = recover_metric_depth(pred_depth_ori, gt_depth)
            # pred_lwlr = sparse_depth_lwlr(pred_depth_ori, gt_depth, sample_mode='grid', sample_num=100)

            # absrel_global = absrel_single(pred_depth_global, gt_depth)
            # absrel_lwlr = absrel_single(pred_lwlr, gt_depth)
            # print('absrel_global :', absrel_global)
            # print('absrel_lwlr :', absrel_lwlr)

            # global_depth_metric_path = os.path.join(os.path.join(image_dir_out, 'global'), v.split('/')[-1].replace('/img', '/depth'))
            # lwlr_depth_metric_path = os.path.join(os.path.join(image_dir_out, 'lwlr'), v.split('/')[-1].replace('/img', '/depth'))
            # os.makedirs(os.path.dirname(global_depth_metric_path), exist_ok=True)
            # os.makedirs(os.path.dirname(lwlr_depth_metric_path), exist_ok=True)

            # cv2.imwrite(global_depth_metric_path, (pred_depth_global * 1000).astype(np.uint16))
            # cv2.imwrite(lwlr_depth_metric_path, (pred_lwlr * 1000).astype(np.uint16))

            # os.makedirs(os.path.dirname(global_depth_metric_path.replace('global', 'global_viz')), exist_ok=True)
            # os.makedirs(os.path.dirname(lwlr_depth_metric_path.replace('lwlr', 'lwlr_viz')), exist_ok=True)
            # plt.imsave(global_depth_metric_path.replace('global', 'global_viz'), pred_depth_global, cmap='rainbow')
            # plt.imsave(lwlr_depth_metric_path.replace('lwlr', 'lwlr_viz'), pred_lwlr, cmap='rainbow')