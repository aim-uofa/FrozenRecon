import numpy as np
from . import fusion
import os

class TSDFFusion:
    def __init__(self):
        pass
    
    def fusion(self, images, depths, intrinsics, poses, frame_mask=None, save_path=None, save_mesh=False, edge_mask=False, n_imgs=None, voxel_size=0.02):
        assert intrinsics.shape[-1] == intrinsics.shape[-1] == 3
        if len(intrinsics.shape) == 2:
            intrinsics = np.repeat(intrinsics[None, ...], images.shape[0], axis=0)
        
        if frame_mask is not None:
            images = images[frame_mask]
            depths = depths[frame_mask]
            intrinsics = intrinsics[frame_mask]
            poses = poses[frame_mask]
        
        assert images.shape[0] == depths.shape[0] == intrinsics.shape[0] == poses.shape[0]
        if n_imgs is None:
            n_imgs = images.shape[0]
        h, w = depths.shape[-2:]
        vol_bnds = np.zeros((3,2))

        for i in range(n_imgs):
            depth_im = np.squeeze(depths[i])
            if edge_mask:
                # valid_mask
                valid_mask = np.zeros_like(depth_im).astype(np.bool_)
                valid_mask[20:h-20, 20:w-20] = True
                depth_im[~valid_mask] = 0

            cam_pose = poses[i]
            cam_intr = intrinsics[i]

            # Compute camera view frustum and extend convex hull
            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
        # print('tsdf fusing with vol_bnds :', vol_bnds)
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

        # Loop through RGB-D images and fuse them together
        for i in range(n_imgs):
            color_image = images[i]
            depth_im = np.squeeze(depths[i])

            if edge_mask:
                # valid_mask
                valid_mask = np.zeros_like(depth_im).astype(np.bool_)
                valid_mask[20:h-20, 20:w-20] = True
                depth_im[~valid_mask] = 0
            
            cam_pose = poses[i]
            cam_intr = intrinsics[i]

            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        
        if save_path is None:
            print('warning! save_pcd_path is None...')
        else:
            assert save_path.endswith('.ply')
            if save_mesh:
                save_path = save_path.replace('.ply', '_mesh.ply')
                # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
                verts, faces, norms, colors = tsdf_vol.get_mesh()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print("Saving mesh to mesh.ply :", save_path)
                fusion.meshwrite(save_path, verts, faces, norms, colors)
            else:
                save_path = save_path.replace('.ply', '_pc.ply')
                # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
                point_cloud = tsdf_vol.get_point_cloud()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print("Saving point cloud to pc.ply :", save_path)
                fusion.pcwrite(save_path, point_cloud)

        return save_path
        