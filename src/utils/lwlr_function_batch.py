from multiprocessing.sharedctypes import Value
import numpy as np
import cv2
import random
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

try:
    import torch
    torch_imported = True
    import torch.nn.functional as F
except:
    torch_imported = False


def torch_det_2x2(tensor):
    '''
    tensor: torch.tensor, [B, 2, 2]
    '''

    a = tensor[:, 0, 0]
    b = tensor[:, 0, 1]
    c = tensor[:, 1, 0]
    d = tensor[:, 1, 1]

    return a * d - b * c

def torch_inverse_2x2(tensor):
    '''
    tensor: torch.tensor, [B, 2, 2]
    [ a, b
      c, d ]
    '''

    a = tensor[:, 0, 0]; b = tensor[:, 0, 1]
    c = tensor[:, 1, 0]; d = tensor[:, 1, 1]
    ad_minus_bc = a * d - b * c
    if torch.any(ad_minus_bc == 0):
        return tensor.inverse()

    coefficient = 1 / (ad_minus_bc) # 1 / (ad - bc)
    
    '''
    [ a, b      --> [  d, -b
      c, d ]          -c,  a ]
    '''
    # tensor = tensor.transpose(1,2).flip(dims=[1,2])
    # tensor[:, 1, 0] *= -1
    # tensor[:, 0, 1] *= -1
    # tensor = tensor * coefficient[:, None, None]

    tensor_new = torch.stack([d, -b, -c, a]).permute(1, 0).view(-1, 2, 2)
    tensor_new = tensor_new * coefficient[:, None, None]

    return tensor_new


def absrel_single(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    mask = (gt > 0)# & (pred > 0)
    pred = pred[mask]
    gt = gt[mask]
    rel = np.abs(gt - pred) / gt  # compute errors
    abs_rel = np.mean(rel)
    # print("abs_rel:", abs_rel)
    return abs_rel


def recover_metric_depth(pred, gt, mask0=None):
    if type(pred).__module__ == torch.__name__ and torch_imported:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__ and torch_imported:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) #& (pred > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        if type(mask0).__module__ == torch.__name__ and torch_imported:
            mask0 = mask0.cpu().numpy()
        mask0 = mask0.squeeze()
        mask0 = mask0 > 0
        mask = mask & mask0
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    #print('a:',a, 'b:', b)
    if True:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / (pred_mean+1e-8))
    return pred_metric


# def lwlr_torch(testPoint_origin, xArr, yArr, uvArr, test_uvArr, k, down_sample_shape=None, lambda_limit_value=1, loop=False):

#     # time1 = time.time()
#     if loop == False:
#         # concat ones matrix
#         if xArr.ndim == 2:
#             xArr_ones = torch.ones((xArr.shape[0], 1)).cuda()
#             xArr = torch.cat((xArr_ones, xArr), dim=1)
#         elif xArr.ndim == 1:
#             xArr_ones = torch.ones((xArr.shape[0], 1)).cuda()
#             xArr = torch.cat((xArr_ones, xArr[:, None]), dim=1)
#         else:
#             raise ValueError()

#         testPoint = testPoint_origin.reshape(-1)
#         if testPoint.ndim == 1:
#             testPoint_ones = torch.ones((testPoint.shape[0], 1)).cuda()
#             testPoint = torch.cat((testPoint_ones, testPoint[:, None]), dim=1)
#         else:
#             raise ValueError()

#     # time2 = time.time()
#     m = xArr.shape[0]
#     n = test_uvArr.shape[0]
#     weights = torch.eye(m)[None, ...].repeat(n, 1, 1).cuda()
#     for j in range(m):
#         # diffArr = torch.linalg.norm(test_uvArr - uvArr[j, :], ord=2, dim=1)  # L2 distance
#         diffArr = torch.norm(test_uvArr - uvArr[j, :], p=2, dim=1)  # L2 distance
#         weights[:, j, j] = torch.exp(diffArr**2 / (-2.0 * k**2))  # weighted matrix
#     xArr_repeat = xArr[None, ...].repeat(n, 1, 1)
#     xTx = torch.matmul(xArr_repeat.permute(0, 2, 1), torch.matmul(weights, xArr_repeat))

#     lambda_limit = torch.zeros_like(xTx).cuda()
#     lambda_limit[:, 0, 0] = lambda_limit_value
#     xTx = xTx + lambda_limit

#     # time3 = time.time()

#     # if torch.any(torch.det(xTx) == 0.0):
#     #     assert False, "This matrix is singular, cannot do inverse"

#     # compute det with self-defined fuction, reduce time cost a lot.
#     if torch.any(torch_det_2x2(xTx) == 0.0):
#         print("This matrix is singular, cannot do inverse, k_para += 25")
#         assert False
#         return lwlr_torch(testPoint_origin, xArr, yArr, uvArr, test_uvArr, k+25, down_sample_shape, lambda_limit_value, loop=True)
        
#         # assert False, "This matrix is singular, cannot do inverse"
#         # return None, None

#     # time4 = time.time()
        
#     # xTx_inverse = torch.linalg.inv(xTx)
#     xTx_inverse = xTx.inverse()
#     # xTx_inverse = torch.linalg.inv(xTx)
#     # time5 = time.time()
#     weights = torch.matmul(weights, yArr[None, :, None].repeat(n, 1, 1))
#     # time6 = time.time()
#     xArr_transpose = xArr_repeat.permute(0, 2, 1)
#     w = torch.matmul(xTx_inverse, torch.matmul(xArr_transpose, weights))


#     w = w.reshape(1, down_sample_shape[0], down_sample_shape[1], 2).permute(0, 3, 1, 2)
#     # w = cv2.resize(w, (testPoint_origin.shape[1], testPoint_origin.shape[0]))
#     w = F.interpolate(w, (testPoint_origin.shape[0], testPoint_origin.shape[1]), mode='bilinear')
#     w = w.permute(0, 2, 3, 1)
#     w = w.reshape(-1, 2, 1)

    
#     # print('1 :', time2-time1)
#     # print('2 :', time3-time2)
#     # print('3 :', time4-time3)
#     # print('4 :', time5-time4)
#     # print('5 :', time6-time5)
#     # assert False

#     return torch.matmul(testPoint[:, None, :], w).squeeze(), w.squeeze()

def lwlr_torch_batch(testPoint_origin, xArr, yArr, uvArr, test_uvArr, k, down_sample_shape=None, lambda_limit_value=1, loop=False, device=torch.device('cpu')):

    # testPoint_origin = testPoint_origin[None, ...].repeat(2, 1, 1)
    # xArr = xArr[None, ...].repeat(2, 1)
    # yArr = yArr[None, ...].repeat(2, 1)
    # uvArr = uvArr[None, ...].repeat(2, 1, 1)
    # test_uvArr = test_uvArr[None, ...].repeat(2, 1, 1)

    num_batch = testPoint_origin.shape[0]
    num_pixels = test_uvArr.shape[1]
    num_samples = xArr.shape[1]

    time1 = time.time()
    if loop == False:
        # concat ones matrix
        xArr_ones = torch.ones((xArr.shape[0], xArr.shape[1], 1)).to(device)
        if xArr.ndim == 3:
            xArr = torch.cat((xArr_ones, xArr), dim=2)
        elif xArr.ndim == 2:
            xArr = torch.cat((xArr_ones, xArr[..., None]), dim=2)
        else:
            raise ValueError()

        testPoint = testPoint_origin.reshape(num_batch, -1)
        if testPoint.ndim == 2:
            testPoint_ones = torch.ones((testPoint.shape[0], testPoint.shape[1], 1)).to(device)
            testPoint = torch.cat((testPoint_ones, testPoint[..., None]), dim=2)
        else:
            raise ValueError()

    time2 = time.time()
    # diffArr = torch.norm(test_uvArr[:, :, None, :] - uvArr[:, None, :, :], p=2, dim=3) # torch.norm is slower than torch.sqrt
    diffArr = torch.sqrt((test_uvArr[:, :, None, :] - uvArr[:, None, :, :]).pow(2).sum(3))

    weights = torch.exp(diffArr**2 / (-2.0 * k**2))
    weights = torch.diag_embed(weights).view(-1, num_samples, num_samples).to(device)

    xArr_repeat = xArr[:, None, ...].repeat(1, num_pixels, 1, 1).view(-1, num_samples, 2)
    xTx = torch.matmul(xArr_repeat.permute(0, 2, 1), torch.matmul(weights, xArr_repeat))

    lambda_limit = torch.zeros_like(xTx).to(device)
    lambda_limit[:, 0, 0] = lambda_limit_value
    xTx = xTx + lambda_limit

    time3 = time.time()

    # if torch.any(torch.det(xTx) == 0.0):
    #     assert False, "This matrix is singular, cannot do inverse"

    # compute det with self-defined fuction, reduce time cost a lot.
    if torch.any(torch_det_2x2(xTx) == 0.0):
        print("This matrix is singular, cannot do inverse, k_para += 25")
        assert False
        return lwlr_torch(testPoint_origin, xArr, yArr, uvArr, test_uvArr, k+25, down_sample_shape, lambda_limit_value, loop=True)
        
        # assert False, "This matrix is singular, cannot do inverse"
        # return None, None

    time4 = time.time()
    
    # xTx_inverse = xTx.inverse()
    xTx_inverse = torch_inverse_2x2(xTx)

    # prevent inf and nan values
    float_max_thr = 1e20
    if torch.any(xTx_inverse > float_max_thr):
        print('warning! overflow happens in function lwlr_torch_batch, fixed...')
        xTx_inverse[xTx_inverse > float_max_thr] = float_max_thr
    if torch.any(xTx_inverse < -float_max_thr):
        print('warning! overflow happens in function lwlr_torch_batch, fixed...')
        xTx_inverse[xTx_inverse < -float_max_thr] = -float_max_thr

    time5 = time.time()
    # weights = torch.matmul(weights, yArr[:, None, :, None].repeat(1, num_pixels, 1, 1).view(-1, num_samples, 1))
    weights = weights @ yArr[:, None, :, None].repeat(1, num_pixels, 1, 1).view(-1, num_samples, 1)
    time6 = time.time()
    xArr_transpose = xArr_repeat.permute(0, 2, 1)
    # xArr_transpose_weights = torch.matmul(xArr_transpose, weights)
    # w = torch.matmul(xTx_inverse, xArr_transpose_weights)
    xArr_transpose_weights = xArr_transpose @ weights
    w = xTx_inverse @ xArr_transpose_weights


    w = w.reshape(num_batch, down_sample_shape[0], down_sample_shape[1], 2).permute(0, 3, 1, 2)
    w = F.interpolate(w, (testPoint_origin.shape[1], testPoint_origin.shape[2]), mode='bilinear')
    w = w.permute(0, 2, 3, 1)
    w = w.reshape(-1, 2, 1)

    
    # print('1 :', time2-time1)
    # print('2 :', time3-time2)
    # print('3 :', time4-time3)
    # print('4 :', time5-time4)
    # print('5 :', time6-time5)
    # assert False

    output = torch.matmul(testPoint.view(-1, 1, 2), w).squeeze().view(num_batch, testPoint_origin.shape[1], testPoint_origin.shape[2])

    return output, w.squeeze().view(2, -1, 2)



def sparse_depth_lwlr_batch(pred_mono, guided_depth, sample_mode='grid', sample_num=100, down_sample_scale=32, lambda_limit_value=1, dataset_name=None, k_para=None, device=torch.device('cpu')):
    '''
    pred_mono:          monocular depth estimation, with shape: [H, W]
    guided_depth:       sparse guided points, from gt or low quality sensors, with shape: [H, W] and 0 for invalid area.
    sample_mode:        'grid' or 'uniform'. Default: 'grid'
    sample_num:         sample number of sparse guided points. Default: 100
    down_sample_scale:  down sample ratio for reducing computation. Default: 32
    lambda_limit_value: extent for limiting shift values during performing lwlr. Default: 1
    dataset_name:       dataset name, special sampling area for 'kitti' dataset, and default k_para for several datasets. Default: None
    k_para:             k parameters during performing lwlr, default k_para from dataset_name will be replaced if inputed by users. Default: None
    '''
    time1 = time.time()
    assert sample_mode in ['grid', 'uniform']
    if dataset_name is not None:
        dataset_name = dataset_name.lower()
        
    # pred_mono = pred_mono.squeeze()
    # guided_depth = guided_depth.squeeze()

    # pred_mono = pred_mono[None, ...].repeat(10, 1, 1)
    # guided_depth = guided_depth[None, ...].repeat(10, 1, 1)

    num_batch = pred_mono.shape[0]
    target = guided_depth

    # mask = (target > 1e-8)
    # TODO: mask = (target != 0) only suitable for reconstruction optimization
    mask = (target != 0)

    u = torch.arange(target.shape[1]).to(device)
    v = torch.arange(target.shape[2]).to(device)
    v, u = torch.meshgrid(v, u)
    u = u.transpose(1, 0)
    v = v.transpose(1, 0)
    uvArr = torch.cat((u[mask[0]][..., None], v[mask[0]][..., None]), dim=1)
    time2 = time.time()

    prediction = pred_mono
    
    # print(mask.shape)
    # print(mask.sum())
    # print(num_batch)
    # print(prediction.shape)
    uvArr = uvArr.float()
    xArr = prediction[mask].view(num_batch, -1)
    yArr = target[mask].view(num_batch, -1)

    down_sample_scale = down_sample_scale
    mask_test = (target != target)[0]
    mask_test[
        ((u + down_sample_scale // 2) % down_sample_scale == 0)
        & ((v + down_sample_scale // 2) % down_sample_scale == 0)
    ] = True
    down_sample_shape0 = (
        (u[:, 0] + down_sample_scale // 2) % down_sample_scale == 0
    ).sum()
    down_sample_shape1 = (
        (v[0] + down_sample_scale // 2) % down_sample_scale == 0
    ).sum()
    down_sample_shape = (down_sample_shape0, down_sample_shape1)
    test_uvArr = torch.cat(
        (u[mask_test][..., None], v[mask_test][..., None]), dim=1
    ).reshape(-1, 2)
    

    if type(k_para) == int:
        k_para = k_para
    elif "kitti" == dataset_name:
        k_para = 25
    elif "eth3d" == dataset_name:
        k_para = 125
    elif type(dataset_name) == str and dataset_name.startswith("diode"):
        k_para = 100
    else:
        k_para = 50
    
    time3 = time.time()

    uvArr = uvArr[None, ...].repeat(num_batch, 1, 1)
    test_uvArr = test_uvArr[None, ...].repeat(num_batch, 1, 1)

    prediction_lwlr, w_squeeze = lwlr_torch_batch(
        prediction,
        xArr,
        yArr,
        uvArr,
        test_uvArr,
        k=k_para,
        down_sample_shape=down_sample_shape,
        lambda_limit_value=lambda_limit_value,
        device=device,
    )

    time4 = time.time()

    # print('1 :', time2-time1)
    # print('2 :', time3-time2)
    # print('3 :', time4-time3)
    # assert False

    # # visualization of scale and shift map
    # shift_value = w_squeeze[:, 0].reshape(target.shape[0], target.shape[1])
    # scale_value = w_squeeze[:, 1].reshape(target.shape[0], target.shape[1])

    # shift_viz = (shift_value - shift_value.min()) / (shift_value.max() - shift_value.min()) * 255
    # scale_viz = (scale_value - scale_value.min()) / (scale_value.max() - scale_value.min()) * 255

    # shift_viz = shift_viz.detach().cpu().numpy()
    # scale_viz = scale_viz.detach().cpu().numpy()

    # shift_viz = np.repeat(shift_viz[..., None], 3, axis=2)
    # scale_viz = np.repeat(scale_viz[..., None], 3, axis=2)
    # shift_viz = cv2.applyColorMap(shift_viz.astype(np.uint8), cv2.COLORMAP_JET)
    # scale_viz = cv2.applyColorMap(scale_viz.astype(np.uint8), cv2.COLORMAP_JET)

    # # viz_scale = 0.8
    # # shift_viz = rgb * (1 - viz_scale) + shift_viz * viz_scale
    # # scale_viz = rgb * (1 - viz_scale) + scale_viz * viz_scale

    # cv2.imwrite('temp_shift_viz.png', shift_viz)
    # cv2.imwrite('temp_scale_viz.png', scale_viz)

    return prediction_lwlr


def fill_grid_sparse_depth_torch_batch(pred_mono, sparse_guided_points, fill_coords=None, step_num=None, device=torch.device('cpu')):
    '''
    pred_mono: [n, H, W]
    sparse_guided_points: [n, 10, 10]
    '''
    num_batch = pred_mono.shape[0]
    # pred_mono = pred_mono.squeeze()
    if step_num is None:
        step_num = sparse_guided_points.shape[-1]
    # step_num = 10
    sparse_guided_depth_map = torch.zeros_like(pred_mono).to(device)
    if fill_coords is None:
        u_step = torch.arange(
            0 + (pred_mono.shape[1] / step_num) // 2,
            pred_mono.shape[1],
            pred_mono.shape[1] / step_num,
        ).to(device)
        v_step = torch.arange(
            0 + (pred_mono.shape[2] / step_num) // 2,
            pred_mono.shape[2],
            pred_mono.shape[2] / step_num,
        ).to(device)
        v_step, u_step = torch.meshgrid(v_step, u_step)
        u_step = u_step.transpose(1, 0)
        v_step = v_step.transpose(1, 0)
        uvArr_step = torch.cat(
            (u_step.reshape(-1, 1), v_step.reshape(-1, 1)), dim=1
        ).long()

        if sparse_guided_points.shape == pred_mono.shape:
            # TODO: how to batch this process?
            target = sparse_guided_points
            masks = (target > 1e-8)
            for i in range(num_batch):
                mask = masks[i]
                u = np.arange(target.shape[1])
                v = np.arange(target.shape[2])
                u, v = np.meshgrid(u, v)
                u = u.transpose(1, 0)
                v = v.transpose(1, 0)
                uvArr = np.concatenate((u[mask][..., None], v[mask][..., None]), axis=1)
                kdtree = KDTree(uvArr)
                mask1 = kdtree.query(uvArr_step.detach().cpu().numpy())[1]

                depth_sparse = np.zeros_like(pred_mono[0].detach().cpu().numpy())
                index = np.argwhere(mask.reshape(-1) == True).reshape(-1)
                index = index[mask1]
                depth_sparse = depth_sparse.reshape(-1)
                depth_sparse[index] = sparse_guided_points[i][mask][mask1]
                depth_sparse = depth_sparse.reshape(sparse_guided_depth_map.shape[1], sparse_guided_depth_map.shape[2])
                sparse_guided_depth_map[i] = torch.from_numpy(depth_sparse).to(sparse_guided_depth_map.device)
        else:
            sparse_guided_depth_map[:, uvArr_step[:, 0], uvArr_step[:, 1]] = sparse_guided_points.view(num_batch, -1)

    else:
        uvArr_step = fill_coords.long()
        sparse_guided_depth_map[:, uvArr_step[:, 0], uvArr_step[:, 1]] = sparse_guided_points.view(num_batch, -1)

    return sparse_guided_depth_map


if __name__ == '__main__':
    
    pred_mono_depth = np.load('./pred_mono_depth.npy')
    gt_depth = cv2.imread('./gt_depth.png', -1) / 5000.

    pred_depth_global = recover_metric_depth(pred_mono_depth, gt_depth)
    pred_depth_lwlr = sparse_depth_lwlr(pred_mono_depth, gt_depth, sample_mode='grid', sample_num=100)

    absrel_global = absrel_single(pred_depth_global, gt_depth)
    absrel_lwlr = absrel_single(pred_depth_lwlr, gt_depth)
    print('absrel_global :', absrel_global)
    print('absrel_lwlr :', absrel_lwlr)
