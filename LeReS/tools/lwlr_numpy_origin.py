import numpy as np
import cv2
import random

try:
    import torch
    torch_imported = True
except:
    torch_imported = False


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

def lwlr(testPoint_origin, xArr, yArr, uvArr, test_uvArr, k, down_sample_shape=None, lambda_limit_value=1):

    # concat ones matrix
    if xArr.ndim == 2:
        xArr_ones = np.ones((xArr.shape[0], 1))
        xArr = np.concatenate((xArr_ones, xArr), axis=1)
    elif xArr.ndim == 1:
        xArr_ones = np.ones((xArr.shape[0], 1))
        xArr = np.concatenate((xArr_ones, xArr[:, None]), axis=1)
    else:
        raise ValueError()

    testPoint = testPoint_origin.reshape(-1)
    if testPoint.ndim == 1:
        testPoint_ones = np.ones((testPoint.shape[0], 1))
        testPoint = np.concatenate((testPoint_ones, testPoint[:, None]), axis=1)
    else:
        raise ValueError()

    m = np.shape(xArr)[0]
    n = test_uvArr.shape[0]
    weights = np.eye(m)[None, ...].repeat(n, axis=0)
    for j in range(m):
        diffArr = np.linalg.norm(test_uvArr - uvArr[j, :], ord=2, axis=1)  # L2 distance
        weights[:, j, j] = np.exp(diffArr**2 / (-2.0 * k**2))  # weighted matrix
    xArr = xArr[None, ...].repeat(n, axis=0)
    xTx = np.matmul(xArr.transpose(0, 2, 1), np.matmul(weights, xArr))

    lambda_limit = np.zeros_like(xTx)
    lambda_limit[:, 0, 0] = lambda_limit_value
    xTx = xTx + lambda_limit

    if np.any(np.linalg.det(xTx) == 0.0):
        print("This matrix is singular, cannot do inverse")
        return
    xTx_inverse = np.linalg.inv(xTx)
    weights = np.matmul(weights, yArr[None, :, None].repeat(n, axis=0))
    xArr_transpose = xArr.transpose(0, 2, 1)
    w = np.matmul(xTx_inverse, np.matmul(xArr_transpose, weights))

    w = w.reshape(down_sample_shape[0], down_sample_shape[1], 2)
    w = cv2.resize(w, (testPoint_origin.shape[1], testPoint_origin.shape[0]))
    w = w.reshape(-1, 2, 1)

    return np.matmul(testPoint[:, None, :], w).squeeze(), w.squeeze()



def sparse_depth_lwlr(pred_mono, guided_depth, sample_mode='grid', sample_num=100, down_sample_scale=32, lambda_limit_value=1, dataset_name=None, k_para=None):
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
    assert sample_mode in ['grid', 'uniform']
    if dataset_name is not None:
        dataset_name = dataset_name.lower()

    if type(pred_mono).__module__ == torch.__name__ and torch_imported:
        pred_mono = pred_mono.cpu().numpy()
    pred_mono = cv2.resize(
        pred_mono, (guided_depth.shape[1], guided_depth.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    pred_mono = pred_mono.squeeze()
    target = guided_depth.squeeze()

    # valid gt for error smaller than 15%
    pred_global_temp = recover_metric_depth(pred_mono, target) # global fitted depth prediction
    pred_global_temp = pred_global_temp.squeeze()
    abs_rel_err_global_temp = np.abs(pred_global_temp - target) / (np.abs(target) + 1e-8)
    mask = (target > 1e-8) & (abs_rel_err_global_temp < 0.15)

    u = np.arange(target.shape[0])
    v = np.arange(target.shape[1])
    u, v = np.meshgrid(u, v)
    u = u.transpose(1, 0)
    v = v.transpose(1, 0)
    uvArr = np.concatenate((u[mask][..., None], v[mask][..., None]), axis=1)

    if sample_mode == 'uniform':
        mask1 = random.sample(range(0, uvArr.shape[0]), sample_num)
    elif sample_mode == 'grid':
        step_num = int(pow(sample_num, 0.5))
        if "kitti" == dataset_name:
            u_step = np.arange(
                target.shape[0] / 3 + (target.shape[0] * 2 / 3 / step_num) // 2,
                target.shape[0],
                target.shape[0] * 2 / 3 / step_num,
            )
        else:
            u_step = np.arange(
                0 + (target.shape[0] / step_num) // 2,
                target.shape[0],
                target.shape[0] / step_num,
            )
        v_step = np.arange(
            0 + (target.shape[1] / step_num) // 2,
            target.shape[1],
            target.shape[1] / step_num,
        )
        u_step, v_step = np.meshgrid(u_step, v_step)
        u_step = u_step.transpose(1, 0)
        v_step = v_step.transpose(1, 0)
        uvArr_step = np.concatenate(
            (u_step.reshape(-1, 1), v_step.reshape(-1, 1)), axis=1
        )
        distance = np.linalg.norm(uvArr[:, None, :] - uvArr_step[None, ...], ord=2, axis=2)
        mask1 = np.argmin(distance, axis=0)

    # sparse depth
    depth_sparse = np.zeros_like(guided_depth)
    index = np.argwhere(mask.reshape(-1) == True).reshape(-1)
    index = index[mask1]
    depth_sparse = depth_sparse.reshape(-1)
    depth_sparse[index] = guided_depth[mask][mask1]
    depth_sparse = depth_sparse.reshape(guided_depth.shape[0], guided_depth.shape[1])
    prediction = recover_metric_depth(pred_mono, depth_sparse).squeeze() # global fitting with sparse guided depth

    uvArr = uvArr[mask1]
    xArr = prediction[mask][mask1]
    yArr = target[mask][mask1]

    down_sample_scale = down_sample_scale
    mask_test = target != target
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
    test_uvArr = np.concatenate(
        (u[mask_test][..., None], v[mask_test][..., None]), axis=1
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

    prediction_lwlr, w_squeeze = lwlr(
        prediction,
        xArr,
        yArr,
        uvArr,
        test_uvArr,
        k=k_para,
        down_sample_shape=down_sample_shape,
        lambda_limit_value=lambda_limit_value,
    )
    prediction_lwlr = prediction_lwlr.squeeze().reshape(
        target.shape[0], target.shape[1]
    )

    # # visualization of scale and shift map
    # shift_value = w_squeeze[:, 0].reshape(target.shape[0], target.shape[1])
    # scale_value = w_squeeze[:, 1].reshape(target.shape[0], target.shape[1])

    # shift_viz = (shift_value - shift_value.min()) / (shift_value.max() - shift_value.min()) * 255
    # scale_viz = (scale_value - scale_value.min()) / (scale_value.max() - scale_value.min()) * 255

    # shift_viz = np.repeat(shift_viz[..., None], 3, axis=2)
    # scale_viz = np.repeat(scale_viz[..., None], 3, axis=2)
    # shift_viz = cv2.applyColorMap(shift_viz.astype(np.uint8), cv2.COLORMAP_JET)
    # scale_viz = cv2.applyColorMap(scale_viz.astype(np.uint8), cv2.COLORMAP_JET)
    # viz_scale = 0.8
    # shift_viz = rgb * (1 - viz_scale) + shift_viz * viz_scale
    # scale_viz = rgb * (1 - viz_scale) + scale_viz * viz_scale

    return prediction_lwlr

if __name__ == '__main__':
    
    pred_mono_depth = np.load('./pred_mono_depth.npy')
    gt_depth = cv2.imread('./gt_depth.png', -1) / 5000.

    pred_depth_global = recover_metric_depth(pred_mono_depth, gt_depth)
    pred_depth_lwlr = sparse_depth_lwlr(pred_mono_depth, gt_depth, sample_mode='grid', sample_num=100)

    absrel_global = absrel_single(pred_depth_global, gt_depth)
    absrel_lwlr = absrel_single(pred_depth_lwlr, gt_depth)
    print('absrel_global :', absrel_global)
    print('absrel_lwlr :', absrel_lwlr)
