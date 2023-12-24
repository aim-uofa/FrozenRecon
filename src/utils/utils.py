import xlwt
import torch
import torch.nn.functional as F
import cv2
import os
import os.path as osp
import numpy as np
import re

from utils.lwlr_function_batch import sparse_depth_lwlr_batch, fill_grid_sparse_depth_torch_batch

def microsoft_sorted(input_list):
    return sorted(input_list, key=lambda s: [int(s) if s.isdigit() else s for s in sum(re.findall(r'(\D+)(\d+)', 'a'+s+'0'), ())])

def img_numpy_to_torch(rgb_img):
    rgb_img = np.transpose(rgb_img, (2, 0, 1))
    rgb_img = torch.from_numpy(rgb_img).float()/255
    return rgb_img

class Excel_Editor:
    def __init__(self, sheet_name, title) -> None:
        self.workbook = xlwt.Workbook(encoding='utf-8')
        self.sheet = self.workbook.add_sheet(sheet_name)
        self.line_cnt = 0
        self.title = title
        for col, column in enumerate(title):
            self.sheet.write(0, col, column)
        self.line_cnt += 1

    def add_data(self, data):
        assert type(data) == list and len(data) == len(self.title)
        for col, colume_data in enumerate(data):
            self.sheet.write(self.line_cnt, col, colume_data)
        self.line_cnt += 1
        
    def save_excel(self, save_path):
        self.workbook.save(save_path)
        print('excel saved to %s' %save_path)


class FeatureSimComputer:
    def __init__(self, fmaps, max_width=40) -> None:
        '''
        fmaps: [n, c, h, w], feature maps of depth model.
        '''
        assert max_width % 2 == 0
        self.fmaps = F.normalize(fmaps, dim=1)
        self.max_width = max_width

    def calculate_similarity(self):
        '''
        similarity_all: [n, max_width+1], n stands for num of imgs.
                        similarity_all[i, max_width // 2] stands for the similarity to itsel]
                        max_width means contains left (max_width//2) imgs and right (max_width//2) imgs
        '''
        n, c, h, w = self.fmaps.shape
        
        similarity_all = []
        for i in range(n):
            half_width = self.max_width // 2
            left = max(i - half_width, 0)
            right = min(i + half_width, n - 1)
            left_index = left - i + half_width
            right_index = right - i + half_width

            fmaps1 = self.fmaps[i:(i+1)].repeat(self.max_width + 1, 1, 1, 1)
            fmaps2 = torch.zeros_like(fmaps1).to(self.fmaps.device)

            valid_mask = torch.zeros(self.max_width + 1).bool()
            valid_mask[left_index:(right_index+1)] = True
            fmaps2[valid_mask] = self.fmaps[left:(right+1)]
            corr = self.corr(fmaps1, fmaps2) # [max_depth+1, h, w]
            corr[~valid_mask] = 0
            similarity1 = corr.max(dim=1)[0].mean(dim=1)
            similarity2 = corr.max(dim=2)[0].mean(dim=1)
            similarity = torch.minimum(similarity1, similarity2)
            similarity_all.append(similarity)
        similarity_all = torch.stack(similarity_all, dim=0)
        
        return similarity_all

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht*wd, ht*wd)
        return corr

def double_pointer_from_similarity(p):
    length = p.shape[0]
    assert (length - 1) % 2 == 0
    half_width = (p.shape[0] - 1) // 2

    # double_pointer
    i = 1
    left = (half_width - i)
    while left >= 0:
        p_left = p[left]
        if p_left == 0:
            break
        i += 1
        left = (half_width - i)
    left += 1
    
    i = 1
    right = (half_width + i)
    while right <= (length - 1):
        p_right = p[right]
        if p_right == 0:
            break
        i += 1
        right = (half_width + i)
    right -= 1

    left -= half_width
    right -= half_width

    return left, right

def image_indexes_from_sim_matrix(sim_matrix_bool):
    num_images = sim_matrix_bool.shape[0]
    image_indexes = [0]
    image_index = 0

    while True:
        _, right = double_pointer_from_similarity(sim_matrix_bool[image_index])
        right += image_index
        if right > (num_images - 1) or image_index == (num_images - 1):
            break
        elif len(image_indexes) != 0 and (right <= image_indexes[-1]):
            image_index += 1
            image_indexes.append(image_index)
        else:
            div_num = 5
            success = False
            for i in range(div_num):
                temp = (right - image_index) * (i + 1) // div_num + image_index
                if ((len(image_indexes) == 0) or temp > image_indexes[-1]):
                    if temp != 0:
                        success = True
                        image_indexes.append(temp)
            if success == True:
                image_index = image_indexes[-1]
            else:
                image_index = 1
                image_indexes.append(image_index)
        
    return image_indexes


def extract_frames(videoPath, save_root, downsample_ratio):
    def getFrame(videoPath, svPath, downsample_ratio=1):
        cap = cv2.VideoCapture(videoPath)
        numFrame = 0
        while True:
            if cap.grab():
                flag, frame = cap.retrieve()
                if not flag:
                    continue
                else:
                    numFrame += 1
                    if numFrame % downsample_ratio == 0:
                        img_id = numFrame // downsample_ratio
                        newPath = svPath + '/' + str(img_id).zfill(6) + ".jpg"
                        cv2.imencode('.jpg', frame)[1].tofile(newPath)
            else:
                break
    
    video_name = osp.basename(videoPath)
    saveResultPath = save_root
    if not os.path.exists(saveResultPath): 
        os.makedirs(saveResultPath)
        getFrame(videoPath, saveResultPath, downsample_ratio)
    else:
        print('video frames exists...')


def torch_det_2x2(tensor):
    '''
    tensor: torch.tensor, [B, 2, 2]
    '''

    a = tensor[:, 0, 0]
    b = tensor[:, 0, 1]
    c = tensor[:, 1, 0]
    d = tensor[:, 1, 1]

    return a * d - b * c

def torch_inverse_3x3(tensor):
    '''
    tensor: torch.tensor, [B, 3, 3]
    [ a1, b1, c1
      a2, b2, c2
      a3, b3, c3 ]
    '''

    a1 = tensor[:, 0, 0]; b1 = tensor[:, 0, 1]; c1 = tensor[:, 0, 2]
    a2 = tensor[:, 1, 0]; b2 = tensor[:, 1, 1]; c2 = tensor[:, 1, 2]
    a3 = tensor[:, 2, 0]; b3 = tensor[:, 2, 1]; c3 = tensor[:, 2, 2]

    coefficient = 1 / (a1 * (b2 * c3 - c2 * b3) - a2 * (b1 * c3 - c1 * b3) + a3 * (b1 * c2 - c1 * b2))

    tensor_new = torch.stack([ 
        (b2 * c3 - c2 * b3), (c1 * b3 - b1 * c3), (b1 * c2 - c1 * b2),
        (c2 * a3 - a2 * c3), (a1 * c3 - c1 * a3), (a2 * c1 - a1 * c2),
        (a2 * b3 - b2 * a3), (b1 * a3 - a1 * b3), (a1 * b2 - a2 * b1),
    ])
    tensor_new = tensor_new.permute(1, 0).view(-1, 3, 3)
    tensor_new = tensor_new * coefficient[:, None, None]

    return tensor_new


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

