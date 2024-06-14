import os
import cv2
import toml
import argparse
import numpy as np
import json

import torch
from torch.nn import functional as F

from tqdm import tqdm
import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import numpy as np
import torch
import random
from easydict import EasyDict

CONFIG = EasyDict({})
CONFIG.benchmark = EasyDict({})
# Model config
CONFIG.benchmark.him2k_img = '/path/to/data/HIM2K_/images/natural'
CONFIG.benchmark.him2k_alpha = '/path/to/data/HIM2K_/alphas/natural'
CONFIG.benchmark.him2k_comp_img = '/path/to/data/HIM2K/images/comp'
CONFIG.benchmark.him2k_comp_alpha = '/path/to/data/HIM2K/alphas/comp'
CONFIG.benchmark.rwp636_img = '/path/to/data/RealWorldPortrait-636/image'
CONFIG.benchmark.rwp636_alpha = '/path/to/data/RealWorldPortrait-636/alpha'
CONFIG.benchmark.ppm100_img = '/path/to/data/PPM-100/image_train'
CONFIG.benchmark.ppm100_alpha = '/path/to/data/PPM-100/matte_train'
CONFIG.benchmark.pm10k_img = '/path/to/data/P3M-10k/validation/P3M-500-NP/original_image'
CONFIG.benchmark.pm10k_alpha = '/path/to/data/P3M-10k/validation/P3M-500-NP/mask'
CONFIG.benchmark.am2k_img = '/path/to/data/AM-2K/validation/original'
CONFIG.benchmark.am2k_alpha = '/path/to/data/AM-2K/validation/mask'
CONFIG.benchmark.rw100_img = '/path/to/data/RefMatte_RW_100/image_copy'
CONFIG.benchmark.rw100_alpha = '/path/to/data/RefMatte_RW_100/mask'
CONFIG.benchmark.rw100_text = '/path/to/data/RefMatte_RW_100/refmatte_rw100_label.json'
CONFIG.benchmark.rw100_index = '/path/to/data/RefMatte_RW_100/eval_index_expression.json'
CONFIG.benchmark.aim_img = '/path/to/data/AIM-500/original'
CONFIG.benchmark.aim_alpha = '/path/to/data/AIM-500/mask'

# Dataloader config
CONFIG.data = EasyDict({})
# feed forward image size (untested)
CONFIG.data.crop_size = 448
# composition of two foregrounds, affine transform, crop and HSV jitter
CONFIG.data.cutmask_prob = 0.25
CONFIG.data.augmentation = True
CONFIG.data.random_interp = True



def GenPoint(_target):
    radius = 20

    foreground_points = np.where(_target > 0.8)
    point_mask = np.zeros(_target.shape, dtype=_target.dtype)
    # 选择要标记的部分点，并将它们标记为1
    points = 10
    selected_points = np.random.choice(len(foreground_points[0]), size=min(len(foreground_points[0]),points), replace=False)
    for idx in selected_points:
        x, y = foreground_points[0][idx], foreground_points[1][idx]
        # 在半径范围内将点标记为1
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= x + i < _target.shape[0] and 0 <= y + j < _target.shape[1]:
                    point_mask[x + i, y + j] = 1

    return point_mask


def GenBox(_target, args):



    if args.benchmark == 'rw100':
        thres = 0.8
    else: 
        thres = 0

    foreground_points = np.argwhere(_target > thres)

    # 获取左上、右上、左下和右下点的坐标
    if len(foreground_points) > 0:
        min_x, min_y = np.min(foreground_points, axis=0)
        max_x, max_y = np.max(foreground_points, axis=0)

        
        # 计算当前边界框的宽度和高度
        width = max_x - min_x
        height = max_y - min_y
        if args.benchmark == 'rw100':
            mul = 0.8
        else: 
            mul = 1
        # 缩小边界框的宽度和高度为0.9倍
        new_width = int(width * mul)
        new_height = int(height * mul)

        # 计算新的 min_x, min_y 和 max_x, max_y 值
        new_min_x = min_x + (width - new_width) // 2
        new_min_y = min_y + (height - new_height) // 2
        new_max_x = new_min_x + new_width
        new_max_y = new_min_y + new_height

        # 创建一个形状与foreground_mask相同的新掩码，初始值为0
        box_mask = np.zeros(_target.shape, dtype=np.uint8)

        # 在新掩码上将盒子内的点值设置为1
        box_mask[new_min_x:new_max_x, new_min_y:new_max_y] = 1

        return box_mask
    else:
        return np.zeros(_target.shape, dtype=np.uint8)

def GenScribble(_target):
    kernel = 5
    if np.max(_target) == 0:
        scribble = np.zeros(_target.shape, dtype=_target.dtype)
    else:
        scribble_points = generate_scribble_strictly(_target, num_c=random.randint(3, 4))
        scribble_map = np.zeros(_target.shape).astype(np.float32)

        for point in scribble_points:
            scribble_map[point[1], point[0]] = 1

        kernel_size = kernel
        dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        scribble = cv2.dilate(scribble_map, dilate_kernel)
    return scribble

def B_spline(control_points, num_i, s=0.5):
    '''
    Using B_spline to interpolate
    args
        control_points: list of control_points
        num_i: number of interpolation points(between two control points or end points)
        s: hyper parameter for b-spline
    return
        points: list of interpolated points
    '''
    points = []
    num_c = len(control_points)
    for i in range(num_c):
        for t in range(num_i):
            i0 = max(0, i - 1)
            i1 = i
            i2 = min(num_c - 1, i + 1)
            i3 = min(num_c - 1, i + 2)
            f = t * 1.0 / num_i
            c0 = (1.0 / 3 - s) * (f ** 3) + (2 * s - 1.0 / 2) * (f ** 2) - s * (f) + 1.0 / 6
            c1 = (1 - s) * (f ** 3) + (s - 3.0 / 2) * (f ** 2) + 2.0 / 3
            c2 = (s - 1) * (f ** 3) + (3.0 / 2 - 2 * s) * (f ** 2) + s * (f) + 1.0 / 6
            c3 = (s - 1.0 / 3) * (f ** 3) + (1.0 / 2 - s) * (f ** 2)
            tmp_point = control_points[i0] * c0 + control_points[i1] * c1 + \
                        control_points[i2] * c2 + control_points[i3] * c3
            points.append(tmp_point.astype('int'))
    return points

def generate_scribble_strictly(mask, num_c=3, num_i=100, coverage_area=0.1, width=10, best_out_of=5):
    '''
    generate one B-spline with 2 end points and several control points to be a scribble
    args 
        mask: 2D np.array shape: H x W dtype bool(1 for target mask, 0 for others)
        num_c: number of control points (points except for the two end points)
        num_i: number of interpolation points(between two control points or end points)
    return 
        scribble points: 2D np.array shape:  L(number of points) x 2 (0 for x, 1 for y)
    '''
    H, W = mask.shape
    mask_points = np.where(mask > 0)
    mask_points = np.array([mask_points[1], mask_points[0]])
    num_mask_points = mask_points.shape[1]
    total_area = mask.sum()
    max_coverage = 0
    best_scribbles = []
    num_of_candidates = 0
    number_of_out_of_bound = 0
    while (num_of_candidates < best_out_of):
        scribble_points = []
        for i in range(num_c):
            sample_index = int(np.random.rand() * num_mask_points)
            control_points = mask_points[:, sample_index]
            scribble_points.append(control_points)
        scribble_points = B_spline(scribble_points, num_i)

        # check out_of_bound_point
        new_scribble_points = []
        out_of_bound = False
        for i in range(len(scribble_points)):
            if mask[scribble_points[i][1], scribble_points[i][0]] < 1 and number_of_out_of_bound < 20:
                out_of_bound = True
                break
            else:
                new_scribble_points.append(scribble_points[i])
        if out_of_bound:
            number_of_out_of_bound += 1
            continue
        number_of_out_of_bound = 0

        # remove duplicate points
        num_of_candidates += 1
        scribble_points = np.array(new_scribble_points)
        # scribble_points = np.unique(scribble_points, axis=0)

        remain_mask = mask.copy()
        for i in range(len(scribble_points)):
            x = scribble_points[i, 0]
            y = scribble_points[i, 1]
            t = max(0, y - width)
            b = min(H - 1, y + width)
            l = max(0, x - width)
            r = min(W - 1, x + width)
            remain_mask[t:b, l:r] = 0
        remain_area = remain_mask.sum()
        if (1 - remain_area * 1.0 / total_area) > max_coverage:
            max_coverage = (1 - remain_area * 1.0 / total_area)
            best_scribbles = scribble_points
    return best_scribbles


# 定义 GenerateScribble 类
class GenerateScribble(object):
    """
    Returns the scribble generated according to gt
    dilate: dilate kernel size
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, kernel=5, elem='gt'):
        self.kernel = kernel
        self.elem = elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('GenerateScribble not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            sample['scribble'] = np.zeros(_target.shape, dtype=_target.dtype)
        else:
            scribble_points = generate_scribble_strictly(_target, num_c=random.randint(3, 4))
            scribble_map = np.zeros(_target.shape).astype(np.float32)

            for point in scribble_points:
                scribble_map[point[1], point[0]] = 1

            if self.kernel > 1:
                kernel_size = self.kernel
            else:
                bbox = sample['meta']['boundary']
                length_short_side = min(bbox[0] - bbox[1], bbox[2] - bbox[3])
                kernel_size = self.kernel * length_short_side
            dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
            sample['prompt'] = cv2.dilate(scribble_map, dilate_kernel)
        return sample

    def __str__(self):
        return 'GenerateScribble:(kernel=' + str(self.kernel) + ', elem=' + str(self.elem) + ')'  

def single_ms_inference(model, image_dict, args):

    with torch.no_grad():
        for k in image_dict.keys():
            if k == 'image_name' or k=='height' or k=='width':
                continue
            else:
                image_dict[k].to(model.device)
        output_ = model(image_dict)
        output = output_['phas']
        output = torch.nn.functional.interpolate(output, (image_dict['height'], image_dict['width']))
        output = output.flatten(0, 2)
        

        
        output = (output*255).detach().cpu().numpy()

        output = output.astype('uint8')
    return output

def generator_tensor_dict(image_path, alpha_path, args):
    # read images
    imgs = Image.open(image_path)
    alpha = cv2.imread(alpha_path,0)/255
    width, height = imgs.size
    sample = {}
    w, h = imgs.size
    new_size = 1560
    if w > new_size and h > new_size:
        aspect_ratio = w / h
        if w > h:
            new_w = new_size
            new_h = int(new_size / aspect_ratio)
        else:
            new_h = new_size
            new_w = int(new_size * aspect_ratio)
        imgs = imgs.resize((new_w, new_h))
        # 使用resize方法缩放图像
        alpha = cv2.resize(alpha, (new_w, new_h))
        
    if args.prompt=='box':
        sample['prompt'] = GenBox(alpha,args)
            
        
    if args.prompt =='point':
        sample['prompt'] = GenPoint(alpha)
    if args.prompt =='scribble':
        sample['prompt'] = GenScribble(alpha)
    if args.prompt == 'none':
        sample['prompt'] = np.zeros_like(alpha)
        
    sample['image'] = F.to_tensor(imgs).unsqueeze(0)
    sample['prompt'] = torch.from_numpy(sample['prompt']).unsqueeze(0).unsqueeze(0)
    sample['trimap'] = sample['prompt']
    sample['height'] = height
    sample['width'] = width
    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ViTMatte_S_100ep_no_trimap_dino_prompt_dataset_prompt_sali.py')
    parser.add_argument('--benchmark', type=str, default='aim', choices=['him2k', 'him2k_comp', 'rwp636', 'p3m500', 'ppm100', 'am2k', 'pm10k', 'rw100', 'aim'])
    parser.add_argument('--checkpoint', type=str, default='output_of_train/dino_frozen_him2k_add_all_45(44_incomplete)/model_0016400.pth',
                        help="path of checkpoint")
    parser.add_argument('--image-ext', type=str, default='.jpg', help="input image ext")
    parser.add_argument('--mask-ext', type=str, default='.png', help="input mask ext")
    parser.add_argument('--output', type=str, default='outputs/am2k-box', help="output dir")
    parser.add_argument('--os8_width', type=int, default=10, help="guidance threshold")
    parser.add_argument('--os4_width', type=int, default=20, help="guidance threshold")
    parser.add_argument('--os1_width', type=int, default=10, help="guidance threshold")
    parser.add_argument('--twoside', action='store_true', default=False, help='post process with twoside of the guidance')        
    parser.add_argument('--sam', action='store_true', default=False, help='return mask')    
    parser.add_argument('--maskguide', action='store_true', default=False, help='mask guidance')    
    parser.add_argument('--alphaguide', action='store_true', default=False, help='alpha guidance')    
    parser.add_argument('--prompt', type=str, default='box', choices=['box', 'point', 'text', 'scribble', 'none'])
    parser.add_argument('--device', type=str, default='cuda')

    # Parse configuration
    args = parser.parse_args()


    # Check if toml config file is loaded

    args.output = os.path.join(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # build model
    cfg = LazyConfig.load(args.config)
    model = instantiate(cfg.model)
    model.to(args.device)
    model.eval()
    DetectionCheckpointer(model).load(args.checkpoint)

    # inference
    model = model.eval()
    n_parameters = sum(p.numel() for p in model.parameters() )
    print('number of params:', n_parameters)
    
    
    if args.benchmark == 'him2k':
        image_dir = CONFIG.benchmark.him2k_img
        alpha_dir = CONFIG.benchmark.him2k_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0])
            output_path = os.path.join(args.output, os.path.splitext(image_name)[0])
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for alpha_single_dir in sorted(os.listdir(alpha_path)):
                alpha_single_path = os.path.join(alpha_path, alpha_single_dir)
                image_dict = generator_tensor_dict(image_path, alpha_single_path, args)
                alpha_pred = single_ms_inference(model, image_dict, args)
                cv2.imwrite(os.path.join(output_path, alpha_single_dir), alpha_pred)
            
    elif args.benchmark == 'him2k_comp':
        image_dir = CONFIG.benchmark.him2k_comp_img
        alpha_dir = CONFIG.benchmark.him2k_comp_alpha
    
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0])
            output_path = os.path.join(args.output, os.path.splitext(image_name)[0])

            for alpha_single_dir in sorted(os.listdir(alpha_path)):
                alpha_single_path = os.path.join(alpha_path, alpha_single_dir)
                image_dict = generator_tensor_dict(image_path, alpha_single_path, args)
                alpha_pred = single_ms_inference(model, image_dict, args)
                cv2.imwrite(os.path.join(output_path, alpha_single_dir), alpha_pred)

    elif args.benchmark == 'rwp636':
        image_dir = CONFIG.benchmark.rwp636_img
        alpha_dir = CONFIG.benchmark.rwp636_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'ppm100':
        image_dir = CONFIG.benchmark.ppm100_img
        alpha_dir = CONFIG.benchmark.ppm100_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, image_name)
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)
            
    elif args.benchmark == 'p3m500':
        image_dir = CONFIG.benchmark.p3m500_img
        alpha_dir = CONFIG.benchmark.p3m500_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'am2k':
        image_dir = CONFIG.benchmark.am2k_img
        alpha_dir = CONFIG.benchmark.am2k_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)
            
    elif args.benchmark == 'aim':
        image_dir = CONFIG.benchmark.aim_img
        alpha_dir = CONFIG.benchmark.aim_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'pm10k':
        image_dir = CONFIG.benchmark.pm10k_img
        alpha_dir = CONFIG.benchmark.pm10k_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'rw100':
        image_dir = CONFIG.benchmark.rw100_img
        alpha_dir = CONFIG.benchmark.rw100_alpha

        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):

            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)