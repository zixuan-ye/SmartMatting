import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


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

def GenPoint(_target):
    radius = 20

    foreground_points = np.where(_target > 0)
    point_mask = np.zeros(_target.shape, dtype=_target.dtype)
    # 选择要标记的部分点，并将它们标记为1
    points = np.random.randint(1, 4)
    selected_points = np.random.choice(len(foreground_points[0]), size=min(len(foreground_points[0]),points), replace=False)
    for idx in selected_points:
        x, y = foreground_points[0][idx], foreground_points[1][idx]
        # 在半径范围内将点标记为1
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= x + i < _target.shape[0] and 0 <= y + j < _target.shape[1]:
                    point_mask[x + i, y + j] = 1

    return point_mask


def GenBox(_target):
    foreground_points = np.argwhere(_target == 1)

    # 获取左上、右上、左下和右下点的坐标
    if len(foreground_points) > 0:
        min_x, min_y = np.min(foreground_points, axis=0)
        max_x, max_y = np.max(foreground_points, axis=0)

      
        # 创建一个形状与foreground_mask相同的新掩码，初始值为0
        box_mask = np.zeros(_target.shape, dtype=np.uint8)

        # 在新掩码上将盒子内的点值设置为1
        box_mask[min_x:max_x,min_y:max_y] = 1

        return box_mask
    else:
        return np.zeros(_target.shape, dtype=np.uint8)
    
