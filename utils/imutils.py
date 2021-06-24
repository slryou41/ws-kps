# Code from https://github.com/bearpaw/pytorch-pose/blob/master/pose/utils/imutils.py

from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import cv2
import math

from .misc import *

def pil_to_numpy(img):
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))
    return img


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    print('%f %f' % (img.min(), img.max()))
    img = scipy.misc.imresize(
            img,
            (oheight, owidth)
        )
    img = im_to_torch(img)
    print('%f %f' % (img.min(), img.max()))
    return img

# =============================================================================
# Keypoint-based augmentation
# =============================================================================
def augment_keypoints(img, pt, inp_res=[128, 128]):
    # Zoom or mask?
    # Masking: input image has similar scale in CUB dataset
    # Zooming: images have various scales in AnimalPose/AWA dataset
    # It's more like a refinement step using pseudo labels...
    
    
    return img

# =============================================================================
# Helpful functions generating groundtruth labelmap
# =============================================================================
def generate_heatmap(heatmap, pt, sigma):
    heatmap[int(pt[1])][int(pt[0])] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am / 255
    return heatmap

def generate_pseudo_heatmaps(pts, out_res=[64, 64]):
    
    pts = torch.stack((pts[0], pts[1]), dim=2)
    
    num_batch, num_class = pts.size(0), pts.size(1)
    
    target15 = np.zeros((num_batch, num_class, out_res[0], out_res[1]))
    target11 = np.zeros((num_batch, num_class, out_res[0], out_res[1]))
    target9 = np.zeros((num_batch, num_class, out_res[0], out_res[1]))
    target7 = np.zeros((num_batch, num_class, out_res[0], out_res[1]))
    
    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)
    pts = (pts + 1) * out_res[0] / 2.0
    
    for b_i in range(num_batch):
        for i in range(num_class):
            if math.isnan(pts[b_i,i,0]) or math.isnan(pts[b_i,i,1]):
                import pdb; pdb.set_trace()
            
            pts[b_i, i, 0] = pts[b_i, i, 0].clamp(max=out_res[0]-1)
            pts[b_i, i, 1] = pts[b_i, i, 1].clamp(max=out_res[1]-1)
            pts[b_i, i, 0] = pts[b_i, i, 0].clamp(min=0)
            pts[b_i, i, 1] = pts[b_i, i, 1].clamp(min=0)

            target15[b_i, i] = generate_heatmap(target15[b_i, i], pts[b_i, i], gk15)
            target11[b_i, i] = generate_heatmap(target11[b_i, i], pts[b_i, i], gk11)
            target9[b_i, i] = generate_heatmap(target9[b_i, i], pts[b_i, i], gk9)
            target7[b_i, i] = generate_heatmap(target7[b_i, i], pts[b_i, i], gk7)
            
    targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
    
    return targets


def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return to_torch(h).float()

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img), 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img), 1

# =============================================================================
# Helpful display functions
# =============================================================================

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def imshow(img):
    npimg = im_to_numpy(img*255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis('off')

def show_joints(img, pts):
    imshow(img)

    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
    plt.axis('off')

def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp*0.5 + color_heatmap(target[n,p,:,:])*0.5
            out = torch.cat((out, tgt), 2)

        imshow(out)
        plt.show()
        
        
def visualize_with_heatmap(inp, out, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2], 
                           num_rows=1, parts_to_show=None):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    inp = (inp * _std + _mean) * 255
    out = to_numpy(out)
    
    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]
        
    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])
    
    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    # inp_small = scipy.misc.imresize(img, [size, size])
    inp_small = cv2.resize(img, (size, size))

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        # out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = cv2.resize(out[part_idx], (size, size))
        out_resized = out_resized.astype(float)/255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img
    

def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float)/255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img

def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0.5, 0.5, 0.5]).cuda(), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return np.concatenate(batch_img)