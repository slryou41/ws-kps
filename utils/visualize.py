import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .misc import *   

from PIL import Image, ImageDraw
import cv2
import math

from mpl_toolkits.mplot3d import Axes3D

__all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single', 'show_img_with_pts', 
           'visualize_transform', 'show_img_with_moving_pts', 'show_img_with_two_pts', 'show_heatmaps',
           'show_img_with_coords', 'show_img_with_heatmap', 'show_img_with_two_moving_pts',
           'show_img_with_confidence']

# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()
    
def show_heatmaps(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    images = images.unsqueeze(1)
    images = torchvision.utils.make_grid(images, nrow=5, normalize=True, padding=3, pad_value=1.0)
    images = images.unsqueeze(0)
    images = upsampling(images)
    
    return images[0]


def show_mask_single(images, mask, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(2, 1, 1)
    plt.imshow(images)
    plt.axis('off')

    # for b in range(mask.size(0)):
    #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
    mask_size = mask.size(2)
    # print('Max %f Min %f' % (mask.max(), mask.min()))
    mask = (upsampling(mask, scale_factor=im_size/mask_size))
    # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
    # for c in range(3):
    #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

    # print(mask.size())
    mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
    # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.axis('off')

def show_mask(images, masklist, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(1+len(masklist), 1, 1)
    plt.imshow(images)
    plt.axis('off')

    for i in range(len(masklist)):
        mask = masklist[i].data.cpu()
        # for b in range(mask.size(0)):
        #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
        mask_size = mask.size(2)
        # print('Max %f Min %f' % (mask.max(), mask.min()))
        mask = (upsampling(mask, scale_factor=im_size/mask_size))
        # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
        # for c in range(3):
        #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

        # print(mask.size())
        mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
        # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
        plt.subplot(1+len(masklist), 1, i+2)
        plt.imshow(mask)
        plt.axis('off')
        
        
def show_img_with_heatmap(image, heatmap, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    image = image.transpose((1,2,0))
    image = image.astype('uint8')
    
    _heatmap = (heatmap / heatmap.max()) * 255
    # heatmap = (heatmap * 255)
    _heatmap = 255 - _heatmap
    _heatmap = cv2.resize(_heatmap, (image.shape[0], image.shape[1]))
    _heatmap = np.uint8(_heatmap)
    
    heatmap_img = cv2.applyColorMap(_heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.6, image, 0.4, 0)
    
    # show score
    x, y, (var_x, var_y, cov) = _mapTokpt(heatmap)
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (20, 20)

    # fontScale
    fontScale = 0.5

    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 1
    
    fin = cv2.putText(fin, '%.2f / %.2f'%(heatmap.max()*100, (var_x+var_y)*100), org, font, 
            fontScale, color, thickness, cv2.LINE_AA)
    
    return fin

def _mapTokpt(heatmap):
    # heatmap: (H, W)    

    H = heatmap.shape[0]
    W = heatmap.shape[1]

    s_y = heatmap.sum(1)  # (H)
    s_x = heatmap.sum(0)  # (W)

    y = np.linspace(-1.0, 1.0, H)
    x = np.linspace(-1.0, 1.0, W)
    
    u_y = (y * s_y).sum() / s_y.sum()  # 1
    u_x = (x * s_x).sum() / s_x.sum()

    y = np.reshape(y, (H, 1))
    x = np.reshape(x, (1, W))

    # Covariance
    var_y = ((heatmap * y**2).sum(0).sum() - u_y**2) #.clamp(min=1e-6)
    var_x = ((heatmap * x**2).sum(0).sum() - u_x**2) #.clamp(min=1e-6)

    cov = ((heatmap * (x - u_x) * (y - u_y)).sum()) #.clamp(min=1e-6)

    return u_x, u_y, (var_x, var_y, cov)


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v


# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)

    
def fit_ellipsoid_from_heatmap(heatmap):
    # Find maximum index
    H = heatmap.shape[0]
    W = heatmap.shape[1]
    
    max_idx = np.argmax(heatmap)
    max_x = max_idx % H;  max_y = max_idx // W
    max_sc = heatmap[max_y, max_x]
    
    # # extract 9 points from peak
    if max_x-1 < 0:
        points_x = [max_x] * 3 + [max_x+1] * 3 + [max_x+2] * 3
    elif max_x+1 >= W:
        points_x = [max_x-2] * 3 + [max_x-1] * 3 + [max_x] * 3
    else:
        points_x = [max_x-1]*3 + [max_x]*3 + [max_x+1]*3
        
    if max_y-1 < 0:
        points_y = [max_y] * 3 + [max_y+1] * 3 + [max_y+2] * 3
    elif max_y+1 >= H:
        points_y = [max_y-2] * 3 + [max_y-1] * 3 + [max_y] * 3
    else:
        points_y = [max_y-1, max_y, max_y+1]*3
    # extract 25 points from peak
    # points_x = [max_x-2]*5 + [max_x-1]*5 + [max_x]*5 + [max_x+1]*5 + [max_x+2]*5
    # points_y = [max_y-2, max_y-1, max_y, max_y+1, max_y+2]*5
    value_z = heatmap[points_y, points_x] * 100
    
    center_x = [-1]*3 + [0]*3 + [1]*3
    center_y = [-1,0,1] * 3
    
    # ellipsoid equation
    X = [points_x, points_y, value_z]
    # X = [center_x, center_y, value_z]
    X = np.asarray(X)
    X = X.transpose((1,0))
    X_b = X.copy()
    X_b[:,2] = -X_b[:,2]
    X = np.concatenate((X, X_b), axis=0)
    
    # ellipse fitting by regression
    center, evecs, radii, v = ellipsoid_fit(X)
    
    data_centered = X - center.T
    
    a, b, c = radii
    r = (a * b * c) ** (1. / 3.)
    D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    #http://www.cs.brandeis.edu/~cs155/Lecture_07_6.pdf
    #affine transformation from ellipsoid to sphere (translation excluded)
    TR = evecs.dot(D).dot(evecs.T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
            
    ax.scatter(data_centered[:,0], data_centered[:,1], data_centered[:,2], marker='o', color='g')

    ellipsoid_plot([0, 0, 0], radii, evecs, ax=ax, plot_axes=True, cage_color='g')
    ellipsoid_plot([0, 0, 0], [r, r, r], evecs, ax=ax, plot_axes=True, cage_color='orange')

    # plt.show()
    plt.savefig('ellipse.png')
    plt.close()
    
    # Compute volume
    # Confidence of detection
    volume = abs(a * b * c)
    
    # Compute surface area
    # Localization uncertainty
    area = abs(a * b)
    # import pdb; pdb.set_trace()
    
    # Ellipse drawing parameters
    axesLength = (abs(a), abs(b))
    rot_matrix = evecs[:2,:2]
    angle = np.arccos(rot_matrix[0][0])
    angle = 180 / math.pi * angle
        
    return volume, area, axesLength, angle, center
    
    

def show_img_with_confidence(image, heatmap, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    image = image.transpose((1,2,0))
    image = image.astype('uint8')
    
    _heatmap = (heatmap / heatmap.max()) * 255
    # heatmap = (heatmap * 255)
    _heatmap = 255 - _heatmap
    _heatmap = cv2.resize(_heatmap, (image.shape[0], image.shape[1]))
    _heatmap = np.uint8(_heatmap)
    
    heatmap_img = cv2.applyColorMap(_heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.6, image, 0.4, 0)
    
    # show score
    # Compute variance
    x, y, (var_x, var_y, cov) = _mapTokpt(heatmap)     
    
    # Ellipse fitting
    confidence, uncertainty, axesLength, angle, center = fit_ellipsoid_from_heatmap(heatmap)
    
    # Visualization with ellipse
    startAngle = 0
    endAngle = 360
    center_coordinates = (int(round(center[0]*2)), int(round(center[1]*2)))           
    axesLength = (max(1, int(round(axesLength[0]))*2), max(1, int(round(axesLength[1]))*2))
    angle = int(angle)
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (10, 20)
    bottom_org = (10, 100)

    # fontScale
    fontScale = 0.5

    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 1
    
    fin = cv2.putText(fin, '%.2f / %.2f'%(confidence, uncertainty), bottom_org, font, 
            fontScale, color, thickness, cv2.LINE_AA)
    # import pdb; pdb.set_trace()
    fin = cv2.ellipse(fin, center_coordinates, axesLength, angle, 0, 360, (255,204,204), 2)
    
    return fin
    
    
def show_img_with_coords(image, pts, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2], vis=None):
    # _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    # image = (image * _std + _mean) * 255
    
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],  # yellow, cyan, pink
              [255,255,255], [0,0,0], [125,125,0], [125,0,125], [0,125,125], [74,29,0],  #w b, gray-green, violet, turquiose
              [23,87,120], [59,100,3], [130,150,10], [40,29,100], [54,12,90], [53,123,40],
              [50,189,34], [230,142,10], [12,203,5], [200,12,50], [12,125,203], [78,80,200],
              [80,24,145], [145,65,16], [67,124,56], [20,56,125], [243,25,65], [53,135,76]]
    
    # scale = (image.shape[1] / 2.0)
    # scale = (256.0 / 16.0)
    for i in range(0,pts.shape[0]):
        if (vis is not None) and (vis[i] == 0):
            continue
        st_y = max(0, int(pts[i,1])-4); st_x = max(0, int(pts[i,0])-4)
        end_y = min(image.shape[1], int(pts[i,1])+4); end_x = min(image.shape[2], int(pts[i,0])+4)
        # if st_y - end_y > 50:
        #     import pdb; pdb.set_trace()
        image[0,st_y:end_y,st_x:end_x] = colors[i][0]
        image[1,st_y:end_y,st_x:end_x] = colors[i][1]
        image[2,st_y:end_y,st_x:end_x] = colors[i][2]
            
    return image

def show_img_with_pts(image, pts, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2], vis=None):
    # import pdb; pdb.set_trace()
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],  # yellow, cyan, pink
              [255,255,255], [0,0,0], [125,125,0], [125,0,125], [0,125,125], [74,29,0],  #w b, gray-green, violet, turquiose
              [23,87,120], [59,100,3], [130,150,10], [40,29,100], [54,12,90], [53,123,40],
              [50,189,34], [230,142,10], [12,203,5], [200,12,50], [12,125,203], [78,80,200],
              [80,24,145], [145,65,16], [67,124,56], [20,56,125], [243,25,65], [53,135,76]]
    
    scale = (image.shape[1] / 2.0)
    # scale = (256.0 / 16.0)
    for i in range(0,pts.shape[0]):
        if (vis is not None) and (vis[i] == 0):
            continue
#         st_y = max(0, int(pts[i,1]*scale)-2); st_x = max(0, int(pts[i,0]*scale)-2)
#         end_y = min(image.shape[1], int(pts[i,1]*scale)+2); end_x = min(image.shape[2], int(pts[i,0]*scale)+2)
#         # if st_y - end_y > 50:
#         #     import pdb; pdb.set_trace()
#         image[0,st_y:end_y,st_x:end_x] = colors[i][0]
#         image[1,st_y:end_y,st_x:end_x] = colors[i][1]
#         image[2,st_y:end_y,st_x:end_x] = colors[i][2]
        
        radius = 4
        st_y = max(0, int(pts[i,1]*scale)-radius); st_x = max(0, int(pts[i,0]*scale)-radius)
        end_y = min(image.shape[1], int(pts[i,1]*scale)+radius+1); end_x = min(image.shape[2], int(pts[i,0]*scale)+radius+1)
        # if st_y - end_y > 50:
        #     import pdb; pdb.set_trace()
#         image[0,st_y:end_y,st_x:end_x] = colors[i][0]
#         image[1,st_y:end_y,st_x:end_x] = colors[i][1]
#         image[2,st_y:end_y,st_x:end_x] = colors[i][2]
            
        margin = 1
    
        # image[0,st_y+margin:end_y-margin,st_x+margin:end_x-margin] = colors[i][0]
        # image[1,st_y+margin:end_y-margin,st_x+margin:end_x-margin] = colors[i][1]
        # image[2,st_y+margin:end_y-margin,st_x+margin:end_x-margin] = colors[i][2]
        
        center_coordinates = (int(pts[i,0]*scale), int(pts[i,1]*scale))
        image[0,center_coordinates[1]-margin:center_coordinates[1]+margin,st_x:end_x] = colors[i][0]
        image[1,center_coordinates[1]-margin:center_coordinates[1]+margin,st_x:end_x] = colors[i][1]
        image[2,center_coordinates[1]-margin:center_coordinates[1]+margin,st_x:end_x] = colors[i][2]
        
        image[0,st_y:end_y,center_coordinates[0]-margin:center_coordinates[0]+margin] = colors[i][0]
        image[1,st_y:end_y,center_coordinates[0]-margin:center_coordinates[0]+margin] = colors[i][1]
        image[2,st_y:end_y,center_coordinates[0]-margin:center_coordinates[0]+margin] = colors[i][2]
    
    return image


def show_img_with_moving_pts(image, pts1, pts2, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    
    new_img = image.astype('uint8').transpose((1,2,0))
    new_img = Image.fromarray(new_img).convert('RGB')
    
    draw = ImageDraw.Draw(new_img)
    
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],
              [255,255,255], [0,0,0], [125,125,0], [125,0,125], [0,125,125], [74,29,0],
              [23,87,120], [59,100,3], [130,150,10], [40,29,100], [54,12,90], [53,123,40]]
    
    # scale = (256.0 / 64.0)
    scale = (image.shape[1] / 2.0)
    for i, pt in enumerate(pts1):
        x = pt[0] * scale;  y = pt[1] * scale
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill = tuple(colors[i]))
        x2 = pts2[i][0] * scale;  y2 = pts2[i][1] * scale
        draw.ellipse([x2 - 2, y2 - 2, x2 + 2, y2 + 2], fill = tuple(colors[i]))
        draw.line((x, y, x2, y2), fill = tuple(colors[i]))
    
    return np.array(new_img).transpose((2,0,1))

def show_img_with_two_moving_pts(image, pts1, pts2, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    
    new_img = image.astype('uint8').transpose((1,2,0))
    new_img = Image.fromarray(new_img).convert('RGB')
    
    draw = ImageDraw.Draw(new_img)
    
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],
              [255,255,255], [0,0,0], [125,125,0], [125,0,125], [0,125,125], [74,29,0],
              [23,87,120], [59,100,3], [130,150,10], [40,29,100], [54,12,90], [53,123,40]]
    
    # scale = (256.0 / 64.0)
    scale = (image.shape[1] / 2.0)
    margin = 1
    for i, pt in enumerate(pts1):
        x = pt[0] * scale;  y = pt[1] * scale
        draw.rectangle([x - margin, y - margin, x + margin, y + margin], fill = tuple(colors[i]))
        x2 = pts2[2*i][0] * scale;  y2 = pts2[2*i][1] * scale
        draw.ellipse([x2 - margin, y2 - margin, x2 + margin, y2 + margin], fill = tuple(colors[i]))
        draw.line((x, y, x2, y2), fill = tuple(colors[i]))
        x3 = pts2[2*i+1][0] * scale; y3 = pts2[2*i+1][1]*scale
        draw.ellipse([x3 - margin, y3 - margin, x3 + margin, y3 + margin], fill = tuple(colors[i]))
        draw.line((x, y, x3, y3), fill = tuple(colors[i]))
    
    return np.array(new_img).transpose((2,0,1))



def show_img_with_two_pts(image, pts1, pts2, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    
    new_img = image.astype('uint8').transpose((1,2,0))
    new_img = Image.fromarray(new_img).convert('RGB')
    
    draw = ImageDraw.Draw(new_img)
    
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],  # yellow, cyan, pink
              [255,255,255], [0,0,0], [125,125,0], [125,0,125], [0,125,125], [74,29,0],  #w b, gray-green, violet, turquiose
              [23,87,120], [59,100,3], [130,150,10], [40,29,100], [54,12,90], [53,123,40],
              [50,189,34], [230,142,10], [12,203,5], [200,12,50], [12,125,203], [78,80,200],
              [80,24,145], [145,65,16], [67,124,56], [20,56,125], [243,25,65], [53,135,76]]
    
    # scale = (256.0 / 64.0)
    scale = (image.shape[1] / 2.0)
    for i in range(0,pts1.shape[0]):
        st_y = max(0, int(pts1[i,1]*scale)-2); st_x = max(0, int(pts1[i,0]*scale)-2)
        end_y = min(image.shape[1], int(pts1[i,1]*scale)+2); end_x = min(image.shape[2], int(pts1[i,0]*scale)+2)
        if st_y - end_y > 50:
            import pdb; pdb.set_trace()
        image[0,st_y:end_y,st_x:end_x] = colors[i][0]
        image[1,st_y:end_y,st_x:end_x] = colors[i][1]
        image[2,st_y:end_y,st_x:end_x] = colors[i][2]
        
        st_y = max(0, int(pts2[i,1]*scale)-1); st_x = max(0, int(pts2[i,0]*scale)-1)
        end_y = min(image.shape[1], int(pts2[i,1]*scale)+1); end_x = min(image.shape[2], int(pts2[i,0]*scale)+1)
        if st_y - end_y > 50:
            import pdb; pdb.set_trace()
        image[0,st_y:end_y,st_x:end_x] = colors[i][0]
        image[1,st_y:end_y,st_x:end_x] = colors[i][1]
        image[2,st_y:end_y,st_x:end_x] = colors[i][2]
        
    return image


def visualize_transform(source, target, source_control_points, grid_size, mean=[0.5, 0.5, 0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    
    source_array = source.data.cpu().numpy()
    target_array = target.data.cpu().numpy()
    
    source_array = (source_array * _std + _mean) * 255
    target_array = (target_array * _std + _mean) * 255
    
    source_array = source_array.astype('uint8').transpose((1,2,0))
    target_array = target_array.astype('uint8').transpose((1,2,0))
    
    # resize for better visualization
    source_image = Image.fromarray(source_array).convert('RGB').resize((128, 128))
    target_image = Image.fromarray(target_array).convert('RGB').resize((128, 128))
    # create grey canvas for external control points
    canvas = Image.new(mode = 'RGB', size = (64 * 7, 64 * 4), color = (128, 128, 128))
    canvas.paste(source_image, (64, 64))
    canvas.paste(target_image, (64 * 4, 64))
    source_points = source_control_points.data
    source_points = (source_points + 1) / 2 * 128 + 64
    draw = ImageDraw.Draw(canvas)
    
    for x, y in source_points:
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill = (255, 0, 0))
        
    source_points = source_points.view(grid_size, grid_size, 2)
    for j in range(grid_size):
        for k in range(grid_size):
            x1, y1 = source_points[j, k]
            if j > 0: # connect to left
                x2, y2 = source_points[j - 1, k]
                draw.line((x1, y1, x2, y2), fill = (255, 0, 0))
            if k > 0: # connect to up
                x2, y2 = source_points[j, k - 1]
                draw.line((x1, y1, x2, y2), fill = (255, 0, 0))
                
    # draw.text((10, 0), 'iter %03d' % (len(paths) - 1 - pi), fill = (255, 0, 0), font = font)
    return np.array(canvas).transpose((2,0,1))

# x = torch.zeros(1, 3, 3)
# out = colorize(x)
# out_im = make_image(out)
# plt.imshow(out_im)
# plt.show()