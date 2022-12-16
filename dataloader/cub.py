import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import random

import sys
sys.path.insert(0,'..')
from utils import *
import torchvision.transforms as transforms

from .tps_sampler import TPSRandomSampler
import torchvision

import cv2
import skimage
import skimage.transform
import scipy

def generate_heatmap(heatmap, pt, sigma):
    heatmap[int(pt[1])][int(pt[0])] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am / 255
    return heatmap

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def color_normalize(x, mean):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    normalized_mean = mean / 255
    for t, m in zip(x, normalized_mean):
        t.sub_(m)
    return x

    
class CUBRefined(Dataset):
    base_folder = '../data/CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    
    def __init__(self, root, train=True, is_train=True, evaluation=False, data_ratio=0.1,
                 transform=None, target_transform=None,
                 loader=default_loader, vertical_points=10, horizontal_points=10,
                 rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
                 warpsd=[0.001, 0.005, 0.001, 0.01],
                 use_ids=None, image_size=[128, 128], download=True, n_kpts=15,
                 return_grid=False, geometry=False, parity=False, subset=False):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.K = n_kpts
        self.sigma = 1.0
        # self.out_res = 56
        # self.inp_res = 224
        self.label_type='Gaussian'
        self.parity = parity
        self.subset = subset
        if self.subset:
            self.K = 10
            n_kpts = 10

        self.mean = [0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if use_ids is not None:
            self.data = self.data.iloc[use_ids]
            
        if evaluation:
            data_ratio = 1
        
        if data_ratio != 1:
            data_num = int(len(self.data) * data_ratio)
            random_idx = np.random.permutation(len(self.data))
            self.data = self.data.iloc[random_idx[:data_num]]
            
        self.num_class = n_kpts
        self.is_train = is_train
        self.evaluation = evaluation

        self.transform = transform
        self.target_transform = target_transform
            
        # Parameters for transformation
        self._image_size = image_size
        self.inp_res = image_size
        self.out_res = [64, 64]
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.pixel_means = [0.485, 0.456, 0.406]
        self.bbox_extend_factor = (0.4, 0.2)
        self.scale_factor=(0.7, 1.35)
        self.rot_factor=45
        
        self.gk15 = (15, 15)
        self.gk11 = (11, 11)
        self.gk9 = (9, 9)
        self.gk7 = (7, 7)
        
        self._target_sampler = TPSRandomSampler(
            image_size[1], image_size[0], rotsd=rotsd[0], scalesd=scalesd[0],
            transsd=transsd[0], warpsd=warpsd[:2], pad=False, return_grid=return_grid)
        self._source_sampler = TPSRandomSampler(
            image_size[1], image_size[0], rotsd=rotsd[1], scalesd=scalesd[1],
            transsd=transsd[1], warpsd=warpsd[2:], pad=False, return_grid=return_grid)
        self.return_grid = return_grid
        
        self.geometry = geometry
        
        
    def _load_metadata(self):
        
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        part_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'parts', 'part_locs.txt'),
                                  sep=' ', names=['img_id', 'part_id', 'x', 'y', 'visible'])
        bounding_box = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
                                   sep=' ', names=['img_id', 'x0', 'y0', 'width', 'height'])
        attr_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
                                  sep=' ', names=['img_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        # part data sorted by image
        self.parts = part_labels.merge(train_test_split, on='img_id')
        self.bbox = bounding_box.merge(train_test_split, on='img_id')
        
        # attribute labels
        self.attrs = attr_labels.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            self.parts = self.parts[self.parts.is_training_img == 1]
            self.bbox = self.bbox[self.bbox.is_training_img == 1]
            self.attrs = self.attrs[self.attrs.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.parts = self.parts[self.parts.is_training_img == 0]
            self.bbox = self.bbox[self.bbox.is_training_img == 0]
            self.attrs = self.attrs[self.attrs.is_training_img == 0]
            
        if self.parity:
            # exclude seabirds
            filter_class_ids = [1, 2, 3, 5, 6, 7, 8, 23, 24, 25, 59, 60, 61, 62, 63, 64, 65, 66,
                                100, 101]
            
            self.data = self.data[~self.data.target.isin(filter_class_ids)]
            filter_data_ids = self.data.img_id
            
            self.parts = self.parts[self.parts.img_id.isin(filter_data_ids)]
            self.bbox = self.bbox[self.bbox.img_id.isin(filter_data_ids)]
            self.attrs = self.attrs[self.attrs.img_id.isin(filter_data_ids)]
            
            # Parity check: left eye-7, right eye-11
            filter_parity = self.parts[(self.parts.part_id == 7) & (self.parts.visible == 1)]
            filter_parity2 = self.parts[(self.parts.part_id == 11) & (self.parts.visible == 0)]
            parity_ids = filter_parity.merge(filter_parity2, on='img_id', how='inner')
            filter_parity_ids = parity_ids.img_id
            
            self.data = self.data[self.data.img_id.isin(filter_parity_ids)]
            self.parts = self.parts[self.parts.img_id.isin(filter_parity_ids)]
            self.bbox = self.bbox[self.bbox.img_id.isin(filter_parity_ids)]
            self.attrs = self.attrs[self.attrs.img_id.isin(filter_parity_ids)]
            
            # Use 10 out of 15 keypoints
            # 5: crown, 11: right eye, 13: right wing, 6: forehead, 3: belly
            filter_part_ids = [1,2,4,7,8,9,10,12,14,15]
            self.parts = self.parts[self.parts.part_id.isin(filter_part_ids)]
            
            # Visibility check (filter images with the same visibility labels)
            visible = self.parts[self.parts.visible == 0]
            visible = visible.img_id
            self.data = self.data[~self.data.img_id.isin(visible)]
            self.parts = self.parts[~self.parts.img_id.isin(visible)]
            self.bbox = self.bbox[~self.bbox.img_id.isin(visible)]        

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)
        
     
    def augmentationCropImage(self, img, bbox, joints=None):  
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])  # width, height
        mean_value = self.pixel_means
        
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value) #.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])      
        bbox += add
        objcenter += add
        if joints is not None:
            joints[:, :2] += add
            inds = np.where(joints[:, -1] == 0)
            joints[inds, :2] = -1000000 # avoid influencing by data processing
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if joints is not None:
            crop_width = crop_width * (1 + 0.25)
            crop_height = crop_height * (1 + 0.25)  
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width  

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)                               
        
        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        if joints is not None:
            joints[:, 0] = joints[:, 0] - min_x
            joints[:, 1] = joints[:, 1] - min_y

            joints[:, 0] *= x_ratio
            joints[:, 1] *= y_ratio
            label = joints[:, :2].copy()
            valid = joints[:, 2].copy()
            
        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))  
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

        if joints is not None:
            return img, joints, details
        else:
            return img, details


    def data_augmentation(self, img, label, operation):
        height, width = img.shape[0], img.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])
        
        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img = skimage.transform.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                             int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        for i in range(n):
            label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            label[i][2] *= (
            (label[i][0] >= 0) & (label[i][0] < width) & (label[i][1] >= 0) & (label[i][1] < height))

        # flip augmentation
        if operation == 1:
            img = cv2.flip(img, 1)
            cod = []
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                if x >= 0:
                    x = width - 1 - x
                cod.append((x, y, label[i][2]))
            # **** the joint index depends on the dataset ****    
            # for (q, w) in self.symmetry:
            #     cod[q], cod[w] = cod[w], cod[q]
            for i in range(n):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
                allc.append(cod[i][2])
            label = np.array(allc).reshape(n, 3)

        # rotated augmentation
        if operation > 1:      
            angle = random.uniform(0, self.rot_factor)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotMat, (width, height))
            
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                v = label[i][2]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(int(coor[0]))
                allc.append(int(coor[1]))
                v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
                allc.append(int(v))
            label = np.array(allc).reshape(n, 3).astype(np.int)
        return img, label

  
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        img_path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        
        bbox = self.bbox.iloc[index]
        gt_bbox = [bbox.x0, bbox.y0, bbox.x0 + bbox.width, bbox.y0 + bbox.height]
        kpts = self.parts[self.parts.img_id==sample.img_id]
        kpts = np.asarray([kpts.x.values, kpts.y.values, kpts.visible.values])
        # kpts = torch.Tensor([kpts.x.values, kpts.y.values, kpts.visible.values])
        kpts = kpts.transpose(1,0)
        
        attrs = self.attrs[self.attrs.img_id==sample.img_id]
        attributes = torch.Tensor(attrs.is_present.values)
        
        points = np.array(kpts).reshape(self.num_class, 3).astype(np.float32)
        
        image = cv2.imread(img_path)
        # if self.is_train:
        #     image, points, details = self.augmentationCropImage(image, gt_bbox, points)
        # else:
        #     image, details = self.augmentationCropImage(image, gt_bbox)
        
        image, points, details = self.augmentationCropImage(image, gt_bbox, points)
        
        # operation = np.random.randint(0, 2) # for now, no flipping
        if self.parity:
            operation = 0
        else:
            operation = np.random.randint(0, 2)
        
        if self.is_train and (not self.evaluation):
            image, points = self.data_augmentation(image, points, operation)  
            
            image = np.float32(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img = im_to_torch(image)  # CxHxW
            
            # Color dithering
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            
            if self.geometry:
                appr_image = torch.clone(img)
                appr_image[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                appr_image[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                appr_image[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                
                # ADD TRANSLATION HERE???
                appr_image = torchvision.transforms.ToPILImage()(appr_image)
                appr_image = torchvision.transforms.Resize((128, 128))(appr_image)
                appr_image = torchvision.transforms.ToTensor()(appr_image)
                appr_image = self.normalize(appr_image)

            points[:, :2] //= 2 # output size is 1/4 input size
            pts = torch.Tensor(points)
        else:
            image = np.float32(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img = im_to_torch(image)
            #### ADDED FOR KNOWN GTS
            points[:, :2] //= 2 # output size is 1/4 input size
            pts = torch.Tensor(points)
            
        # img = self.normalize(img)
        
        height, width = self._image_size[:2]
        
        mask = self._get_smooth_mask(height, width, 10, 20)[None, :, :]
        # mask = self._get_smooth_mask(height, width, 5, 20)[None, :, :]
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        # Batch-wise implementation?
        if self.return_grid:
            _image, future_im, _mask, im_mask, _grid, future_grid = self._apply_tps(img, mask)
        else:    
            _image, future_im, _mask, im_mask = self._apply_tps(img, mask)
        
        img = self.normalize(img[0])
        
        future_im = torchvision.transforms.ToPILImage()(future_im[0])
        future_im = torchvision.transforms.Resize((128, 128))(future_im)
        future_im = torchvision.transforms.ToTensor()(future_im)
        future_im = self.normalize(future_im)
        # future_im = self.normalize(future_im[0])
                
        if self.return_grid:
            if self.geometry:
                return (img, future_im, _mask[0], mask[0], _grid, future_grid, appr_image), target
            return (img, future_im, _mask[0], mask[0], _grid, future_grid), target
        else:
            return (img, future_im, _mask[0], mask[0]), target  # attributes #
    
    
    def __len__(self):
        return len(self.data)
    
    def _get_smooth_step(self, n, b):
        x = torch.linspace(-1.0, 1, n)
        y = 0.5 + 0.5 * torch.tanh(x / b)
        return y


    def _get_smooth_mask(self, h, w, margin, step):
        b = 0.4
        step_up = self._get_smooth_step(step, b)
        step_down = self._get_smooth_step(step, -b)
        def create_strip(size):
            return torch.cat(
              (torch.zeros(margin, dtype=torch.float32),
               step_up,
               torch.ones(size - 2 * margin - 2 * step, dtype=torch.float32),
               step_down,
               torch.zeros(margin, dtype=torch.float32)), dim=0)
        mask_x = create_strip(w)
        mask_y = create_strip(h)
        mask2d = mask_y[:, None] * mask_x[None]
        return mask2d
    
    def _apply_tps(self, image, mask):

        def target_warp(images):
            return self._target_sampler.forward_py(images)
        def source_warp(images):
            return self._source_sampler.forward_py(images)

        image = torch.cat((mask, image), dim=1)
        shape = image.shape
        
        # future_image = target_warp(image)
        # image = source_warp(future_image)
        
        if self.return_grid:
            future_image, future_grid = target_warp(image)
            image, grid = source_warp(future_image)
        else:
            future_image = target_warp(image)
            image = source_warp(future_image)
        
        future_mask = future_image[:,0:1,...]
        future_image = future_image[:,1:,...]
        mask = image[:,0:1,...]
        image = image[:,1:,...]

        # inputs['image'] = image
        # inputs['future_image'] = future_image
        # inputs['mask'] = future_mask
        
        # return image, future_image, future_mask, mask
        if self.return_grid:
            return image, future_image, future_mask, mask, grid, future_grid
        else:
            return image, future_image, future_mask, mask
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    
class CUBLinear(Dataset):
    base_folder = '../data/CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    
    def __init__(self, root, train=True, is_train=True, evaluation=False, data_ratio=0.1,
                 transform=None, target_transform=None,
                 loader=default_loader, vertical_points=10, horizontal_points=10,
                 rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
                 warpsd=[0.001, 0.005, 0.001, 0.01],
                 use_ids=None, image_size=[128, 128], download=True, n_kpts=15,
                 return_grid=False, geometry=True, parity=True, subset=True):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.K = n_kpts
        self.sigma = 1.0
        # self.out_res = 56
        # self.inp_res = 224
        self.label_type='Gaussian'
        self.parity = parity
        self.subset = subset
        if self.subset:
            self.K = 10
            n_kpts = 10

        self.mean = [0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if use_ids is not None:
            self.data = self.data.iloc[use_ids]
            
        if evaluation:
            data_ratio = 1
        
        if data_ratio != 1:
            data_num = int(len(self.data) * data_ratio)
            random_idx = np.random.permutation(len(self.data))
            self.data = self.data.iloc[random_idx[:data_num]]
            
        self.num_class = n_kpts
        self.is_train = is_train
        self.evaluation = evaluation

        self.transform = transform
        self.target_transform = target_transform
            
        # Parameters for transformation
        self._image_size = image_size
        self.inp_res = image_size
        self.out_res = [64, 64]
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.pixel_means = [0.485, 0.456, 0.406]
        self.bbox_extend_factor = (0.4, 0.2)
        self.scale_factor=(0.7, 1.35)
        self.rot_factor=45
        
        self.gk15 = (15, 15)
        self.gk11 = (11, 11)
        self.gk9 = (9, 9)
        self.gk7 = (7, 7)
        
        self.return_grid = return_grid
        
        self.geometry = geometry
        
        
    def _load_metadata(self):
        
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        part_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'parts', 'part_locs.txt'),
                                  sep=' ', names=['img_id', 'part_id', 'x', 'y', 'visible'])
        bounding_box = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
                                   sep=' ', names=['img_id', 'x0', 'y0', 'width', 'height'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        # part data sorted by image
        self.parts = part_labels.merge(train_test_split, on='img_id')
        self.bbox = bounding_box.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            self.parts = self.parts[self.parts.is_training_img == 1]
            self.bbox = self.bbox[self.bbox.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.parts = self.parts[self.parts.is_training_img == 0]
            self.bbox = self.bbox[self.bbox.is_training_img == 0]
            
        if self.parity:
            # exclude seabirds
            filter_class_ids = [1, 2, 3, 5, 6, 7, 8, 23, 24, 25, 59, 60, 61, 62, 63, 64, 65, 66,
                                100, 101]
            
            self.data = self.data[~self.data.target.isin(filter_class_ids)]
            filter_data_ids = self.data.img_id
            
            self.parts = self.parts[self.parts.img_id.isin(filter_data_ids)]
            self.bbox = self.bbox[self.bbox.img_id.isin(filter_data_ids)]
            
            # Parity check: left eye-7, right eye-11
            filter_parity = self.parts[(self.parts.part_id == 7) & (self.parts.visible == 1)]
            filter_parity2 = self.parts[(self.parts.part_id == 11) & (self.parts.visible == 0)]
            parity_ids = filter_parity.merge(filter_parity2, on='img_id', how='inner')
            filter_parity_ids = parity_ids.img_id
            
            self.data = self.data[self.data.img_id.isin(filter_parity_ids)]
            self.parts = self.parts[self.parts.img_id.isin(filter_parity_ids)]
            self.bbox = self.bbox[self.bbox.img_id.isin(filter_parity_ids)]
            
            
        if self.subset:
            # Use 10 out of 15 keypoints
            # 5: crown, 11: right eye, 13: right wing, 6: forehead, 3: belly
            filter_part_ids = [1,2,4,7,8,9,10,12,14,15]
            self.parts = self.parts[self.parts.part_id.isin(filter_part_ids)]
            
            # Visibility check (filter images with the same visibility labels)
            visible = self.parts[self.parts.visible == 0]
            visible = visible.img_id
            self.data = self.data[~self.data.img_id.isin(visible)]
            self.parts = self.parts[~self.parts.img_id.isin(visible)]
            self.bbox = self.bbox[~self.bbox.img_id.isin(visible)]
            # vis_filter_ids = self.parts[(self.parts.part_id)(self.parts.visible == 1)
        
        

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)
        
     
    def augmentationCropImage(self, img, bbox, joints=None):  
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])  # width, height
        mean_value = self.pixel_means
        
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value) #.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])      
        bbox += add
        objcenter += add
        if joints is not None:
            joints[:, :2] += add
            inds = np.where(joints[:, -1] == 0)
            joints[inds, :2] = -1000000 # avoid influencing by data processing
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if joints is not None:
            crop_width = crop_width * (1 + 0.25)
            crop_height = crop_height * (1 + 0.25)  
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width  

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)                               
        
        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        if joints is not None:
            joints[:, 0] = joints[:, 0] - min_x
            joints[:, 1] = joints[:, 1] - min_y

            joints[:, 0] *= x_ratio
            joints[:, 1] *= y_ratio
            label = joints[:, :2].copy()
            valid = joints[:, 2].copy()
        
        # if bimg.shape[0] < min_x:
        #     import pdb; pdb.set_trace()
            
        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))  
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

        if joints is not None:
            return img, joints, details
        else:
            return img, details



    def data_augmentation(self, img, label, operation):
        height, width = img.shape[0], img.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])
        
        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img = skimage.transform.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                             int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        for i in range(n):
            label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            label[i][2] *= (
            (label[i][0] >= 0) & (label[i][0] < width) & (label[i][1] >= 0) & (label[i][1] < height))

        # flip augmentation
        if operation == 1:
            img = cv2.flip(img, 1)
            cod = []
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                if x >= 0:
                    x = width - 1 - x
                cod.append((x, y, label[i][2]))
            # **** the joint index depends on the dataset ****    
            # for (q, w) in self.symmetry:
            #     cod[q], cod[w] = cod[w], cod[q]
            for i in range(n):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
                allc.append(cod[i][2])
            label = np.array(allc).reshape(n, 3)

        # rotated augmentation
        if operation > 1:      
            angle = random.uniform(0, self.rot_factor)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotMat, (width, height))
            
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                v = label[i][2]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(int(coor[0]))
                allc.append(int(coor[1]))
                v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
                allc.append(int(v))
            label = np.array(allc).reshape(n, 3).astype(np.int)
        return img, label

  
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        img_path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        
        bbox = self.bbox.iloc[index]
        gt_bbox = [bbox.x0, bbox.y0, bbox.x0 + bbox.width, bbox.y0 + bbox.height]
        kpts = self.parts[self.parts.img_id==sample.img_id]
        kpts = np.asarray([kpts.x.values, kpts.y.values, kpts.visible.values])
        # kpts = torch.Tensor([kpts.x.values, kpts.y.values, kpts.visible.values])
        kpts = kpts.transpose(1,0)
        
        points = np.array(kpts).reshape(self.num_class, 3).astype(np.float32)
        
        image = cv2.imread(img_path)
        # if self.is_train:
        #     image, points, details = self.augmentationCropImage(image, gt_bbox, points)
        # else:
        #     image, details = self.augmentationCropImage(image, gt_bbox)
        
        image, points, details = self.augmentationCropImage(image, gt_bbox, points)
        
        # operation = np.random.randint(0, 2) # for now, no flipping
        if self.parity:
            operation = 0
        else:
            operation = np.random.randint(0, 2)
        
        if self.is_train and (not self.evaluation):
            image, points = self.data_augmentation(image, points, operation)  
            
            image = np.float32(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img = im_to_torch(image)  # CxHxW
            
            # Color dithering
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            
            if self.geometry:
                appr_image = torch.clone(img)
                appr_image[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                appr_image[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                appr_image[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                
                # ADD TRANSLATION HERE???
                appr_image = torchvision.transforms.ToPILImage()(appr_image)
                appr_image = torchvision.transforms.Resize((128, 128))(appr_image)
                appr_image = torchvision.transforms.ToTensor()(appr_image)
                appr_image = self.normalize(appr_image)

            points[:, :2] //= 2 # output size is 1/4 input size
            pts = torch.Tensor(points)
        else:
            image = np.float32(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img = im_to_torch(image)
            #### ADDED FOR KNOWN GTS
            points[:, :2] //= 2 # output size is 1/4 input size
            pts = torch.Tensor(points)
        
        pts[:,0] = (pts[:,0] / float(self.out_res[0])) * 2 - 1
        pts[:,1] = (pts[:,1] / float(self.out_res[1])) * 2 - 1
        
        img = self.normalize(img)
        
        return (img, pts, img_path, gt_bbox, details), target
        
    
    def __len__(self):
        return len(self.data)
    
    def _get_smooth_step(self, n, b):
        x = torch.linspace(-1.0, 1, n)
        y = 0.5 + 0.5 * torch.tanh(x / b)
        return y


    def _get_smooth_mask(self, h, w, margin, step):
        b = 0.4
        step_up = self._get_smooth_step(step, b)
        step_down = self._get_smooth_step(step, -b)
        def create_strip(size):
            return torch.cat(
              (torch.zeros(margin, dtype=torch.float32),
               step_up,
               torch.ones(size - 2 * margin - 2 * step, dtype=torch.float32),
               step_down,
               torch.zeros(margin, dtype=torch.float32)), dim=0)
        mask_x = create_strip(w)
        mask_y = create_strip(h)
        mask2d = mask_y[:, None] * mask_x[None]
        return mask2d

        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
