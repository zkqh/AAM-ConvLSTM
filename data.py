import os
import pickle
import numbers
from random import Random
import numpy as np
import torch
import torch as th
from torch import nn
import torch.utils.data as data
from PIL import Image
from scipy import signal
from tqdm import tqdm
from torch.utils import data as tdata
#import cv2
#from functools import lru_cache
import torchvision.transforms as tf

class VRVideo(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frames_per_data=10,
                frame_interval=1, transform=None, train=True,
                gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, 
                kernel_size=(30, 60), cache_gt=True, rnd_seed=367643,rotate=True):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.rotate = rotate
        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        self.sequence_frame_map = []
        self.frames_per_data=frames_per_data
        count = 0
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for i in range(0, len(frame_list), frames_per_data):
                if i+frames_per_data > len(frame_list):
                    continue
                frames = frame_list[i:i+frames_per_data]
                frame_number_list = []
                for frame in frames:
                    frame_number_list.append(count)
                    count += 1
                    fid = frame[:4]
                    self.i2v[len(self.data)] = (vid, fid)
                    self.v2i[(vid, fid)] = len(self.data)
                    self.data.append(os.path.join(obj_path, frame))
                    self.target.append(self.vinfo[vid][fid])
                self.sequence_frame_map.append(frame_number_list)

        self.target.append([(0.5, 0.5)])
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def __getitem__(self, item):

        sequence_frames = self.sequence_frame_map[item]
        list_img = []
        list_target = []
        list_fix=[]
        for frame_item in sequence_frames[:self.frames_per_data]:
            if self.train:
                img,target,fix = self._getitem(frame_item)
            else:
                img,name,target,fix = self._getitem(frame_item)

            list_img.append(img)
            list_target.append(target)
            list_fix.append(fix)

        if self.train:
            img_tensor=torch.stack(list_img)
            target_tensor=torch.stack(list_target)
            fix_tensor=torch.stack(list_fix)

            if self.rotate==True:
                tf = Rotate()
                sample = [img_tensor, target_tensor,fix_tensor]
                img_tensor ,target_tensor,fix_tensor=tf(sample)
                return img_tensor ,target_tensor,fix_tensor

            return img_tensor,target_tensor,fix_tensor

        else:
            return torch.stack(list_img), torch.stack(list_target),torch.stack(list_fix)


    def _getitem(self, item_id):

        img = Image.open(open(self.data[item_id], 'rb'))
        if img.size == (self.frame_w, self.frame_h) and img.layers == 3:
            transform = tf.ToTensor()
            img = transform(img)
        else:
            if self.transform:
                img = self.transform(img)
            else:
                img = np.array(img)

        target,fix = self._get_salency_map(item_id)
        target=(target-target.min())/(target.max()-target.min())

        if self.train:
            return img,target,fix
        else:
            return img,self.data[item_id], target,fix

    def __len__(self):
        return len(self.sequence_frame_map)

    def _get_salency_map(self, item):
        tmp = self.data[item][:-4]
        cfile = tmp + '_gt.npy'
        cfile_hw = tmp + '_gt_{}_{}.npy'.format(self.frame_h, self.frame_w)
        fix=np.zeros((self.frame_h,self.frame_w))

        if item >= 0: # tong men bie juan le
            for x_norm,y_norm in self.target[item]:
                x,y=min(int(x_norm * self.frame_w + 0.5),self.frame_w-1),min(int(y_norm*self.frame_h+0.5),self.frame_h-1)
                fix[y,x]=10
            fix_tensor=torch.tensor(fix).unsqueeze(0)
            if self.cache_gt:
                if os.path.isfile(cfile_hw):
                    target_map = th.from_numpy(np.load(cfile_hw)).float()
                    assert target_map.size() == (1, self.frame_h, self.frame_w)
                    return target_map,fix_tensor  # th.from_numpy(np.load(cfile)).float()
                else:
                    target_map = th.from_numpy(np.load(cfile)).float()
                    target_map = nn.functional.interpolate(target_map.unsqueeze(0), size=(self.frame_h, self.frame_w),
                                                           mode='bilinear')
                    target_map = target_map.squeeze(0)
                    assert target_map.size() == (1, self.frame_h, self.frame_w)
                    # np.save(cfile_hw, target_map.data.cpu().numpy())
                    return target_map,fix_tensor  # th.from_numpy(np.load(cfile)).float()

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))
        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):
            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class Rotate(object):
    """
    Rotate the 360º image with respect to the vertical axis on the sphere.
    """

    def __call__(self, sample):
        input = sample[0]
        sal_map = sample[1]
        fix_map = sample[2]
        t = np.random.randint(input.shape[-1])

        new_sample = sample
        new_sample[0] = torch.cat((input[:, :, :, t:], input[:, :, :,0:t]), dim=3)
        new_sample[1] = torch.cat((sal_map[:, :, :, t:], sal_map[:, :, :, 0:t]), dim=3)
        new_sample[2] = torch.cat((fix_map[:, :, :,t:], fix_map[:, :, :,0:t]), dim=3)

        return new_sample

class VRVideoLSTM(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, sequence_len=2, frame_interval=5, transform=None,
                 train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi / 7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643,conv_type = 'sphereconv'):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.sequence_len = sequence_len
        self.conv_type=conv_type
        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('_240_320.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                # fcnt = 0

        self.target.append([(0.5, 0.5)])

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def __getitem__(self, item):
        img_list = []
        last_list = []
        target_list = []

        if item - self.sequence_len + 1 < 0:
            item = self.sequence_len - 1

        for idx in range(item - self.sequence_len + 1, item + 1, 1):
            # print(idx, item)
            img = Image.open(open(self.data[idx], 'rb'))
            # img = img.resize((self.frame_w, self.frame_h))
            if self.transform:
                img = self.transform(img)
            else:
                img = np.array(img)

            img_list.append(img)

            vid, fid = self.i2v[idx]
            if int(fid) - self.frame_interval <= 0:
                last = self._get_salency_map(-1)
            else:
                last = self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])

            target = self._get_salency_map(idx)  # 2019-07-08 Fix bug, item -> idx, xuyy

            if self.conv_type == 'sphereconv' or self.conv_type == 'standard' or self.conv_type == 'gafilters':
                last = (last - last.min()) / (last.max() - last.min())
                target = (target - target.min()) / (target.max() - target.min())

            last_list.append(last)
            target_list.append(target)

        img_var = torch.stack(img_list, 0)
        last_var = torch.stack(last_list, 0)
        target_var = torch.stack(target_list, 0)
        # exit()
        if self.train:
            return img_var, last_var, target_var
        else:
            return img_var, self.data[item], last_var, target_var, vid, fid

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):

        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()

                # upplayer = Interpolate(size=(), mode='bilinear')
                target_map = nn.functional.interpolate(target_map.unsqueeze(0), size=(self.frame_h, self.frame_w),
                                                       mode='bilinear')

                target_map = target_map.squeeze(0)

                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return target_map  # th.from_numpy(np.load(cfile)).float()
        target_map = th.zeros((1, self.frame_h, self.frame_w))
        # if item >= 0 and self.cache_gt:
        #     np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map  # .data / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):
            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self

# data_dir='/home/yf302/Desktop/teacher_wan/360_Saliency_dataset_2018ECCV/360_Saliency_dataset_2018ECCV'
# bs=1
# transform = tf.Compose([
#     tf.Resize((160, 320)),
#     tf.ToTensor()
# ])
# dataset = VRVideo(data_dir, 240, 320, 104, frame_interval=1, cache_gt=True, transform=transform, train = True, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7,frames_per_data=20)
# print(len(dataset))
# loader = tdata.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
# for i, (img_batch, target_batch,fix_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
#     img_var = img_batch.cuda()
#     t_var = (target_batch * 10).cuda()
#     pass