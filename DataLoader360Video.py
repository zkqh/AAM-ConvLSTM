import torch
import os
import cv2
import math
from torch.utils.data import Dataset
import numpy as np
import time
from tqdm import tqdm
import config
from torch.utils.data import DataLoader

class RGB(Dataset):
    def __init__(self, path_to_frames,path_to_saliency_maps, path_to_fixation_maps,video_names, frames_per_data=20, split_percentage=0.2, split='train', resolution = [240, 320], skip=0, load_names=False, transform=False):
        self.sequences = []
        self.correspondent_sal_maps = []
        self.frames_per_data = frames_per_data
        self.path_frames = path_to_frames
        self.path_sal_maps = path_to_saliency_maps
        self.resolution = resolution
        self.load_names = load_names
        self.transform = transform
        self.path_to_fixation_maps=path_to_fixation_maps

        # Different videos for each split
        # sp = int(math.ceil(split_percentage * len(video_names)))
        # if split == "validation":
        #     video_names = video_names[:sp]
        # elif split == "train":
        #     video_names = video_names[sp:]

        # i_start = 0
        
        for name in video_names:
            video_frames_names = os.listdir(os.path.join(self.path_frames, name))
            video_frames_names = sorted(video_frames_names, key=lambda x: int((x.split(".")[0]).split("_")[1]))
            # Skip the first frames to avoid biases due to the eye-tracking capture procedure
            # (Observers are usually asked to look at a certain point at the beginning of each video )
            sts = 0

            # Split the videos in sequences of equal length
            for end in range(skip + frames_per_data, len(video_frames_names)-5, frames_per_data):

                # Check if exist the ground truth saliency map for all the frames in the sequence
                valid_sequence = True

                if not self.path_frames is None:

                    for frame in video_frames_names[sts:end]:

                        if not os.path.exists(os.path.join(self.path_frames, name, frame)):
                        # if not os.path.exists(
                        #         os.path.join(self.path_sal_maps, name, frame)):
                            valid_sequence = False
                            break

                if valid_sequence: self.sequences.append(video_frames_names[sts:end])
                sts = end

                    

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        frame_img = []
        label = []
        frame_names = []
        fixation=[]

        # Read the RGB images and saliency maps for each frame in the sequence
        for frame_name in self.sequences[idx]:

            # Obtain the name of the frame
            fn = os.path.splitext(os.path.basename(frame_name))[0]
            frame_names.append(fn)

            frame_path = os.path.join(self.path_frames, frame_name.split("_")[0], frame_name)
            assert os.path.exists(frame_path), 'Image frame has not been found in path: ' + frame_path
            img_frame = cv2.imread(frame_path)

            if img_frame.shape[1] != self.resolution[1] or img_frame.shape[0] != self.resolution[0]:
                img_frame = cv2.resize(img_frame, (self.resolution[1], self.resolution[0]),
                                        interpolation=cv2.INTER_AREA)
            img_frame = img_frame.astype(np.float32)
            img_frame = img_frame / 255.0

            img_frame = torch.FloatTensor(img_frame)
            img_frame = img_frame.permute(2, 0, 1)

            frame_img.append(img_frame.unsqueeze(0))  # Adding dimension: (n_frames, ch, h, w)

            if not self.path_to_fixation_maps == None:
                #fixation_map_path = os.path.join(self.path_to_fixation_maps, frame_name.split("_")[0], frame_name)
                fixation_name = frame_name.split("_")[1].split(".")[0] + '.png'
                fixation_map_path = os.path.join(self.path_to_fixation_maps, frame_name.split("_")[0], fixation_name)
                assert os.path.exists(fixation_map_path), 'Saliency map has not been found in path: ' + fixation_map_path

                fixation_map = cv2.imread(fixation_map_path, cv2.IMREAD_GRAYSCALE)

                if fixation_map.shape[1] != 320 or fixation_map.shape[0] != 240:
                    fixation_map = cv2.resize(fixation_map, (320, 240),
                                                interpolation=cv2.INTER_AREA)

                fixation_map = fixation_map.astype(np.float32)
                fixation_map = (fixation_map - np.min(fixation_map)) / (np.max(fixation_map) - np.min(fixation_map))
                fixation_map = torch.FloatTensor(fixation_map).unsqueeze(0)
                fixation.append(fixation_map.unsqueeze(0))

            if not self.path_sal_maps == None:
                sal_frame_name=frame_name.split("_")[1].split(".")[0]+'.png'
                sal_map_path = os.path.join(self.path_sal_maps, frame_name.split("_")[0],sal_frame_name )
                assert os.path.exists(sal_map_path), 'Saliency map has not been found in path: ' + sal_map_path

                saliency_img = cv2.imread(sal_map_path, cv2.IMREAD_GRAYSCALE)

                if saliency_img.shape[1] != self.resolution[1] or saliency_img.shape[0] != self.resolution[0]:
                    saliency_img = cv2.resize(saliency_img, (self.resolution[1], self.resolution[0]),
                                                interpolation=cv2.INTER_AREA)

                saliency_img = saliency_img.astype(np.float32)
                # saliency_img = (saliency_img - np.min(saliency_img)) / (np.max(saliency_img) - np.min(saliency_img))
                saliency_img=saliency_img/255
                saliency_img = torch.FloatTensor(saliency_img).unsqueeze(0)
                label.append(saliency_img.unsqueeze(0))



        if self.load_names:

            if self.path_sal_maps is None or self.path_to_fixation_maps is None: sample = [torch.cat(frame_img, 0), frame_names]
            else: sample = [torch.cat(frame_img, 0), torch.cat(label, 0), torch.cat(fixation,0),frame_names]

        else:

            if self.path_sal_maps is None or self.path_to_fixation_maps is None:
                sample = [torch.cat(frame_img, 0)]
            else:
                sample = [torch.cat(frame_img, 0), torch.cat(label, 0),torch.cat(fixation,0)]
            

        if self.transform:
            tf = Rotate()
            return tf(sample)
        return sample[0], sample[1], sample[2]

class Rotate(object):
    """
    Rotate the 360º image with respect to the vertical axis on the sphere.
    """

    def __call__(self, sample):
        
        input = sample[0]
        sal_map = sample[1]
        fix_map=sample[2]
        t = np.random.randint(input.shape[-1])

        new_sample = sample
        new_sample[0] = torch.cat((input[:,:,:,t:], input[:,:,:,0:t]),dim=3)
        new_sample[1] = torch.cat((sal_map[:,:,:,t:], sal_map[:,:,:,0:t]),dim=3)
        new_sample[2] = torch.cat((fix_map[:, :, :, t:], fix_map[:, :, :, 0:t]), dim=3)

        return new_sample
# train_video360_dataset = RGB(config.frames_dir,  config.gt_dir, config.fixation_dir,config.train_set, config.sequence_length, split='train', resolution=config.resolution,transform=True)
# train_data= DataLoader(train_video360_dataset,batch_size=1,shuffle=True)
# train_data_tqdm=tqdm(train_data)
# epoch=0
# for x,y,z in train_data_tqdm:
#     epoch=epoch+1
# pass