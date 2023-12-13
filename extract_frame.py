import os
import cv2
from tqdm import tqdm
import config


samples_per_second = 8
# destination_folder = os.path.join(config.videos_folder, 'frames')
destination_folder='/media/yf302/Lenovo/vr-eyetracking/frames'
videos_folder = '/media/yf302/Lenovo/vr-eyetracking/videos'

video_names = os.listdir(videos_folder)
video_names = sorted(video_names, key=lambda x: int((x.split(".")[0])))

# with tqdm(range(len(video_names)), ascii=True) as pbar:
for v_n, video_name in enumerate(video_names):

        video = cv2.VideoCapture(os.path.join(videos_folder, video_name))
        n=video.get(7)
        fps = video.get(cv2.CAP_PROP_FPS)
        step = 1
        video_name=video_name.split('.')[0]
        new_video_folder = os.path.join(destination_folder, video_name)
        # new_video_folder = os.path.join(destination_folder, str(v_n).zfill(3))

        if not os.path.exists(new_video_folder):
            os.makedirs(new_video_folder)

        success, frame = video.read()
        frame_id = 0
        frame_name = video_name + '_' + str(frame_id).zfill(4) + '.png'
        # frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
        frame = cv2.resize(frame, (320, 240))
        cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
        frame_id += 1

        while success:
            success, frame = video.read()
            if frame_id % step == 0 and success:
                # frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
                frame_name = video_name + '_' + str(frame_id).zfill(4) + '.png'
                frame = cv2.resize(frame, (320, 240))
                cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
            frame_id += 1
        # pbar.update(1)
