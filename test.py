import os
import numpy as np
import torch
import config
from DataLoader360Video import RGB_and_OF, RGB
from torch.utils.data import DataLoader
import cv2
import tqdm
from utils import frames_extraction
from metrics import cc_v, similarity, nss_v, kldiv_v, auc_judd
from utils import save_video,read_txt_file
import pandas as pd
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
videos_folder = '/media/vilab/141E39B41E39902A/VR_sal/videos'

def loss_function(prediction, label, fixation):
    _kldiv = kldiv_v(prediction, label)
    _cc = cc_v(prediction, label)
    _nss = nss_v(prediction, fixation)
    return _kldiv, _cc, _nss
def dellist(L):
    L1=[]
    for i in L:
        if i not in L1:
            L1.append(i)
    return L1
def eval(test_data, model, device, result_imp_path):

    model.to(device)
    model.eval()
    counter_val = 0
    kldiv_list = []
    cc_list = []
    nss_list = []
    list=[]
    with torch.no_grad():
        current_time = time.time()
        test_data_tqdm = tqdm.tqdm(test_data)
        for x,y,z,names in test_data_tqdm:
            counter_val += 1

            pred = model(x.to(device))
            # endtime = time.time()
            # if endtime-current_time>0.01:
            #     break
            pre_list=[]
            y_list=[]
            z_list=[]
            batch_size, Nframes, h_x, w_x = pred[:, :, 0, :, :].shape
            for bs in range(1):
                for iFrame in range(4, Nframes):
                    pred_in=pred[bs, iFrame, 0, :, :].cpu()
                    y_in=y[bs, iFrame, 0, :, :].cpu()
                    z_in=z[bs, iFrame, 0, :, :].cpu()
                    pred_in=np.array(pred_in)
                    y_in=np.array(y_in)
                    z_in=np.array(z_in)
                    pred_in=cv2.resize(pred_in,(1024,2048))
                    y_in=cv2.resize(y_in,(1024,2048))
                    z_in=cv2.resize(z_in,(1024,2048))
                    pred_in=torch.tensor(pred_in)
                    y_in = torch.tensor(y_in)
                    z_in = torch.tensor(z_in)
                    pre_list.append(pred_in)
                    y_list.append(y_in)
                    z_list.append(z_in)
            pred=torch.stack(pre_list,dim=0)
            y=torch.stack(y_list,dim=0)
            z=torch.stack(z_list,dim=0)
            pred=pred.unsqueeze(1)
            y = y.unsqueeze(1)
            z = z.unsqueeze(1)
            pred=pred.unsqueeze(0)
            y = y.unsqueeze(0)
            z = z.unsqueeze(0)
            pred=pred.to(device)
            y = y.to(device)
            z=z.to(device)
            # pred=cv2.resize(pred,(2048,1024))
            _kldiv, _cc, _nss = loss_function(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device), z[:, :, 0, :, :].to(device))
            batch_size, Nframes, h_x,w_x = pred[:, :, 0, :, :].shape
            bs,ns,h_y,w_y=y[:,:,0,:,:].shape

            for bs in range(batch_size):
                for iFrame in range(4,Nframes):
                        video = cv2.VideoCapture(os.path.join(videos_folder, names[iFrame][bs].split('_')[0]+'.mp4'))
                        video_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                        video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        folder = os.path.join(result_imp_path, names[iFrame][bs].split('_')[0])
                        if not os.path.exists(folder):
                               os.makedirs(folder)
                        sal = pred[bs, iFrame, 0, :, :].cpu()
                        sal = np.array((sal - torch.min(sal)) / (torch.max(sal) - torch.min(sal)))
                        sal=cv2.resize(sal,(video_width,video_HEIGHT))
                        cv2.imwrite(os.path.join(folder, names[iFrame][bs]+'_%.2f'%_cc + '.png'), (sal * 255).astype(np.uint8))
            list_in = [names[iFrame][bs].split('_')[0], str(video_width) + '*' + str(video_HEIGHT), str(video_width) + '*' + str(video_HEIGHT)]
            list.append(list_in)
            LIST=dellist(list)
            kldiv_list.append(_kldiv.cpu().item())
            cc_list.append(_cc.cpu().item())
            nss_list.append(_nss.cpu().item())

    column=['视频名称','视频帧图分辨率','显著性图分辨率']
    test=pd.DataFrame(columns=column,data=LIST)
    test.to_csv('/media/vilab/11be89dd-c494-4cd1-a2b7-cb0fd483b60a/vilab/全景视频显著性测试软件-分析结果/分辨率结果.csv',index=False)
    # print("CSV file created in /home/vilab/Desktop/saliency_prediction")
    print("cc 线性相关系数 {:.3f}".format((sum(cc_list) / len(cc_list))))
    print("done")



if __name__ == "__main__":

    # Extract video frames if hasn't been done yet
    #if not os.path.exists(os.path.join(config.videos_folder, 'frames')):
       # frames_extraction(config.videos_folder)
    # Obtain video names from the new folder 'frames'
    video_test_names =read_txt_file(config.videos_val_file)

    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device") 

    # Load the model
    model =  torch.load(config.inference_model, map_location=device)

    # Load the data. Use the appropiate data loader depending on the expected input data
    if config.of_available:
        test_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, None, video_test_names, config.sequence_length, split='test', load_names=True)
    else:
        test_video360_dataset = RGB(config.frames_dir, config.gt_dir, config.fixation_dir, video_test_names, config.sequence_length, split='test',load_names=True, resolution=config.resolution,transform=False)

    test_data = DataLoader(test_video360_dataset, batch_size=config.batch_size, shuffle=False)
    eval(test_data, model, device, config.results_dir)

    # print("done")


