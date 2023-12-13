import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import numpy as np
import torchvision.transforms as tf
from data import VRVideoLSTM,VRVideo
from torch.utils import data as tdata
import torch
from sphericalKLDiv import KLWeightedLossSequence
from newloss import SphereMSE
import datetime
import os
import time
from torch.utils.data import DataLoader
import models_ablation_1
from utils import read_txt_file
from saliencyMeasures import normalize,KLD,CC,AUC_Judd,AUC_Borji,SIM,NSS
import config
from tqdm import tqdm
import pandas as pd
import csv

def loss_function(prediction, label,fix):
    label=label.cpu()
    prediction=prediction.cpu()
    fix = fix.cpu()
    prediction,label=spher_weight(prediction,label)
    prediction = normalize(prediction, method='sum')
    label = normalize(label, method='sum')
    # kldiv = k(prediction, label)
    cc_l=[]
    nss_l=[]
    au_j_l=[]
    au_b_l=[]
    kld_l=[]
    for i in range(0,10):
        kld =KLD(prediction[:,i,:,:], label[:,i,:,:])
        cc = CC(prediction[:,i,:,:], label[:,i,:,:])
        nss =NSS(prediction[:,i,:,:],fix[:,i,:,:])
        au_j=AUC_Judd(prediction[:,i,:,:],fix[:,i,:,:])
        # au_b=AUC_Borji(prediction[:,i,:,:],fix[:,i,:,:])
        cc_l.append(cc)
        nss_l.append(nss)
        au_j_l.append(au_j)
        # au_b_l.append(au_b)
        kld_l.append(kld)

    return sum(kld_l)/len(kld_l),sum(cc_l)/len(cc_l),sum(nss_l)/len(nss_l),sum(au_j_l)/len(au_j_l)

def train(train_data, val_data, model, device, criterion, lr = 0.01, EPOCHS=10, model_name='Model'):

    #writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(path)
    os.mkdir(ckp_path)
    optimizer =torch.optim.Adam(model.parameters(),lr=lr,eps=1e-08, betas=(0.9, 0.999),weight_decay=0,amsgrad=False)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    model.train()
    model.to(device)
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []
    cu_cc = []
    #
    # # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0

        kldiv_list = []
        cc_list = []
        nss_list=[]
        au_j_list=[]
        au_b_list=[]
    # #
        time.sleep(0.01)
        train_data_tqdm = tqdm(train_data)
        train_data_tqdm.set_description('Epoch {}'.format(epoch+1))

        for x,z,f in train_data_tqdm:

            model.zero_grad()
            x=x.cuda()
            z=(z*10)
            f=(f*10)
            pred = model(x)
            loss = criterion(pred[:, :, 0, :, :], z[:, :, 0, :, :].to(device),f[:, :, 0, :, :].to(device))

            loss.sum().backward()
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            avg_loss_train += loss.sum().item()
            counter_train += 1
            train_data_tqdm.set_postfix(Average_Loss=avg_loss_train / counter_train)

        time.sleep(0.01)
        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))

        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            for x,z2,f2 in tqdm(val_data):
                counter_val += 1
                x=x.cuda()
                z2=(z2*10)
                pred = model(x.to(device))
                f2[f2 > 0] = 1
                loss = criterion(pred[:, :, 0, :, :], z2[:, :, 0, :, :].to(device),f2[:, :, 0, :, :].to(device))
                _kldiv, _cc,_nss,_auj= loss_function(pred[:, :, 0, :, :], z2[:, :, 0, :, :].to(device),f2[:, :, 0, :, :].to(device))

                kldiv_list.append(_kldiv)
                cc_list.append(_cc)
                nss_list.append(_nss)
                au_j_list.append(_auj)
                # au_b_list.append(_aub)
                avg_loss_val += loss.sum().item()

        print("val----epoch{}----avgloss{}---kl{}----cc{}----nss{}----auj{}".format(epoch, (avg_loss_val / counter_val),
                                                                                 (sum(kldiv_list) / len(kldiv_list)),
                                                                                 (sum(cc_list) / len(cc_list)),
                                                                      (sum(nss_list) / len(nss_list)),
                                                                      (sum(au_j_list) / len(au_j_list))
                                                                        ))
        lr_scheduler.step((sum(cc_list) / len(cc_list)))
        cu_cc.append(sum(cc_list) / len(cc_list))
        if (sum(cc_list) / len(cc_list))>= max(cu_cc):
            dict = {'cc': cc_list, 'nss': nss_list, 'auj': au_j_list}
            df = pd.DataFrame(dict)
            df.to_csv('test.csv')
            value = max(cu_cc)
            idx = cu_cc.index(value)
            torch.save(model, path + '/'+ 'cc'+str(cu_cc[idx])+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')
    # Save final model and checkpoints
    torch.save(model, path + '/model.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')

    return model


if __name__ == '__main__':

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    data_dir = '/media/yf302/Lenovo/360_Saliency_dataset_2018ECCV/360_Saliency_dataset_2018ECCV'
    # Train SST-Sal
    bs = 1
    transform = tf.Compose([
        tf.Resize((240, 320)),
        tf.ToTensor()
    ])
    model = models_ablation_1.SST_Sal(hidden_dim=9)
    model_trained = torch.load(config.inference_model, map_location='cpu')
    dict_new=model.state_dict()
    model_trained=model_trained.state_dict()
    model_load={k: v for k, v in model_trained.items() if k in dict_new}
    dict_new.update(model_load)
    model.load_state_dict(dict_new)

    # model_trained = torch.load('/home/yf302/Desktop/teacher_wan/0807sst/SST_Sal_wo_OF.pth', map_location=torch.device('cuda'))
    # state_trained = model_trained.state_dict()
    # dict_new = model.state_dict()
    # dict_trained = {k: v for k, v in state_trained.items() if k in dict_new}
    # dict_new['encoder_rev.lstm.conv.weight'] = model_trained['encoder.lstm.conv.weight']
    # dict_new['encoder_rev.lstm.conv.bias'] = model_trained['encoder.lstm.conv.bias']
    # dict_new['decoder_rev.lstm.conv.weight'] = model_trained['decoder.lstm.conv.weight']
    # dict_new['decoder_rev.lstm.conv.bias'] =model_trained['decoder.lstm.conv.bias']

    # 2. overwrite entries in the existing state dict
    # dict_new.update(dict_trained)
    # model.load_state_dict(dict_new)

    criterion = KLWeightedLossSequence()
    #criterion = SphereMSE(240,320).float()
    dataset_train = VRVideo(data_dir, 240, 320, 80, frame_interval=5, cache_gt=True, transform=transform, train=True,
                       gaussian_sigma=np.pi / 20, kernel_rad=np.pi / 7,frames_per_data=10)
    print(len(dataset_train))
    dataset_test=VRVideo(data_dir, 240, 320, 80, frame_interval=1, cache_gt=True, transform=transform, train=False,
                      gaussian_sigma=np.pi / 20, kernel_rad=np.pi / 7,frames_per_data=10)
    loader_train = tdata.DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
    loader_test= tdata.DataLoader(dataset_test, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    model = train(loader_train, loader_test, model, device, criterion, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)
    print("Training finished")





