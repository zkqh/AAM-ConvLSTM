import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from vr_dataloader import  RGB
from sphericalKLDiv import  KLWeightedLossSequence
#from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import time
from saliencyMeasures import normalize,KLD,CC,AUC_Judd,AUC_Borji,SIM,NSS,AUC_shuffled
from torch.utils.data import DataLoader
import models
from utils import read_txt_file
from metrics import cc, similarity, nss, kldiv, auc_judd,nss_v,cc_v,kldiv_v,spher_weight
# Import config file
import config
from tqdm import tqdm
from data import VRVideo
import pandas as pd
import numpy as np

def loss_function(prediction, label,fix):
    label=label.cpu()
    prediction=prediction.cpu()
    fix = fix.cpu()
    # othermap = np.load('/media/yf302/Lenovo/TASED-Net-master/othermap.npy')
    prediction,label=spher_weight(prediction,label)
    prediction = normalize(prediction, method='sum')
    label = normalize(label, method='sum')
    # kldiv = k(prediction, label)
    # othermap=np.resize(othermap,(240,320))
    # othermap=np.expand_dims(othermap,axis=0)
    # othermap = np.expand_dims(othermap, axis=1)

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
        # au_b=AUC_shuffled(prediction[:,i,:,:],fix[:,i,:,:],othermap)
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

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=0.0001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08, betas=(0.9, 0.999), weight_decay=0,amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # model.train()

    model.to(device)
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []
    cu_cc = []

    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0

        kldiv_list = []
        cc_list = []
        nss_list = []
        au_j_list=[]
        au_b_list=[]

        time.sleep(0.01)
        train_data_tqdm = tqdm(train_data)
        train_data_tqdm.set_description('Epoch {}'.format(epoch+1))
        for x, y,z in train_data_tqdm:

            model.zero_grad()
            pred = model(x.to(device))
            loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device), z[:, :, 0, :, :].to(device))
            loss.sum().backward()
            #torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            avg_loss_train += loss.sum().item()
            counter_train += 1
            train_data_tqdm.set_postfix(Average_Loss=avg_loss_train / counter_train)
            # if counter_train % 20 == 0:
            #     print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter_train, len(train_data),
            #                                                                             avg_loss_train / counter_train))
        time.sleep(0.01)
        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))

        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            for x, y, z in val_data:
                counter_val += 1
                # x = x.permute(0, 2, 1, 3, 4)
                pred = model(x.to(device))
                loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device),z[:, :, 0, :, :].to(device))
                _kldiv, _cc, _nss, _auj= loss_function(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device),
                                                              z[:, :, 0, :, :].to(device))

                kldiv_list.append(_kldiv)
                cc_list.append(_cc)
                nss_list.append(_nss)
                au_j_list.append(_auj)
                # au_b_list.append(_aub)
                avg_loss_val += loss.sum().item()

            print("val----epoch{}----avgloss{}---kl{}----cc{}----nss{}----auj{}".format(epoch, (
                        avg_loss_val / counter_val),
                                                                                                 (sum(kldiv_list) / len(
                                                                                                     kldiv_list)),
                                                                                                 (sum(cc_list) / len(
                                                                                                     cc_list)),
                                                                                                 (sum(nss_list) / len(
                                                                                                     nss_list)),
                                                                                                 (sum(au_j_list) / len(
                                                                                                     au_j_list))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Train SST-Sal

    model = models.SST_Sal(hidden_dim=9)

    model_trained = torch.load(config.inference_model, map_location=device)
    dict_new=model.state_dict()
    state_trained=model_trained.state_dict()
    model_load={k: v for k, v in state_trained.items() if k in dict_new}
    dict_new.update(model_load)
    model.load_state_dict(dict_new)

    criterion = KLWeightedLossSequence()

    train_video360_dataset = RGB(config.frames_dir,  config.gt_dir, config.fixation_dir,config.VR_TRAIN_set, config.sequence_length, resolution=config.resolution,transform=False)
    val_video360_dataset = RGB(config.frames_dir,  config.gt_dir, config.fixation_dir,config.VR_Test_set, config.sequence_length,  resolution=config.resolution)

    train_data = DataLoader(train_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)
    val_data = DataLoader(val_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=False)


    model = train(train_data, val_data, model, device, criterion, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)
    print("Training finished")