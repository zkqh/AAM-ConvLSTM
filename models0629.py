import Modules0629
import torch.nn as nn
import torch
from axial_attention import AxialAttention,AxialPositionalEmbedding,AxialImageTransformer
from Modules0629 import ChannelAttention1,Tconv2d
from dropblock import DropBlock2D
from gaussian import get_guasspriors_3d
from spherenet import SphereConv2D, SphereMaxPool2D

class Tconv2d(nn.Module):
    def __init__(self, input_step, input_channel, output_channel, kernel=3, bias=True):
        super(Tconv2d, self).__init__()
        self.input_step = input_step
        self.op_conv = SphereConv2D(input_channel, output_channel,bias=bias)
        #self.op_conv = nn.Conv2d(input_channel,output_channel, kernel_size=kernel, bias=bias,padding=1)
        self.dropblock = DropBlock2D(block_size=3, drop_prob=0.7)
    def forward(self, x):
        temp = []
        for i in range(self.input_step):
            data = x[:,i,:,:,:]
            data = self.op_conv(data)
            temp.append(data)

        return torch.stack(temp).permute(1,0,2,3,4)

class SST_Sal(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=9, output_dim=1):
        super(SST_Sal, self).__init__()
        self.encoder = Modules0629.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules0629.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)
        self.encoder_rev = Modules0629.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder_rev = Modules0629.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)

        # self.fusion_conv = Tconv2d(input_step=10, input_channel=output_dim * 2, output_channel=output_dim, kernel=1)
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.dropout=nn.Dropout(p=0.7)
        self.dropblock=DropBlock2D(block_size=3,drop_prob=0.7)
        self.bn=nn.BatchNorm2d(num_features=9)

        self.gauconv1 = Tconv2d(input_step=10, input_channel=8, output_channel=32, kernel=3)
        self.gauconv2 = Tconv2d(input_step=10, input_channel=32, output_channel=32, kernel=3)
        self.cgpconv1 = Tconv2d(input_step=10, input_channel=32 + 1, output_channel=33, kernel=3)
        self.cgpconv2 = Tconv2d(input_step=10, input_channel=33, output_channel=1, kernel=3)

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):

        b, _, _, h, w = x.size()

        state_e_f = self.encoder.init_hidden(b, (h, w))
        state_e_b = self.encoder.init_hidden(b, (h, w))


        state_e_f_prevmemory = torch.zeros(b, 36, h//4, w//4,device=self.encoder.lstm.conv.weight.device)
        state_e_b_prevmemory = torch.zeros(b, 36, h//4, w//4,device=self.encoder.lstm.conv.weight.device)
        
        state_d_f = self.decoder.init_hidden(b, (h//2, w//2))
        state_d_b = self.decoder.init_hidden(b, (h//2, w//2))
        
        state_d_f_prevmemory = torch.zeros(b, self.output_dim * 32, h//8, w//8,device=self.encoder.lstm.conv.weight.device)
        state_d_b_prevmemory = torch.zeros(b, self.output_dim * 32, h//8, w//8,device=self.encoder.lstm.conv.weight.device)

        outputs_f = []
        outputs_b = []

        for t in range(x.shape[1]):
            out_f, state_e_f ,state_e_f_prevmemory= self.encoder(x[:, t, :, :, :], state_e_f, state_e_f_prevmemory)
            out_f=self.dropblock(out_f)
            out_f = self.bn(out_f)
            out_f, state_d_f = self.decoder(out_f, state_d_f)
            outputs_f.append(out_f)
        x_f=torch.stack(outputs_f, dim=1)


        for t_b in range(x.shape[1]-1,-1,-1):
             out_b, state_e_b ,state_e_b_prevmemory= self.encoder(x[:, t_b, :, :, :], state_e_b, state_e_b_prevmemory)
             out_b=self.dropblock(out_b)
             out_b = self.bn(out_b)
             out_b, state_d_b = self.decoder(out_b, state_d_b)
             outputs_b.append(out_b)
             outputs_b.reverse()
        x_b=torch.stack(outputs_b, dim=1)
        #
        x=x_b+x_f

        gaussian = get_guasspriors_3d(type='dy', b_s=1, time_dims=10, shape_r=240, shape_c=320, channels=8)
        gaussian = torch.tensor(gaussian).cuda()
        gaussian = gaussian.permute(0, 1, 4, 2, 3)
        gaussian = gaussian.float()
        gaussian = self.gauconv1(gaussian)
        gaussian = self.gauconv2(gaussian)
        gaussian = self.relu(gaussian)
        cgp = torch.cat((x, gaussian), 2)
        cgp = self.cgpconv1(cgp)
        sal = self.cgpconv2(cgp)
        return sal


if __name__ == '__main__':

    sst=SST_Sal(input_dim=3,hidden_dim=9,output_dim=1).cuda()
    x=torch.randn(1,10,3,240,320).cuda()
    y=sst(x)
    print(y.size())
