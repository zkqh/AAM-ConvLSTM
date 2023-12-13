import torch.nn as nn
import torch
from spherenet import SphereConv2D, SphereMaxPool2D
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from Axial_Layer_new import Axial_Layer
from dropblock import DropBlock2D


init_func = {
    'uniform':nn.init.uniform_,
    'normal':nn.init.normal_,
    'constant':nn.init.constant_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
}


class Tconv2d(nn.Module):
    def __init__(self, input_step, input_channel, output_channel, kernel=3, bias=True):
        super(Tconv2d, self).__init__()
        self.input_step = input_step
        self.op_conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel, bias=bias)

    def forward(self, x):
        temp = []
        for i in range(self.input_step):
            data = x[:,i,:,:,:]
            data = self.op_conv(data)
            temp.append(data)

        return torch.stack(temp,dim=1)

def init_weights(model, funcname='xavier_uniform'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init_func[funcname](m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0.)

class SST_Sal(nn.Module):
    pass

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size,num_layers,
                 bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list= []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # cell_list.append(SpherConvLSTMCell(input_dim=cur_input_dim, hidden_dim=hidden_dim, bias=bias))
            cell_list.append(SST_Sal)
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
            # raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class SpherConvLSTM_EncoderCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, stride=4):
        super(SpherConvLSTM_EncoderCell, self).__init__()

        # Encoder
        self.lstm = SpherConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, bias=bias)
        self.pool = SphereMaxPool2D(stride=stride)
        self.conv_u = SphereConv2D(hidden_dim,4*hidden_dim,bias=True)
        self.conv_d = SphereConv2D(4*hidden_dim,hidden_dim,bias=True)
        self.up_sampling = nn.Upsample(scale_factor=2)
        self.dropout= nn.Dropout(p=0.7)
        self.attention_memory_w=Axial_Layer(in_channels=4*hidden_dim,num_heads=1,kernel_size=80,stride=1,height_dim=False,inference=False)
        self.attention_memory_h = Axial_Layer(in_channels=4*hidden_dim, num_heads=1, kernel_size=60, stride=1,height_dim=True, inference=False)
    def forward(self, x, state,prev_memory):

        h, c = self.lstm(x, state)
        out = self.conv_u(h)
        out=self.pool(out)
        out, prev_memory = self.attention_memory_w(out, prev_memory)
        out, prev_memory = self.attention_memory_h(out, prev_memory)
        out=self.conv_d(out)
        out=self.up_sampling(out)
        return out, [h, c],prev_memory

    def init_hidden(self, b, shape):
        h, c = self.lstm.init_hidden(b, shape)
        return [h, c]


class SpherConvLSTM_DecoderCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, scale_factor=2):
        super(SpherConvLSTM_DecoderCell, self).__init__()

        # Decoder
        self.lstm = SpherConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, bias=bias)
        self.up_sampling = nn.Upsample(scale_factor=2)
        self.pool = SphereMaxPool2D(stride=4)
        
    def forward(self, x, state):

        h, c = self.lstm(x,state)
        out = self.up_sampling(h)
        return out, [h, c]

    def init_hidden(self, b, shape):
        h, c = self.lstm.init_hidden(b, shape)
        return [h, c]

class SpherConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, bias):
        """
        Initialize Spherical ConvLSTM cell.
        ----------
        input_dim: Number of channels of input tensor.
        hidden_dim: Dimension of the hidden states.  
        bias: Whether or not to add the bias.
        """

        super(SpherConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = (3,3) # Spherical convolutions only compatible with 3x3 kernels
        self.bias = bias

        self.conv = SphereConv2D(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)

        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
                )
        #return (torch.zeros(batch_size, self.hidden_dim, height, width),
                #torch.zeros(batch_size, self.hidden_dim, height, width,))






