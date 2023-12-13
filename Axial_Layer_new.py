import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=56, stride=1, height_dim=True, inference=False):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False).to(device)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        self.w_z = nn.Conv2d(in_channels*2,in_channels*2,1,padding="same")
        self.w = nn.Conv2d(in_channels*3,in_channels*3,1,padding="same")
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        self.y_rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        y_key_index = torch.arange(kernel_size)
        y_query_index = torch.arange(kernel_size)
        self.dropout=nn.Dropout(p=0.7)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        y_distance_matrix = (y_key_index[None, :] - y_query_index[:, None]) + kernel_size - 1

        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x, y):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
            y = y.permute(0, 3, 1, 2)
        else:
            x = x.permute(0, 2, 1, 3)
            y = y.permute(0, 2, 1, 3)# batch_size, height, depth, width
            
        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)
        y = y.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        kqv = self.kqv_bn(kqv) # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)

        y_kqv = self.kqv_conv(y)
        y_kqv = self.kqv_bn(y_kqv) # apply batch normalization on k, q, v
        y_k, y_q, y_v = torch.split(y_kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2, self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        y_rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2,self.kernel_size,self.kernel_size)
        y_q_encoding, y_k_encoding, y_v_encoding = torch.split(y_rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        y_qk = torch.matmul(q.transpose(2, 3), y_k)
        y_qr = torch.einsum('bhdx,dxy->bhxy', y_q, y_q_encoding)
        y_kr = torch.einsum('bhdx,dxy->bhxy', y_k, y_k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        #logits = self.dropout(logits)
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        y_logits = torch.cat([y_qk, y_qr, y_kr], dim=1)
        y_logits = self.logits_bn(y_logits) # apply batch normalization on qk, qr, kr
        #y_logits = self.dropout(y_logits)
        y_logits = y_logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)

        weights = F.softmax(logits, dim=3)
        y_weights=F.softmax(y_logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        y_attn = torch.matmul(y_weights, y_v.transpose(2,3)).transpose(2,3)
        y_attn_encoding = torch.einsum('bhxy,dxy->bhdx', y_weights, y_v_encoding)
        y_attn_out = torch.cat([y_attn, y_attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        y_output = y_attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
            y_output = y_output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
            y_output = y_output.permute(0, 2, 1, 3)
        if self.height_dim:
            Z = torch.cat([output,y_output], dim=1)
            Z = self.w_z(Z)

            # channel concat of Z and h
            W = torch.cat([Z, output], dim=1)
            W = self.w(W)

            # mi_conv: Wm;zi * Z + Wm;hi * Ht + bm;i
            # mg_conv: Wm;zg * Z + Wm;hg * Ht + bm;g
            # mo_conv: Wm;zo * Z + Wm;ho * Ht + bm;o
            mi_conv, mg_conv, mo_conv = torch.chunk(W, chunks=3, dim=1)
            input_gate = torch.sigmoid(mi_conv)
            g = torch.tanh(mg_conv)
            new_M = (1 - input_gate) * y_output + input_gate * g
            output_gate = torch.sigmoid(mo_conv)
            new_H = output_gate * new_M

            return new_H, new_M
        else:
            return output,y_output


#x=torch.rand(1,16,60,80)
#y=torch.rand(1,16,60,80)
#model_x=Axial_Layer(in_channels=16, num_heads=1, kernel_size=80, stride=1, height_dim=False, inference=False)
#model_y=Axial_Layer(in_channels=18, num_heads=1, kernel_size=60, stride=1, height_dim=True, inference=False)
#y=model_x(x,y)
# x,y=model_y(x,y)
# print(x.size(),y.size())
