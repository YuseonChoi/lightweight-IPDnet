import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Module as at_module
import math
from utils_ import pad_segments,split_segments
from torch.autograd import Variable

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(dimension))
            self.bias = nn.Parameter(torch.zeros(dimension))
        else:
            self.gain = Variable(torch.ones(dimension), requires_grad=False)
            self.bias = Variable(torch.zeros(dimension), requires_grad=False)

    def forward(self, input):
        # input size: ([Batch, Time, *, Channel])
        # cumulative mean for each time step

        input_shape = input.shape
        batch, time_step, channel = input_shape[0], input_shape[1], input_shape[-1]
        
        input0 = input
        input = input0.view(batch, time_step, -1, channel)
        etc_dim = input.size(2)
        input = input.transpose(1, 2).contiguous().view(-1, time_step, channel)

        step_sum = input.sum(-1)  # B, T
        step_pow_sum = input.pow(2).sum(-1)  # B, T
        cum_sum = torch.cumsum(step_sum, -1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, -1)  # B, T

        entry_cnt = torch.arange(channel, channel*(time_step+1), channel, device=input.device)
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = cum_pow_sum / entry_cnt - cum_mean.pow(2)  # B, T
        cum_std = (torch.clip(cum_var, self.eps)).sqrt()  # B, T

        cum_mean = cum_mean.view(batch, etc_dim, time_step, 1).transpose(1, 2).contiguous().view(*input_shape[:-1], 1)
        cum_std = cum_std.view(batch, etc_dim, time_step, 1).transpose(1, 2).contiguous().view(*input_shape[:-1], 1)

        x = (input0 - cum_mean) / cum_std
        return x * self.gain + self.bias


class FNblock(nn.Module):
    """
    The implementation of the full-band and narrow-band fusion block
    """
    def __init__(self, input_size, hidden_size=128, dsample_factor=2, dropout=0.2, is_online=False):
        super(FNblock, self).__init__()
        self.input_size = input_size
        self.full_hidden_size =  hidden_size // 2
        
        self.dsample_conv = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=(3,dsample_factor*2), stride=(1,dsample_factor)),
            nn.PReLU())
        self.norm = cLN(hidden_size)
        self.dsample_factor = dsample_factor

        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size  // 2
        self.dropout = dropout
        self.dropout_full =  nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        self.fullLstm = nn.LSTM(input_size=hidden_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)       
    def forward(self, x):
        x = self.dsample_conv(F.pad(x.permute(0,3,2,1), (self.dsample_factor, 0, 1, 1)))
        x = self.norm(x.permute(0, 3, 2, 1))
        
        nb, nt, nf, nc = x.shape
        fb_skip = x.reshape(nb*nt,nf,nc)
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,nc)
        x = x.reshape(nb*nt,nf,-1)
        x, _ = self.fullLstm(x)
        x = self.dropout_full(x)
        x = x + fb_skip
        x = x.view(nb,nt,nf,-1).permute(0,2,1,3).reshape(nb*nf,nt,-1) 
        x, _ = self.narrLstm(x)
        x = self.dropout_narr(x)
        x = x + nb_skip
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        return x


class FNblockNorm(nn.Module):
    """
    The implementation of the full-band and narrow-band fusion block
    """
    def __init__(self, input_size, hidden_size=128, dsample_factor=2, dropout=0.2, is_online=False, norm_type="cLN"):
        super(FNblockNorm, self).__init__()
        self.input_size = input_size
        self.full_hidden_size =  hidden_size // 2
        
        self.norm_type = norm_type
        self.dsample_conv = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=(3,dsample_factor*2), stride=(1,dsample_factor)),
            nn.PReLU())
        if norm_type == "LN":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm_type == "cLN":
            self.norm = cLN(hidden_size)
        else:
            raise NotImplementedError("norm_type must be one of [\"LN\", \"cLN\"]")
        self.dsample_factor = dsample_factor
        
        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size  // 2
        self.dropout = dropout
        self.dropout_full =  nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        self.norm_full = cLN(hidden_size)
        self.norm_narr = cLN(hidden_size)
        self.fullLstm = nn.LSTM(input_size=hidden_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)       
    def forward(self, x):
        x = self.dsample_conv(F.pad(x.permute(0,3,2,1), (self.dsample_factor, 0, 1, 1)))
        x = self.norm(x.permute(0, 3, 2, 1))
        
        nb, nt, nf, nc = x.shape
        fb_skip = x.reshape(nb*nt,nf,nc)
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,nc)
        x = x.reshape(nb*nt,nf,-1)
        x, _ = self.fullLstm(x)

        x = self.norm_full((x + fb_skip).view(nb,nt,nf,-1))
        x = self.dropout_full(x)
        x = x.permute(0,2,1,3).reshape(nb*nf,nt,-1) 
        x, _ = self.narrLstm(x)
        x = self.norm_narr((x + nb_skip).view(nb,nf,nt,-1).permute(0,2,1,3))
        x = self.dropout_narr(x)
        return x

class CausCnnBlock(nn.Module): 
	""" 
    Function: Basic causal convolutional block
    """
	# expansion = 1
	def __init__(self, inp_dim,out_dim,cnn_hidden_dim=128,kernel=(3,3), stride=(1,1), padding=(1,2),dsample_factor=1):
		super(CausCnnBlock, self).__init__()

        # convlutional layers
		self.conv1 = nn.Conv2d(inp_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.conv3 = nn.Conv2d(cnn_hidden_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		
        # Time compression           
		self.pad = padding
		self.relu = nn.ReLU(inplace=True)
		self.tanh = nn.Tanh()
	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = out[:, :, :, :-self.pad[1]]
		out = self.conv2(out)
		out = self.relu(out)
		out = out[:, :, :, :-self.pad[1]]
		out = self.conv3(out)
		out = out[:, :, :, :-self.pad[1]]
		out = self.tanh(out)
		return out


class IPDnet(nn.Module):
    '''
    The implementation of the IPDnet
    '''
    def __init__(self,input_size=4,hidden_size=128,num_layers=3,intermediate_layer=0,max_track=2,is_online=True,n_seg=312,dropout=0.2,use_norm_fn=False,norm_type="LN"):
        super(IPDnet, self).__init__()
        self.input_size = input_size
        self.is_online = is_online
        self.hidden_size = hidden_size
        self.intermediate_layer = intermediate_layer
        
        dsample_block_idx = [i*(num_layers//3)-1 for i in range(1, 4)]
        self.block = nn.ModuleList([])
        if use_norm_fn:
            for i in range(num_layers):
                if i == dsample_block_idx[0] or i == dsample_block_idx[1]:
                    if i == 0:
                        self.block.append(FNblockNorm(input_size=self.input_size,hidden_size=self.hidden_size,dsample_factor=2,dropout=dropout,is_online=self.is_online,norm_type=norm_type))
                    else:
                        self.block.append(FNblockNorm(input_size=self.hidden_size,hidden_size=self.hidden_size,dsample_factor=2,dropout=dropout,is_online=self.is_online,norm_type=norm_type))
                elif i == dsample_block_idx[2]:
                    self.block.append(FNblockNorm(input_size=self.hidden_size,hidden_size=self.hidden_size,dsample_factor=3,dropout=dropout,is_online=self.is_online,norm_type=norm_type))
                else:
                    if i == 0:
                        self.block.append(FNblockNorm(input_size=self.input_size,hidden_size=self.hidden_size,dsample_factor=1,dropout=dropout,is_online=self.is_online,norm_type=norm_type))
                    else:
                        self.block.append(FNblockNorm(input_size=self.hidden_size,hidden_size=self.hidden_size,dsample_factor=1,dropout=dropout,is_online=self.is_online,norm_type=norm_type))
        else:
            for i in range(num_layers):
                if i == dsample_block_idx[0] or i == dsample_block_idx[1]:
                    if i == 0:
                        self.block.append(FNblock(input_size=self.input_size,hidden_size=self.hidden_size,dsample_factor=2,dropout=dropout,is_online=self.is_online))
                    else:
                        self.block.append(FNblock(input_size=self.hidden_size,hidden_size=self.hidden_size,dsample_factor=2,dropout=dropout,is_online=self.is_online))
                elif i == dsample_block_idx[2]:
                    self.block.append(FNblock(input_size=self.hidden_size,hidden_size=self.hidden_size,dsample_factor=3,dropout=dropout,is_online=self.is_online))
                else:
                    if i == 0:
                        self.block.append(FNblock(input_size=self.input_size,hidden_size=self.hidden_size,dsample_factor=1,dropout=dropout,is_online=self.is_online))
                    else:
                        self.block.append(FNblock(input_size=self.hidden_size,hidden_size=self.hidden_size,dsample_factor=1,dropout=dropout,is_online=self.is_online))
        self.cnn_out_dim = 2 * ((input_size // 2) - 1) * max_track
        self.cnn_inp_dim = hidden_size # + input_size
        self.conv = CausCnnBlock(inp_dim = self.cnn_inp_dim, out_dim = self.cnn_out_dim)
        self.n = n_seg
    def forward(self,x,offline_inference=False):
        
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape
        ou_frame = nt//12
        # chunk-wise inference for offline 
        if not self.is_online and offline_inference:
            x = split_segments(x, self.n)  # Split into segments of length n
            nb,nseg,seg_nt,nf,nc = x.shape
            x = x.reshape(nb*nseg,seg_nt,nf,nc)
            nb,nt,nf,nc = x.shape
        
        # FN blocks
        for block in self.block:
            x = block(x)
        nb,nt,nf,nc = x.shape
        x = x.permute(0,3,2,1)
        
        nt2 = nt
        
        x = self.conv(x).permute(0,3,2,1).reshape(nb,nt2,nf,2,-1).permute(0,1,3,2,4)
        if not self.is_online and offline_inference: 
            x = x.reshape(nb//nseg,nt2*nseg,2,nf*2,-1).permute(0,1,3,4,2)
            output = x[:,:ou_frame,:,:,:]
        else:
            output = x.reshape(nb,nt2,2,nf*2,-1).permute(0,1,3,4,2)
        return output




if __name__ == "__main__":
    x = torch.randn((1,4,257,280)) 
    # for 2-mic IPDnet
    model = IPDnet(hidden_size=128,num_layers=3, dropout=0.1)

    # for >2-mic IPDnet, for example, 4-mic IPDnet
    #model = IPDnet(input_size=8,hidden_size=256,max_track=2,is_online=True)
    import time
    ts = time.time()
    y = model(x)
    te = time.time()
    print(model)
    print(y.shape)
    print(te - ts)
    model = model.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(model, display=False) as fcm:
        y = model(x)
        flops_forward_eval = (fcm.get_total_flops()) / 4.5
        res = y.sum()
        # res.backward()
        # flops_backward_eval = (fcm.get_total_flops() - flops_forward_eval) / 4.5
    params_eval = sum(param.numel() for param in model.parameters())
    # print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")