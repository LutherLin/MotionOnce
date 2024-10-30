import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import scipy.stats as stats
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
# from models.x_transformers.x_transformers import ContinuousTransformerWrapper, Encoder
# from x_transformers import ContinuousAutoregressiveWrapper, ContinuousTransformerWrapper, Encoder,Decoder, TransformerWrapper
from models.cross_attention import (SkipTransformerEncoder,
                                    TransformerDecoder,
                                    TransformerDecoderLayer,
                                    TransformerEncoder,
                                    TransformerEncoderLayer)
from collections import OrderedDict

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class Conv(nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(p)),
             ('fc2', Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):

        input_ = input_.permute(1,0,2)
        out = self.layer(input_) # [seqlen, bs, d]

        return out




class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """
    def __init__(self, input_dims, outputs_per_step,num_hidden):
        """
        
        :param num_hidden: dimension of hidden 
        """
        super(PostConvNet, self).__init__()
        self.conv1 = Conv(in_channels=input_dims * outputs_per_step,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4,
                          w_init='tanh')
        self.conv_list = clones(Conv(in_channels=num_hidden,
                                     out_channels=num_hidden,
                                     kernel_size=5,
                                     padding=4,
                                     w_init='tanh'), 3)
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=input_dims * outputs_per_step,
                          kernel_size=5,
                          padding=4)

        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])

    def forward(self, input_, mask=None):
        # Causal Convolution (for auto-regressive)
        input_ = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(input_)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        return input_

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        # output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualMLP, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, zt):
        # 前向传播
        x = F.relu(self.fc1(zt))
        x = F.relu(self.fc2(x))
        residual = self.fc3(x)
        
        # 残差加回原输入
        y_prime_t = zt + residual
        
        return y_prime_t

class LatentSampling(nn.Module):
    def __init__(self,input_dim, output_dim, max_lens = 196//4):
        super().__init__()
        # output_dim 1052
        # input_dim 1024
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_lens = max_lens
        self.sampling = nn.Linear(self.input_dim, self.output_dim * 2)
        # self.sampling_mean = nn.Linear(self.input_dim, self.output_dim)
        # self.sampling_log_var = nn.Linear(self.input_dim, self.output_dim)


        
        self.residualMLP = ResidualMLP(input_dim=self.output_dim, hidden_dim= self.output_dim*2, output_dim=self.output_dim)
    def reparameterize(self, mean, log_var):
        # import pdb;pdb.set_trace()
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  
        return mean + eps * std 
    def forward(self, x):
        '''
        x: [bs, nframes, dim]
        '''
        # device = x.device
        # bs, nframes, nfeats = x.shape
        # x = x.permute(1, 0, 2) # now [nframes,bs, dim]
        x = self.sampling(x)
        # dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        mean = x[..., :self.output_dim]
        log_var = x[..., self.output_dim:]
        # resampling
        std = log_var.exp().pow(0.5)
        dist = torch.distributions.Normal(mean, std)
        Zt = dist.rsample()
        # Zt = self.reparameterize(mean, log_var)
        Yt = self.residualMLP(Zt)
        return mean, log_var, Zt, Yt

class MaskTransformer(nn.Module):
    def __init__(self, num_joints, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        self.max_seq_len = 1024
        self.num_joints = num_joints
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.use_pos_enc = True
        self.outputs_per_step = 4
        self.mask_ratio_min = 0.7
        print("slidding window!!!!!!",self.opt.pre_lens)

        print("-----------------使用自回归-------------------")
        # norm_first = False
        # seqTransDecoderLayer = nn.TransformerDecoderLayer(
        #     d_model=self.latent_dim,
        #     nhead=num_heads,
        #     dim_feedforward=ff_size,
        #     dropout=dropout,
        #     activation='gelu',
        #     # norm_first,
        # )
        # self.seqTransDecoder = nn.TransformerDecoder(
        #     decoder_layer=seqTransDecoderLayer,
        #     num_layers=num_layers,
        #     norm=nn.LayerNorm(self.latent_dim) if norm_first else None,
        # )
        self.norm = nn.LayerNorm(self.latent_dim)
        self.norm_first = nn.LayerNorm(self.latent_dim)
        self.mask_ratio_generator = stats.truncnorm((self.mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.output_process = OutputProcess(out_feats=self.num_joints * self.outputs_per_step,latent_dim=self.latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=ff_size,
                                                        dropout=dropout,
                                                        activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=num_layers,
                                                    norm=self.norm)
        # ================================================
        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        # self.input_process = InputProcess(self.num_joints, self.latent_dim)
        self.input_process = Prenet(self.num_joints * self.outputs_per_step, self.latent_dim * 4 ,self.latent_dim, 0.5)
        print("input_process``````````` ",self.num_joints, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        # self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))
        # self.stop_linear = Linear(self.latent_dim, self.outputs_per_step, w_init='sigmoid')
        self.stop_linear = Linear(self.num_joints * self.outputs_per_step, self.outputs_per_step, w_init='sigmoid')
        
        self.postconvnet = PostConvNet(input_dims=self.num_joints, outputs_per_step=self.outputs_per_step, num_hidden=latent_dim)
        self.latentsampling = LatentSampling(input_dim=self.num_joints * self.outputs_per_step, output_dim=self.num_joints * self.outputs_per_step,max_lens=196//self.outputs_per_step )

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        # self.mask_id = opt.num_tokens
        self.pad_id = 0
        print(225,"~~~~~~~~~~~",self.pad_id)
        # self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)

        # self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule
        if self.use_pos_enc:
            print("~~~~~~~~~~~~~~使用位置编码~~~~~~~~~~~~~~~~~")


    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def sparse_attention_mask(self,xseq, k, n):
        xseq_len = xseq.shape[0]
        # 创建一个全为True的mask，表示所有位置默认被掩盖
        mask = torch.ones((xseq_len, xseq_len), dtype=torch.bool, device=xseq.device)
        
        # 将主对角线及以下n条对角线设置为False，表示不被掩盖
        for diag_offset in range(-n, 1):
            mask.diagonal(diag_offset).fill_(False)
        mask[:, 0].fill_(False)
        return mask
    def generate_autoregressive_mask(self, seq_len, device=None):
        """
        生成用于Transformer的自回归掩码。
        
        :param seq_len: 序列长度
        :param device: 设备（如 'cpu' 或 'cuda'）
        :return: 自回归掩码 (seq_len, seq_len)
        """
        # 创建一个全为False的mask
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        
        # 将上三角部分（包括对角线）设置为True
        mask = ~mask
        
        if device is not None:
            mask = mask.to(device)
        
        return mask
    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(196 // self.outputs_per_step)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # import pdb;pdb.set_trace()
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask
    
    def trans_forward(self, motions, cond, padding_mask, force_mask=False,skip_cond = False,mask = None):
        # import pdb;pdb.set_trace()
        '''
        :param motions: (seqlen-1, b, latent_dim)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b,  seqlen, dim)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)
        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)
        # print(motion_ids.shape)

        t = len(motions)
        if t:
            mask = torch.cat([torch.zeros_like(mask[:, 0:1]), mask], dim=1).bool() #(b, seqlen+1)

        xseq = torch.cat([cond, motions], dim=0)  if t else cond#(seqlen, b, latent_dim)


        if self.use_pos_enc:
            xseq = self.position_enc(xseq)

        # tgt_mask = self.sparse_attention_mask(xseq, 1, 100)
        # xseq = self.norm_first(xseq)
        output = self.seqTransEncoder(xseq,src_key_padding_mask=mask)[1:,...]
        output = self.output_process(output)
        logits = output.permute(1,0,2)

        return logits

    def forward(self, motions, y, m_lens):
        # import pdb; pdb.set_trace()
        '''
        :param motions: (b, n,joints )
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''
        # import pdb; pdb.set_trace() 
        bs, ntokens,joints = motions.shape[0], motions.shape[1],motions.shape[2]
        device = motions.device
        labels = motions.clone()

        g_motions = motions.view(bs,ntokens//self.outputs_per_step,joints * self.outputs_per_step)
        # Positions that are PADDED are ALL FALSE
        non_pad_mask = ~lengths_to_mask(m_lens//self.outputs_per_step, ntokens//self.outputs_per_step) #(b, n)
        # ids = torch.where(non_pad_mask, ids, self.pad_id)

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")


        '''
        Prepare mask
        '''
        x_ids = g_motions.clone()

        # labels = motions.clone()


        # x_ids = x_ids[:, :-1]
        #====================                ===================
        non_pad_mask = non_pad_mask[:,:x_ids.shape[1]]

        motions_ = self.input_process(x_ids)# (b, seqlen-1, num_joints) -> (seqlen-1, b, latent_dim)
        # add mask
        orders = self.sample_orders(bsz=bs)
        mask = self.random_masking(motions_.permute(1,0,2), orders)        
        # import pdb;pdb.set_trace()
        logits = self.trans_forward(motions_, cond_vector, non_pad_mask, force_mask,mask=mask)
        # 111
        # log = self.out_norm(logits)
        # motion_out = self.motion_linear(logits)
        mean, log_var, Zt, motion_out = self.latentsampling(logits)
        # motion_out -> y'
        # Zt -> z_t
        # out -> y''
        postnet_input = motion_out.transpose(1, 2)
        
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)
        '''
        mean 64 49 1052
        val 64 49 1052
        motion_out 64 49 1052
        out 64 49 1052
        '''

        stop_tokens = self.stop_linear(logits).view(logits.size(0), -1)
        # import pdb;pdb.set_trace()
        # mean = mean.reshape(bs, ntokens,joints)
        # log_val = log_val.reshape(bs, ntokens,joints)
        # motion_out = motion_out.reshape(bs, ntokens,joints)
        labels = labels.reshape(bs,ntokens // self.outputs_per_step,joints * self.outputs_per_step)
        regre_loss, bce_loss, kl_loss, Flux_loss = cal_new_loss(mean, log_var,motion_out, out, labels, stop_tokens,m_lens)
        out = out.reshape(bs, ntokens,joints)

        return regre_loss, bce_loss, kl_loss, Flux_loss, out, stop_tokens

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                skip_cond,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            logits= self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, skip_cond=skip_cond)
            return logits

        logits= self.trans_forward(motion_ids, cond_vector, padding_mask, skip_cond=skip_cond)
        if cond_scale == 1:
            return logits

        # aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, skip_cond=skip_cond)

        # scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        # return scaled_logits
        return logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):

        

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")
        
        num_dims = len(cond_vector.shape)
        
        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'

        padding_mask = ~lengths_to_mask(m_lens//self.outputs_per_step, seq_len//self.outputs_per_step)
        # Start from all tokens being masked

        # import pdb; pdb.set_trace()
        # out = torch.full(size = (batch_size, 1), fill_value = self.pad_id,device=device) #(batch , 1)
        out = []

        for t in range(seq_len//self.outputs_per_step):  # 使用t代替lens以避免与len()函数混淆
            x = out
            # 确保至少有一个元素在out中，以避免负索引问题
            if t == 0 or len(out) == 0:
                padding_mask0 = None # 使用当前时间步t+1（因为索引从0开始）
            else:
                padding_mask0 = padding_mask[:, :t]
                x = self.input_process(x)
                # x = x.permute(1,0,2)


            # logits = self.forward_with_cond_scale(x, cond_vector=cond_vector,
            #                                         padding_mask=padding_mask0,
            #                                         cond_scale=cond_scale,
            #                                         force_mask=force_mask,
            #                                         skip_cond=True)

            logits = self.trans_forward(motions=x, 
                                        cond = cond_vector, 
                                        padding_mask=padding_mask0)
            # 11111
            mean, log_val, Zt, motion_out = self.latentsampling(logits)
            postnet_input = motion_out.transpose(1, 2)
            output = self.postconvnet(postnet_input)
            
            output = postnet_input + output
            output = output.transpose(1, 2)

            logit = output[:,-1,:].unsqueeze(1)

            pred_id = logit
            if len(out):
                out = torch.cat((out,pred_id), dim = 1)
            else:
                out = pred_id
        # import pdb;pdb.set_trace()
        out = out
        # print(out.shape)

        if num_dims == 2:
            out = out.squeeze(0)
        # ids = torch.where(padding_mask, -1, out)
        # print("Final", ids.max(), ids.min())
        # mean, log_val, Zt, motion_out = self.latentsampling(out)
        # motion_out = self.motion_linear(out)
        # postnet_input = motion_out.transpose(1, 2)
        # output = self.postconvnet(postnet_input)
        # output = postnet_input + output
        # output = output.transpose(1, 2)

        stop_tokens = self.stop_linear(logits).view(out.size(0), -1)
        output_ = out.reshape(batch_size,seq_len,self.num_joints)
        return output_
    
    @torch.no_grad()
    @eval_decorator
    def long_generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 mids=torch.tensor([]),
                 force_mask=False
                 ):
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        # import pdb; pdb.set_trace()
        # mid.shape = (seq_len)

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")
        
        num_dims = len(cond_vector.shape)
        
        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.pad_id)
        # 全部用pad_id填充  
        # ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        starting_temperature = temperature
        # import pdb; pdb.set_trace()
        # out = torch.full(size = (batch_size, 1), fill_value = self.pad_id,device=device) #(batch , 1)
        out = mids.unsqueeze(0)
        mids_len = len(mids)
        # print(seq_len)
        for t in range(seq_len):  # 使用t代替lens以避免与len()函数混淆
            x = out
            # 确保至少有一个元素在out中，以避免负索引问题
            if t == 0 or len(out) == 0:
                # padding_mask0 = None # 使用当前时间步t+1（因为索引从0开始）
                padding_mask0 = torch.zeros_like(padding_mask[:, 0:mids_len])
            else:
                padding_mask0 = padding_mask[:, :t]
                padding_mask0 = torch.cat([torch.zeros_like(padding_mask[:, 0:mids_len]), padding_mask0], dim=1) #(b, seqlen+mids_len)
            logits = self.forward_with_cond_scale(x, cond_vector=cond_vector,
                                                    padding_mask=padding_mask0,
                                                    cond_scale=cond_scale,
                                                    force_mask=force_mask,
                                                    skip_cond=True)
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob tokenn
            logit = logits[:,-1,:].unsqueeze(1)
            filtered_logit = top_k(logit, topk_filter_thres, dim=-1)
            temperature = starting_temperature

            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_id = gumbel_sample(filtered_logit, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                prob = F.softmax(filtered_logit / temperature, dim=-1)  # (b, seqlen, ntoken)

                pred_id = Categorical(prob).sample()  # (b, 1)
            if len(out):
                out = torch.cat((out,pred_id), dim = 1)
            else:
                print("Errror!!!!!!!!!!!!!!!!!!!!!!!!!")
                out = pred_id
        out = out[:,mids_len:]
        # print(out.shape)

        if num_dims == 2:
            out = out.squeeze(0)
        ids = torch.where(padding_mask, -1, out)
        # print("Final", ids.max(), ids.min())
        return ids


