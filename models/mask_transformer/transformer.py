import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
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


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

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
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        self.max_seq_len = 1024
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.use_pos_enc = True
        print("slidding window!!!!!!",self.opt.pre_lens)

        print("11使用自回归-------------------")
        norm_first = False
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            # norm_first,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            decoder_layer=seqTransDecoderLayer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.latent_dim) if norm_first else None,
        )
        # ================================================
        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)



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


        _num_tokens = opt.num_tokens +1  # two dummy tokens, one for masking, one for padding
        # self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens
        print(225,"~~~~~~~~~~~",self.pad_id)
        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

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

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

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

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False,skip_cond = False):
        # import pdb;pdb.set_trace()
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)
        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)
        # print(motion_ids.shape)
        # import pdb; pdb.set_trace()
        if len(motion_ids):
            x = self.token_emb(motion_ids)# (b, seqlen-1, d) -> (seqlen-1, b, latent_dim)
            x = self.input_process(x)
 
            xseq = torch.cat([cond, x], dim=0) #(seqlen, b, latent_dim)

            padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1) #(b, seqlen+1)

        else:
            xseq = cond #(1, b, latent_dim)

        if self.use_pos_enc:
            xseq = self.position_enc(xseq)


        tgt_mask = self.sparse_attention_mask(xseq, 1, self.opt.pre_lens)
        
        output = self.seqTransDecoder(tgt = xseq, 
                                        memory = xseq,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask = padding_mask,
                                        memory_key_padding_mask = padding_mask,
                                        memory_mask = tgt_mask,
                                        ) #( seqlen,b, latent_dim)

        logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens):
        # import pdb; pdb.set_trace()
        '''
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''
        # import pdb; pdb.set_trace() 
        bs, ntokens = ids.shape
        device = ids.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

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
        # rand_time = uniform((bs,), device=device)
        # rand_mask_probs = self.noise_schedule(rand_time)
        # num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        # batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # # Positions to be MASKED are ALL TRUE
        # mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # # Positions to be MASKED must also be NON-PADDED
        # mask &= non_pad_mask

        # Note this is our training target, not input
        # labels = torch.where(mask, ids, self.mask_id)
        x_ids = ids.clone()
        # tmp_ids =  torch.where(non_pad_mask, ids, self.mask_id)
        # _, labels = self.pad_y_eos(tmp_ids)
        # x_ids = x_ids[:, :-1]
        labels = ids.clone()
        # labels = x_ids
        # print(labels.shape)
        x_ids = x_ids[:, :-1]
        #====================                ===================
        non_pad_mask = non_pad_mask[:,:x_ids.shape[1]]
        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)
        # print(logits.shape)
        # ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.pad_id)


        return ce_loss, pred_id, acc

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
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, skip_cond=skip_cond)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask, skip_cond=skip_cond)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, skip_cond=skip_cond)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

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
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        # import pdb; pdb.set_trace()
        

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
        out = []
        # print(seq_len)
        # import pdb;pdb.set_trace()
        for t in range(seq_len):  # 使用t代替lens以避免与len()函数混淆
            x = out
            # 确保至少有一个元素在out中，以避免负索引问题
            if t == 0 or len(out) == 0:
                padding_mask0 = None # 使用当前时间步t+1（因为索引从0开始）
            else:
                padding_mask0 = padding_mask[:, :t]

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
                out = pred_id
        out = out
        # print(out.shape)

        if num_dims == 2:
            out = out.squeeze(0)
        ids = torch.where(padding_mask, -1, out)
        # print("Final", ids.max(), ids.min())
        return ids
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

    # @torch.no_grad()
    # @eval_decorator
    # def edit(self,
    #          conds,
    #          tokens,
    #          m_lens,
    #          timesteps: int,
    #          cond_scale: int,
    #          temperature=1,
    #          topk_filter_thres=0.9,
    #          gsample=False,
    #          force_mask=False,
    #          edit_mask=None,
    #          padding_mask=None,
    #          ):

    #     assert edit_mask.shape == tokens.shape if edit_mask is not None else True
    #     device = next(self.parameters()).device
    #     seq_len = tokens.shape[1]

    #     if self.cond_mode == 'text':
    #         with torch.no_grad():
    #             cond_vector = self.encode_text(conds)
    #     elif self.cond_mode == 'action':
    #         cond_vector = self.enc_action(conds).to(device)
    #     elif self.cond_mode == 'uncond':
    #         cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
    #     else:
    #         raise NotImplementedError("Unsupported condition mode!!!")

    #     if padding_mask == None:
    #         padding_mask = ~lengths_to_mask(m_lens, seq_len)

    #     # Start from all tokens being masked
    #     if edit_mask == None:
    #         mask_free = True
    #         ids = torch.where(padding_mask, self.pad_id, tokens)
    #         edit_mask = torch.ones_like(padding_mask)
    #         edit_mask = edit_mask & ~padding_mask
    #         edit_len = edit_mask.sum(dim=-1)
    #         scores = torch.where(edit_mask, 0., 1e5)
    #     else:
    #         mask_free = False
    #         edit_mask = edit_mask & ~padding_mask
    #         edit_len = edit_mask.sum(dim=-1)
    #         ids = torch.where(edit_mask, self.mask_id, tokens)
    #         scores = torch.where(edit_mask, 0., 1e5)
    #     starting_temperature = temperature

    #     for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
    #         # 0 < timestep < 1
    #         rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)  # Tensor

    #         '''
    #         Maskout, and cope with variable length
    #         '''
    #         # fix: the ratio regarding lengths, instead of seq_len
    #         num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

    #         # select num_token_masked tokens with lowest scores to be masked
    #         sorted_indices = scores.argsort(
    #             dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
    #         ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
    #         is_mask = (ranks < num_token_masked.unsqueeze(-1))
    #         # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
    #         ids = torch.where(is_mask, self.mask_id, ids)

    #         '''
    #         Preparing input
    #         '''
    #         # (b, num_token, seqlen)
    #         logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
    #                                               padding_mask=padding_mask,
    #                                               cond_scale=cond_scale,
    #                                               force_mask=force_mask)

    #         logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
    #         # print(logits.shape, self.opt.num_tokens)
    #         # clean low prob token
    #         filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

    #         '''
    #         Update ids
    #         '''
    #         # if force_mask:
    #         temperature = starting_temperature
    #         # else:
    #         # temperature = starting_temperature * (steps_until_x0 / timesteps)
    #         # temperature = max(temperature, 1e-4)
    #         # print(filtered_logits.shape)
    #         # temperature is annealed, gradually reducing temperature as well as randomness
    #         if gsample:  # use gumbel_softmax sampling
    #             # print("1111")
    #             pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
    #         else:  # use multinomial sampling
    #             # print("2222")
    #             probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
    #             # print(temperature, starting_temperature, steps_until_x0, timesteps)
    #             # print(probs / temperature)
    #             pred_ids = Categorical(probs).sample()  # (b, seqlen)

    #         # print(pred_ids.max(), pred_ids.min())
    #         # if pred_ids.
    #         ids = torch.where(is_mask, pred_ids, ids)

    #         '''
    #         Updating scores
    #         '''
    #         probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
    #         scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
    #         scores = scores.squeeze(-1)  # (b, seqlen)

    #         # We do not want to re-mask the previously kept tokens, or pad tokens
    #         scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

    #     ids = torch.where(padding_mask, -1, ids)
    #     # print("Final", ids.max(), ids.min())
    #     return ids

    # @torch.no_grad()
    # @eval_decorator
    # def edit_beta(self,
    #               conds,
    #               conds_og,
    #               tokens,
    #               m_lens,
    #               cond_scale: int,
    #               force_mask=False,
    #               ):

    #     device = next(self.parameters()).device
    #     seq_len = tokens.shape[1]

    #     if self.cond_mode == 'text':
    #         with torch.no_grad():
    #             cond_vector = self.encode_text(conds)
    #             if conds_og is not None:
    #                 cond_vector_og = self.encode_text(conds_og)
    #             else:
    #                 cond_vector_og = None
    #     elif self.cond_mode == 'action':
    #         cond_vector = self.enc_action(conds).to(device)
    #         if conds_og is not None:
    #             cond_vector_og = self.enc_action(conds_og).to(device)
    #         else:
    #             cond_vector_og = None
    #     else:
    #         raise NotImplementedError("Unsupported condition mode!!!")

    #     padding_mask = ~lengths_to_mask(m_lens, seq_len)

    #     # Start from all tokens being masked
    #     ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

    #     '''
    #     Preparing input
    #     '''
    #     # (b, num_token, seqlen)
    #     logits = self.forward_with_cond_scale(ids,
    #                                           cond_vector=cond_vector,
    #                                           cond_vector_neg=cond_vector_og,
    #                                           padding_mask=padding_mask,
    #                                           cond_scale=cond_scale,
    #                                           force_mask=force_mask)

    #     logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

    #     '''
    #     Updating scores
    #     '''
    #     probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
    #     tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
    #     og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
    #     og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

    #     return og_tokens_scores


