import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from einops import rearrange

# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids==special_id)
    return mask

# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# classifier free guidance functions

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

# noise schedules

# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low

def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    import pdb; pdb.set_trace() 
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc

def create_stop_tokens(m_lens):
    # 获取 batch size 和 max sequence length
    batch_size = m_lens.size(0)
    max_sequence_length = m_lens.max().item()
    
    # 创建一个全零的张量
    mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.float32,device=m_lens.device)
    
    # 根据 m_lens 的值填充1
    for i, length in enumerate(m_lens):
        if length < max_sequence_length:
            mask[i, length:] = 1
    return mask
def kl_divergence(y, mean, log_var):
    # 将对数方差转换为方差
    var = log_var.exp()
    
    # 计算 KL 散度
    kl_loss = 0.5 * ((var + (mean - y)**2 - 1) - log_var)  # 
    
    # 对所有维度求和
    kl_loss = kl_loss.sum(dim=-1)  # 在最后一个维度上求和
    kl_loss = kl_loss.sum(dim=-2)  # 在倒数第二个维度上求和
    
    # 平均化损失
    kl_loss = kl_loss.mean()
    
    return kl_loss

def cal_new_loss(mean, log_var,motion_out, post_out, labels , stop_tokens ,m_lens):
    # import pdb;pdb.set_trace()
    kl_loss = 0.1 * kl_divergence(labels, mean, log_var)
    mel_lossl1 = F.smooth_l1_loss(motion_out, labels)
    mel_lossl2 = F.mse_loss(motion_out, labels)
    mel_loss = mel_lossl1 + mel_lossl2
    post_mel_lossl1 = F.smooth_l1_loss(post_out, labels)
    post_mel_lossl2 = F.mse_loss(post_out, labels)
    post_mel_loss = post_mel_lossl1 + post_mel_lossl2

    stop_label = create_stop_tokens(m_lens=m_lens).detach()
    stop_tokens = stop_tokens.to(dtype=torch.float32)
    bce_loss = nn.BCEWithLogitsLoss()(stop_tokens, stop_label)

    regre_loss = mel_loss + post_mel_loss

    Flux_loss = 0.5 * nn.L1Loss()(mean[:,1:,:], labels[:,:-1,:])
    # regre_loss = regre_loss + Flux_loss
    return regre_loss, bce_loss, kl_loss, Flux_loss

def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss