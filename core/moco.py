# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from .mixin import TrainStepMixin
import copy

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MoCo(nn.Module, TrainStepMixin):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722

    The code is mainly ported from the official repo:
    https://github.com/facebookresearch/moco

    """

    def __init__(self,
                 i3d_online,
                 out_channels: int,
                 queue_size: int = 1024,
                 momentum: float = 0.999,
                 temperature: float = 0.07):
        super(MoCo, self).__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        self.register_buffer("queue", torch.randn(out_channels, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.key_encoder = copy.deepcopy(i3d_online)
        self.fc = nn.Linear(1024,128)
        self.key_fc = nn.Linear(1024,28)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self, backbone):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(backbone.parameters(),
                                    self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc.parameters(),
                                    self.key_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, q, k_in, backbone):

        q = self.fc(q)
        q = nn.functional.normalize(q, dim=1)

        # # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder(backbone=backbone)
            i3d_tgt_tubelet = self.key_encoder(k_in)
            i3d_tgt_tubelet = i3d_tgt_tubelet.squeeze(3).squeeze(3).squeeze(2)
            k = self.key_encoder(i3d_tgt_tubelet)
            k = self.key_fc(k)
            k = nn.functional.normalize(k, dim=1)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        nce_loss = nn.functional.cross_entropy(logits, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return dict(nce_loss=nce_loss)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
