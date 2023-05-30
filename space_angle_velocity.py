import os
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
import data_utils


def loc_exchange(input):
    diff = input[:, 1:, :, :] - input[:, :-1, :, :]
    velocity = torch.norm(diff, p=2, dim=-1)
    angle = diff / velocity.unsqueeze(dim=-1).repeat(1, 1, 1, 3)
    angle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
    return torch.cat((angle, velocity.unsqueeze(dim=-1)), dim=-1)


def reconstruction_motion(angle, v, last_pose, use_gpu):
    b, f, n = v.shape
    if use_gpu:
        re_data = torch.zeros((b, f, n, 3)).cuda()
    else:
        re_data = torch.zeros((b, f, n, 3))
    for a in range(f):
        re_data[:, a] = last_pose + angle[:, a] * v[:, a].unsqueeze(dim=-1)
        last_pose = re_data[:, a]
    return re_data


def mpjpe(input, target):
    return torch.mean(torch.norm(input - target, 2, 1))
