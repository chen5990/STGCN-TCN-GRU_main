from __future__ import print_function, absolute_import, division
import yaml
import os
import torch
import torch.nn as nn
from torch import optim
import time
import numpy as np
from torch.utils.data.dataloader import DataLoader

import CMU_motion_3d
import data_utils

import space_angle_velocity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

node_num = 25
input_n = 10
output_n = 25
input_size = 9
hidden_size = 128
output_size = 25
lr = 0.0005
batch_size = 16


actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
           "washwindow"]

for action in actions:
    eval_data = CMU_motion_3d.CMU_Motion3D(input_n, output_n, 1, action)
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model_path = "./model/cmu"
    model_path = os.path.join(model_path, action)
    model_x = torch.load(os.path.join(model_path, 'best_generator_x_4GRU.pkl')).to(device)
    model_y = torch.load(os.path.join(model_path, 'best_generator_y_4GRU.pkl')).to(device)
    model_z = torch.load(os.path.join(model_path, 'best_generator_z_4GRU.pkl')).to(device)
    model_v = torch.load(os.path.join(model_path, 'best_generator_v_4GRU.pkl')).to(device)

    batch_cnt = 0

    sum_total_loss = [0.0] * output_n

    for i, data in enumerate(eval_loader):
        batch_cnt += 1
        data = data.cuda()
        input_dataset = data[:, 0:input_n]
        output_dataset = data[:, input_n:]
        # print("input_data", input_dataset.shape)
        # print("out_data", output_dataset.shape)
        # input_dataset = data_utils.add_frame(input_dataset, use_gpu=True)

        input_angle = input_dataset[:, 1:, :, :3]
        input_velocity = input_dataset[:, 1:, :, 3].permute(0, 2, 1)

        target_angle = output_dataset[:, :, :, :3]
        target_velocity = output_dataset[:, :, :, 3]
        # read velocity
        input_velocity = input_velocity.float()
        target_velocity = target_velocity.float()
        # read angle_x
        input_angle_x = input_angle[:, :, :, 0].permute(0, 2, 1).float()
        target_angle_x = target_angle[:, :, :, 0].float()
        # read angle_y
        input_angle_y = input_angle[:, :, :, 1].permute(0, 2, 1).float()
        target_angle_y = target_angle[:, :, :, 1].float()
        # read angle_z
        input_angle_z = input_angle[:, :, :, 2].permute(0, 2, 1).float()
        target_angle_z = target_angle[:, :, :, 2].float()
        # read 3D data
        input_3d_data = input_dataset[:, :, :, 4:]
        target_3d_data = output_dataset[:, :, :, 4:]

        output_v = model_v(input_velocity, hidden_size)
        output_v = output_v.view(target_velocity.shape[0], target_velocity.shape[2], output_n)

        output_x = model_x(input_angle_x, hidden_size)
        output_x = output_x.view(target_angle_x.shape[0], target_angle_x.shape[2], output_n)

        output_y = model_y(input_angle_y, hidden_size)
        output_y = output_y.view(target_angle_y.shape[0], target_angle_y.shape[2], output_n)

        output_z = model_z(input_angle_z, hidden_size)
        output_z = output_z.view(target_angle_z.shape[0], target_angle_z.shape[2], output_n)

        angle_x = output_x.permute(0, 2, 1)
        angle_y = output_y.permute(0, 2, 1)
        angle_z = output_z.permute(0, 2, 1)
        pred_v = output_v.permute(0, 2, 1)

        pred_angle_set = torch.stack((angle_x, angle_y, angle_z), 3)
        pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0], pred_angle_set.shape[1], -1, 3)

        # reconstruction_loss
        re_data = space_angle_velocity.reconstruction_motion(pred_angle_set, pred_v, input_3d_data[:, -1],
                                                             True)

        # action_loss = mpjpe_error(re_data, target_3d_data)

        frame_loss = data_utils.frame_mpjpe_error(re_data, target_3d_data)
        total_loss = [0.0] * output_n
        sum_loss = 0.0
        for j in range(len(total_loss)):
            sum_loss += frame_loss[j]
            total_loss[j] = sum_loss / (j + 1)
        for j in range(len(sum_total_loss)):
            sum_total_loss[j] += total_loss[j]

    for j in range(len(sum_total_loss)):
        sum_total_loss[j] = sum_total_loss[j] / batch_cnt
    print('action:', action)
    print('80ms:\n', sum_total_loss[1])
    print('160ms:\n', sum_total_loss[3])
    print('320ms:\n', sum_total_loss[7])
    print('400ms:\n', sum_total_loss[9])
    print('560ms:\n', sum_total_loss[13])
    print('720ms:\n', sum_total_loss[17])
    print('1000ms:\n', sum_total_loss[24])
