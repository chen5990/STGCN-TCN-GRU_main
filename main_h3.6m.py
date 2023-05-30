from __future__ import print_function, absolute_import, division
import yaml
import os
import torch
import torch.nn as nn
from torch import optim
import time
import numpy as np
from torch.utils.data.dataloader import DataLoader
import data_utils
import model_4GRU_STGCN_TCN
import eval_loss
import logging
import sys

model_config = 'STGCN_TCN_GRU_3.6m'

log_dir = './train_logs/'
log_path = log_dir + model_config + '.txt'
formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter, datefmt='%Y-%d-%m %H:%M:%S',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
#ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

node_num = config['node_num']
input_n = config['input_n']
output_n = config['output_n']
input_size = config['in_features']
hidden_size = config['hidden_size']
output_size = config['out_features']
lr = config['learning_rate']
batch_size = config['batch_size']

base_dataset_path = './dataset/H3.6M_dataset/dataset_train'
base_save_path = './train_model/STGCN_TCN_GRU_h3.6m'
actions = ["Walking", "Eating","Smoking", "Discussion", "Directions",
           "Greeting", "Phoning", "Posing", "Purchases", "Sitting",
           "SittingDown", "Photo", "Waiting", "WalkDog",
           "WalkTogether"]
#"Walking", "Eating", 

# 25node
chain = [[1], [132.95, 442.89, 454.21, 162.77, 75], [132.95, 442.89, 454.21, 162.77, 75],
         [233.58, 257.08, 121.13, 115], [257.08, 151.03, 278.88, 251.73, 100],
         [257.08, 151.03, 278.88, 251.73, 100]]

for x in chain:
    s = sum(x)
    if s == 0:
        continue
    for i in range(len(x)):
        x[i] = (i + 1) * sum(x[i:]) / s  # ?

chain = [item for sublist in chain for item in sublist]
nodes_weight = torch.tensor(chain, device='cuda')
nodes_weight = nodes_weight.unsqueeze(1)
nodes_frame_weight = nodes_weight.expand(25, output_n)

# frame_weight = torch.tensor([3,2,1.5,1.5,1,0.5,0.2,0.2,0.05,0.05])
frame_weight = torch.tensor([3, 2, 1.5, 1.5, 1, 0.5, 0.2, 0.2, 0.1, 0.1,
                             0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02,
                             0.02, 0.02, 0.02, 0.02], device='cuda')

def caculate_loss(x, y, action):
    #print('action', action)
    if action == "Walking" or action == "Eating" or action == "Smoking":
        return torch.mean(torch.abs(x - y))
    else:
        return torch.mean(torch.norm((x - y) * frame_weight * nodes_frame_weight, 2, 1))
    
        
for action in actions:

    logging.info(action)
    min_loss = 1000
    decline_cnt = 0
    best_epoch = -1
    best_loss = [0.0] * 25
    train_epoch = 100
    print('train_epoch: ', train_epoch)

    model_x = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)
    model_y = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)
    model_z = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)
    model_v = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)

    mse = nn.MSELoss(reduction='mean')  # 均方损失函数
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model_v.parameters()) / 1000000.0))

    save_path = os.path.join(base_save_path, action)
    save_path = save_path.replace("\\", "/")
    if os.path.exists(save_path):
        os.rmdir(save_path)
    os.makedirs(save_path)

    train_save_path = os.path.join(base_dataset_path, action + '.npy')
    train_save_path = train_save_path.replace("\\", "/")
    train_dataset = np.load(train_save_path, allow_pickle=True)
    train_data = data_utils.H36Datasets(train_dataset, input_n, output_n, is_train=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    base_path = save_path


    optimizer_x = optim.Adam(model_x.parameters(), lr)
    optimizer_y = optim.Adam(model_y.parameters(), lr)
    optimizer_z = optim.Adam(model_z.parameters(), lr)
    optimizer_v = optim.Adam(model_v.parameters(), lr)

    for epoch in range(train_epoch): # config['train_epoches']
        start_time = time.time()
        for i, data in enumerate(train_loader):
            # print("i:",i)
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()
            optimizer_z.zero_grad()
            optimizer_v.zero_grad()

            in_shots, out_shot = data

            # in_shots = data_utils.add_frame(in_shots, use_gpu=True)

            input_angle = in_shots[:, 1:, :, :3]  # 取1-最后？0-2 角度
            input_velocity = in_shots[:, 1:, :, 3].permute(0, 2, 1)
            target_angle = out_shot[:, :, :, :3]
            target_velocity = out_shot[:, :, :, 3]
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
            input_3d_data = in_shots[:, :, :, 4:]
            target_3d_data = out_shot[:, :, :, 4:]

            loss_v = 0
            loss_x = 0
            loss_y = 0
            loss_z = 0
            loss_rec = 0

            output_v = model_v(input_velocity, hidden_size)
            output_v = output_v.view(target_velocity.shape[0], target_velocity.shape[2], output_size)
            target_velocity_loss = target_velocity.permute(0, 2, 1)
            # loss_v += torch.mean(torch.abs(output_v - target_velocity_loss))
            loss_v += caculate_loss(output_v, target_velocity_loss, action)

            output_x = model_x(input_angle_x, hidden_size)
            output_x = output_x.view(target_angle_x.shape[0], target_angle_x.shape[2], output_size)
            target_angle_x_loss = target_angle_x.permute(0, 2, 1)
            # loss_x += torch.mean(torch.abs(output_x - target_angle_x_loss))
            loss_x += caculate_loss(output_x, target_angle_x_loss, action)

            output_y = model_y(input_angle_y, hidden_size)
            output_y = output_y.view(target_angle_y.shape[0], target_angle_y.shape[2], output_size)
            target_angle_y_loss = target_angle_y.permute(0, 2, 1)
            # loss_y += torch.mean(torch.abs(output_y - target_angle_y_loss))
            loss_y += caculate_loss(output_y, target_angle_y_loss, action)

            output_z = model_z(input_angle_z, hidden_size)
            output_z = output_z.view(target_angle_z.shape[0], target_angle_z.shape[2], output_size)
            target_angle_z_loss = target_angle_z.permute(0, 2, 1)
            # loss_z += torch.mean(torch.abs(output_z - target_angle_z_loss))
            loss_z += caculate_loss(output_z, target_angle_z_loss, action)

            total_loss = loss_v + loss_x + loss_y + loss_z

            total_loss.backward()
            nn.utils.clip_grad_norm_(model_x.parameters(), 5.0)
            nn.utils.clip_grad_norm_(model_y.parameters(), 5.0)
            nn.utils.clip_grad_norm_(model_z.parameters(), 5.0)
            nn.utils.clip_grad_norm_(model_v.parameters(), 5.0)

            optimizer_x.step()
            optimizer_y.step()
            optimizer_z.step()
            optimizer_v.step()
            if i % 10 == 0:
                print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
        now_loss = eval_loss.eval_loss(model_x, model_y, model_z, model_v, action, input_n, output_n, batch_size,
                               hidden_size)
        avg_now_loss = sum(now_loss) / len(now_loss)
        print('epoch', epoch)
        print('now_loss', now_loss)
        print('best_loss', best_loss)
        if min_loss > avg_now_loss:
            min_loss = avg_now_loss
            best_loss = now_loss
            best_epoch = epoch
            decline_cnt = 0
            torch.save(model_x, os.path.join(save_path, 'best_generator_x_4GRU.pkl'))
            torch.save(model_y, os.path.join(save_path, 'best_generator_y_4GRU.pkl'))
            torch.save(model_z, os.path.join(save_path, 'best_generator_z_4GRU.pkl'))
            torch.save(model_v, os.path.join(save_path, 'best_generator_v_4GRU.pkl'))
        else:
            decline_cnt += 1
        if decline_cnt >= 10:
            break
    logging.info('best_epoch: ' + str(best_epoch))
    logging.info(best_loss)
    # logging.info('80ms:\n' + str(best_loss[1]))
    # logging.info('160ms:\n' + str(best_loss[3]))
    # logging.info('320ms:\n' + str(best_loss[7]))
    # logging.info('400ms:\n' + str(best_loss[9]))
    # logging.info('560ms:\n' + str(best_loss[13]))
    # logging.info('720ms:\n' + str(best_loss[17]))
    # logging.info('1000ms:\n' + str(best_loss[24]))
