from __future__ import print_function, absolute_import, division
import os
import sys
import CMU_motion_3d
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
import data_utils
import eval_loss
import model_4GRU_STGCN_TCN
import logging


model_config = 'STGCN_TCN_GRU_3dpw'

log_dir = './train_logs/'
log_path = log_dir + model_config + '.txt'
formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter, datefmt='%Y-%d-%m %H:%M:%S',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

node_num = 25
input_n = 10
output_n = 25
base_dataset_path = './dataset/3dpw_dataset'
base_save_path = './train_model/STGCN_TCN_GRU_3dpw'
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)
input_size = 9
hidden_size = 128
output_size = 25
lr = 0.002
batch_size = 256
def caculate_loss(x, y):
    return torch.mean(torch.abs(x - y))


min_loss = 1000
decline_cnt = 0
best_epoch = -1
best_loss = [0.0] * 25

save_path = base_save_path
save_path = save_path.replace("\\", "/")
if os.path.exists(save_path):
    os.rmdir(save_path)
os.makedirs(save_path)


train_data = data_utils.DPWDatasets(input_n, output_n,split=0)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

model_x = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)
model_y = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)
model_z = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)
model_v = model_4GRU_STGCN_TCN.Generator(input_size, hidden_size, output_size, node_num, batch_size).to(device)

print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model_v.parameters()) / 1000000.0))

optimizer_x = optim.Adam(model_x.parameters(), lr)
optimizer_y = optim.Adam(model_y.parameters(), lr)
optimizer_z = optim.Adam(model_z.parameters(), lr)
optimizer_v = optim.Adam(model_v.parameters(), lr)

for epoch in range(80):  # config['train_epoches']
    for i, data in enumerate(train_loader):
        # print("i:",i)
        optimizer_x.zero_grad()
        optimizer_y.zero_grad()
        optimizer_z.zero_grad()
        optimizer_v.zero_grad()

        data = data.cuda()
        in_shots, out_shot = data[:, 0: input_n], data[:, input_n:]

        # 扩充至25
        in_shots_zero = np.zeros((batch_size, input_n, 3, 7))
        out_shot_zero = np.zeros((batch_size, output_n, 3, 7))

        in_shots_zero = torch.tensor(in_shots_zero, device='cuda')
        in_shots = torch.tensor(in_shots, device='cuda')
        in_shots = torch.cat([in_shots, in_shots_zero], dim=2)

        out_shot_zero = torch.tensor(out_shot_zero, device='cuda')
        out_shot = torch.tensor(out_shot, device='cuda')
        out_shot = torch.cat([out_shot, out_shot_zero], dim=2)

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

        loss_rec = 0

        output_v = model_v(input_velocity, hidden_size)
        output_v = output_v.view(target_velocity.shape[0], target_velocity.shape[2], output_size)
        target_velocity_loss = target_velocity.permute(0, 2, 1)
        loss_v = caculate_loss(output_v, target_velocity_loss)

        output_x = model_x(input_angle_x, hidden_size)
        output_x = output_x.view(target_angle_x.shape[0], target_angle_x.shape[2], output_size)
        target_angle_x_loss = target_angle_x.permute(0, 2, 1)
        loss_x = caculate_loss(output_x, target_angle_x_loss)

        output_y = model_y(input_angle_y, hidden_size)
        output_y = output_y.view(target_angle_y.shape[0], target_angle_y.shape[2], output_size)
        target_angle_y_loss = target_angle_y.permute(0, 2, 1)
        loss_y = caculate_loss(output_y, target_angle_y_loss)

        output_z = model_z(input_angle_z, hidden_size)
        output_z = output_z.view(target_angle_z.shape[0], target_angle_z.shape[2], output_size)
        target_angle_z_loss = target_angle_z.permute(0, 2, 1)
        loss_z = caculate_loss(output_z, target_angle_z_loss)

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

    now_loss = eval_loss.eval_3dpw_loss(model_x, model_y, model_z, model_v, input_n, output_n, batch_size, hidden_size)
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
logging.info('80ms:\n' + str(best_loss[1]))
logging.info('160ms:\n' + str(best_loss[3]))
logging.info('320ms:\n' + str(best_loss[7]))
logging.info('400ms:\n' + str(best_loss[9]))
logging.info('560ms:\n' + str(best_loss[13]))
logging.info('720ms:\n' + str(best_loss[17]))
logging.info('1000ms:\n' + str(best_loss[24]))