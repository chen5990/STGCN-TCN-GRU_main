import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import CMU_motion_3d
import data_utils
import space_angle_velocity


def eval_loss(model_x, model_y, model_z, model_v, action, input_n, output_n, batch_size, hidden_size):
    test_path = './dataset/H3.6M_dataset/dataset_valid'
    test_path = os.path.join(test_path, action + '.npy')
    test_path = test_path.replace("\\", "/")
    dataset = np.load(test_path, allow_pickle=True)

    batch_cnt = 0

    sum_total_loss = [0.0] * output_n

    for i in range(dataset.shape[0]):
        test_data = dataset[i]
        test_data = np.array(test_data)
        test_data = data_utils.LPDataset(test_data, input_n, output_n, True)
        # print("test_data", np.array(test_data).shape)
        train_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        for i, data in enumerate(train_loader):
            batch_cnt += 1
            # print("data", np.array(data).shape)
            input_dataset = data[0]
            output_dataset = data[1]
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

    return sum_total_loss


def stage_eval_loss(model_x, model_y, model_z, model_v, action, input_n, output_n, batch_size, hidden_size):
    dataset = np.load(os.path.join('dataset_test/', action + '.npy'), allow_pickle=True)

    batch_cnt = 0

    sum_total_loss = [0.0] * output_n

    for i in range(dataset.shape[0]):
        test_data = dataset[i]
        test_data = np.array(test_data)
        test_data = data_utils.LPDataset(test_data, input_n, output_n, True)
        # print("test_data", np.array(test_data).shape)
        train_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        for i, data in enumerate(train_loader):
            batch_cnt += 1
            # print("data", np.array(data).shape)
            input_dataset = data[0]
            output_dataset = data[1]

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

            output_v1, output_v2, output_v3, output_v = model_v(input_velocity, hidden_size)
            output_v = output_v.view(target_velocity.shape[0], target_velocity.shape[2], output_n)

            output_x1, output_x2, output_x3, output_x = model_x(input_angle_x, hidden_size)
            output_x = output_x.view(target_angle_x.shape[0], target_angle_x.shape[2], output_n)

            output_y1, output_y2, output_y3, output_y = model_y(input_angle_y, hidden_size)
            output_y = output_y.view(target_angle_y.shape[0], target_angle_y.shape[2], output_n)

            output_z1, output_z2, output_z3, output_z = model_z(input_angle_z, hidden_size)
            output_z = output_z.view(target_angle_z.shape[0], target_angle_z.shape[2], output_n)

            angle_x = output_x.permute(0, 2, 1)
            angle_y = output_y.permute(0, 2, 1)
            angle_z = output_z.permute(0, 2, 1)
            pred_v = output_v.permute(0, 2, 1)

            pred_angle_set = torch.stack((angle_x, angle_y, angle_z), 3)
            pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0], pred_angle_set.shape[1], -1, 3)

            # reconstruction_loss
            re_data = t0421space_angle_velocity.reconstruction_motion(pred_angle_set, pred_v, input_3d_data[:, -1],
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

    return sum_total_loss


def eval_merge_loss(model, action, input_n, output_n, batch_size):
    dataset = np.load(os.path.join('dataset_test/', action + '.npy'), allow_pickle=True)

    batch_cnt = 0

    sum_total_loss = [0.0] * output_n

    for i in range(dataset.shape[0]):
        test_data = dataset[i]
        test_data = np.array(test_data)
        test_data = data_utils.LPDataset(test_data, input_n, output_n, True)
        train_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        for i, data in enumerate(train_loader):
            batch_cnt += 1
            input_dataset = data[0]
            output_dataset = data[1]

            # input_dataset = data_utils.add_frame(input_dataset, use_gpu=True)

            input_angle_velocity = input_dataset[:, 1:, :, :4]
            target_angle_velocity = output_dataset[:, :, :, :4]
            input_3d = input_dataset[:, :, :, 4:]
            target_3d = output_dataset[:, :, :, 4:]

            output = model(input_angle_velocity)

            # reconstruction_loss
            re_data = space_angle_velocity.reconstruction_motion(output[:, :, :, :3], output[:, :, :, 3], input_3d[:, -1], True)

            frame_loss = data_utils.frame_mpjpe_error(re_data, target_3d)
            total_loss = [0.0] * output_n
            sum_loss = 0.0
            for j in range(len(total_loss)):
                sum_loss += frame_loss[j]
                total_loss[j] = sum_loss / (j + 1)
            for j in range(len(sum_total_loss)):
                sum_total_loss[j] += total_loss[j]

    for j in range(len(sum_total_loss)):
        sum_total_loss[j] = sum_total_loss[j] / batch_cnt

    return sum_total_loss


def eval_cmu_loss(model_x, model_y, model_z, model_v, action, input_n, output_n, batch_size, hidden_size):
    eval_data = CMU_motion_3d.CMU_Motion3D(input_n, output_n, 1, action)
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

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

    return sum_total_loss


def eval_3dpw_loss(model_x, model_y, model_z, model_v, input_n, output_n, batch_size, hidden_size):
    eval_data = data_utils.DPWDatasets(input_n, output_n, split=1)

    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    batch_cnt = 0

    sum_total_loss = [0.0] * output_n

    for i, data in enumerate(eval_loader):
        batch_cnt += 1
        data = data.cuda()
        input_dataset = data[:, 0:input_n]
        output_dataset = data[:, input_n:]
        
        # 扩充至25
        input_dataset_zero = np.zeros((batch_size, input_n, 3, 7))
        output_dataset_zero = np.zeros((batch_size, output_n, 3, 7))

        input_dataset_zero = torch.tensor(input_dataset_zero, device='cuda')
        input_dataset = torch.tensor(input_dataset, device='cuda')
        input_dataset = torch.cat([input_dataset, input_dataset_zero], dim=2)

        output_dataset_zero = torch.tensor(output_dataset_zero, device='cuda')
        output_dataset = torch.tensor(output_dataset, device='cuda')
        output_dataset = torch.cat([output_dataset, output_dataset_zero], dim=2)

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

    if batch_size != 0:
        for j in range(len(sum_total_loss)):
            sum_total_loss[j] = sum_total_loss[j] / batch_cnt

    return sum_total_loss
