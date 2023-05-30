import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset

import ang2joint
from space_angle_velocity import loc_exchange



class LPDataset(Dataset):  # Dataset是数据集位置，就是下面的path

    def __init__(self, data, input_n, output_n, is_train):
        super(LPDataset, self).__init__()
        self.data = data
        self.data = torch.from_numpy(self.data)
        self.data = self.data.reshape((-1, self.data.shape[1], 7))
        self.input_n = input_n
        self.output_n = output_n
        self.num = self.data.shape[0] - (input_n + output_n)
        self.is_train = is_train

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        a = item
        b = item + self.input_n
        c = item + self.input_n + self.output_n
        if self.is_train:
            first = torch.tensor(self.data[a: b], device='cuda')
            second = torch.tensor(self.data[b: c], device='cuda')
        else:
            first = self.data[a: b]
            second = self.data[b: c]
        return first, second


class SymDataset(Dataset):  # Dataset是数据集位置，就是下面的path

    def __init__(self, data, input_n, output_n):
        super(SymDataset, self).__init__()
        self.data = data
        self.data = torch.from_numpy(self.data)
        self.data = self.data.reshape((-1, self.data.shape[1], 6))
        self.input_n = input_n
        self.output_n = output_n
        self.num = self.data.shape[0] - (input_n + output_n)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        a = item
        b = item + self.input_n
        c = item + self.input_n + self.output_n
        return self.data[a: b], self.data[b: c]


class H36Datasets(Dataset):

    def __init__(self, data, input_n, output_n, is_train):
        self.data = data
        self.is_train = is_train
        self.input_n = input_n
        self.output_n = output_n
        self.data_idx = []
        seq_len = input_n + output_n

        for i in range(data.shape[0]):
            num_frames = len(data[i])
            valid_frames = np.arange(0, num_frames - seq_len + 1)
            key = [i] * num_frames
            valid_frames = list(valid_frames)
            self.data_idx.extend(zip(key, valid_frames))

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start = self.data_idx[item]
        mid = start + self.input_n
        end = start + self.input_n + self.output_n
        if self.is_train:
            first = torch.tensor(self.data[key][start: mid], device='cuda')
            second = torch.tensor(self.data[key][mid: end], device='cuda')
        else:
            first = torch.tensor(self.data[key][start: mid])
            second = torch.tensor(self.data[key][mid: end])
        return first, second


class DPWDatasets(Dataset):

    def __init__(self,input_n,output_n,skip_rate=1,split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = './dataset/3dpw_dataset/sequenceFiles'
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        #self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        seq_len = self.in_n + self.out_n

        if split == 0:
            data_path = self.path_to_data + '/train/'
        elif split == 2:
            data_path = self.path_to_data + '/test/'
        elif split == 1:
            data_path = self.path_to_data + '/validation/'
        files = []
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            files.extend(filenames)

        skel = np.load('./smpl_skeleton.npz')
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()[:, :22]
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            parent[i] = parents[i]
        n = 0

        sample_rate = int(60 // 25)

        for f in files:
            #print('f', f)
            with open(data_path + f, 'rb') as f:
                # print('>>> loading {}'.format(f))
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['poses_60Hz']
                for i in range(len(joint_pos)):
                    poses = joint_pos[i]
                    fn = poses.shape[0]
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    poses = poses[:, :-2]
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                    p3d = p3d * 1000
                    #self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    the_sequence = p3d.cpu().data.numpy()
                    diff = the_sequence[1:, :, :] - the_sequence[:-1, :, :]
                    velocity = np.linalg.norm(diff, ord=2, axis=-1)
                    angle = diff / velocity[:, :, np.newaxis]
                    angle = np.where(np.isnan(angle), np.full_like(angle, 0), angle)
                    the_angle_velocity = np.concatenate((angle, velocity[:, :, np.newaxis]), axis=-1)
                    the_sequence = np.concatenate((the_angle_velocity, the_sequence[1:]), axis=-1)
                    self.p3d.append(the_sequence)
                    fn -= 1
                    # # vis
                    # import utils.vis_util as vis_util
                    # from mpl_toolkits.mplot3d import Axes3D
                    # ax = plt.subplot(111, projection='3d')
                    # vis_util.draw_skeleton_smpl(ax, self.p3d[0][0], parents=parents[:22])

                    if split == 2:
                        # valid_frames = np.arange(0, fn - seq_len + 1, opt.skip_rate_test)
                        # valid_frames = np.arange(0, fn - seq_len + 1, 2)
                        valid_frames = np.arange(0, fn - seq_len + 1)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]


def add_frame(x, use_gpu):
    actual_xyz = x[:, :, :, 4:]
    b, f, n, a = actual_xyz.shape
    diff1 = (actual_xyz[:, 1:] - actual_xyz[:, :-1]).mean(dim=[-2, -1]) > 1.0
    diff2 = (actual_xyz[:, 1:] - actual_xyz[:, :-1]).mean(dim=[-2, -1]) < -1.0
    if use_gpu:
        diff = torch.BoolTensor(b, f * 2 - 1).cuda()
    else:
        diff = torch.BoolTensor(b, f * 2 - 1)
    diff[:, 0::2] = True
    diff[:, 1::2] = diff1 | diff2
    if use_gpu:
        y = torch.FloatTensor(b, f * 2 - 1, n, a).cuda()
    else:
        y = torch.FloatTensor(b, f * 2 - 1, n, a)
    y[:, 0::2] = actual_xyz
    y[:, 1::2] = (actual_xyz[:, :-1] + actual_xyz[:, 1:]) / 2
    if use_gpu:
        result = torch.tensor([]).cuda()
    else:
        result = torch.tensor([])
    for i in range(b):
        choose = y[i][torch.nonzero(diff[i] == True).squeeze(dim=-1)]
        ex = 2 * f - 1 - choose.shape[0]
        if use_gpu:
            one_batch = torch.tensor([]).cuda()
        else:
            one_batch = torch.tensor([])
        one_batch = torch.cat((one_batch, choose[0].unsqueeze(dim=0).repeat(ex, 1, 1), choose))
        result = torch.cat((result, one_batch.unsqueeze(dim=0)))
    if use_gpu:
        ans = torch.FloatTensor(b, f * 2 - 1, n, 7).cuda()
    else:
        ans = torch.FloatTensor(b, f * 2 - 1, n, 7)
    ans[:, 1:, :, :4] = loc_exchange(result)
    ans[:, :, :, 4:] = result
    return ans


def frame_mpjpe_error(batch_pred, batch_gt):
    batch_pred.contiguous()
    batch_gt.contiguous()

    frame_num = batch_pred.shape[1]

    ans = []

    for i in range(frame_num):
        pred = batch_pred[:, i, :, :]
        gt = batch_gt[:, i, :, :]
        pred = pred.contiguous().view(-1, 3)
        gt = gt.contiguous().view(-1, 3)
        ans.append(torch.mean(torch.norm(gt - pred, 2, 1)).item())

    return ans
