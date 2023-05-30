import os

import torch
from torch.utils.data import Dataset
import numpy as np

import forward_kinematics


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def define_actions_cmu(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]
    if action in actions:
        return [action]

    if action == "all":
        return actions

    raise (ValueError, "Unrecognized action: %d" % action)


def expmap2xyz_torch_cmu(opt, expmap):
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables_cmu()
    xyz = forward_kinematics.fkl_torch(opt, expmap, parent, offset, rotInd, expmapInd)

    return xyz


def load_data_cmu_3d_all(opt, path_to_dataset, actions, input_n, output_n, is_test=False):
    joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(38 * 3), dimensions_to_ignore)
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            if '.txt' in _:
                count = count + 1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            print('read ' + filename)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            exptmps = torch.from_numpy(action_sequence).float().to('cuda:0')
            xyz = expmap2xyz_torch_cmu(opt, exptmps)
            xyz = xyz.view(-1, 38 * 3)
            xyz = xyz.cpu().data.numpy()
            xyz = xyz[:, dimensions_to_use]

            save_filename = '{}/{}/{}_{}.npy'.format(path_to_dataset, action, action, examp_index + 1)
            np.save(save_filename, xyz)
            action_sequence = xyz

            # 以间距为2进行采样 调整帧率
            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            the_sequence = the_sequence.reshape(num_frames, -1, 3)
            diff = the_sequence[1:, :, :] - the_sequence[:-1, :, :]
            velocity = np.linalg.norm(diff, ord=2, axis=-1)
            angle = diff / velocity[:, :, np.newaxis]
            angle = np.where(np.isnan(angle), np.full_like(angle, 0), angle)
            the_angle_velocity = np.concatenate((angle, velocity[:, :, np.newaxis]), axis=-1)
            the_sequence = np.concatenate((the_angle_velocity, the_sequence[1:]), axis=-1)
            num_frames -= 1
            # 采样的样本数
            fs = np.arange(0, num_frames - seq_len + 1)
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))
            fs_sel = fs_sel.transpose()
            seq_sel = the_sequence[fs_sel, :]
            if len(sampled_seq) == 0:
                sampled_seq = seq_sel
                complete_seq = the_sequence
            else:
                sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                complete_seq = np.append(complete_seq, the_sequence, axis=0)

    return sampled_seq, dimensions_to_ignore, dimensions_to_use


def load_data_cmu_3d_n(opt, path_to_dataset, actions, input_n, output_n, is_test=False):
    joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(38 * 3), dimensions_to_ignore)
    test_sample_num = 256
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            if 'txt' in _:
                count = count + 1
        for examp_index in np.arange(count):
            print('eval or test read {}'.format(action))
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            exptmps = torch.from_numpy(action_sequence).float().to('cuda:0')
            xyz = expmap2xyz_torch_cmu(opt, exptmps)
            xyz = xyz.view(-1, 38 * 3)
            xyz = xyz.cpu().data.numpy()
            xyz = xyz[:, dimensions_to_use]
            action_sequence = xyz

            save_filename = '{}/{}/{}_{}.npy'.format(path_to_dataset, action, action, examp_index + 1)
            np.save(save_filename, xyz)

            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            the_sequence = the_sequence.reshape(num_frames, -1, 3)
            diff = the_sequence[1:, :, :] - the_sequence[:-1, :, :]
            velocity = np.linalg.norm(diff, ord=2, axis=-1)
            angle = diff / velocity[:, :, np.newaxis]
            angle = np.where(np.isnan(angle), np.full_like(angle, 0), angle)
            the_angle_velocity = np.concatenate((angle, velocity[:, :, np.newaxis]), axis=-1)
            the_sequence = np.concatenate((the_angle_velocity, the_sequence[1:]), axis=-1)
            num_frames -= 1

            if (not is_test) or test_sample_num<0:
                #如果不是测试的话就不用随机采样
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                #这里为什么source_seq_len 被固定为50帧？因为为了和之前一个方法保持一致，实际运行过程中会根据实际的输入长度进行调整。
                source_seq_len = 50
                target_seq_len = output_n
                total_frames = source_seq_len + target_seq_len
                batch_size = test_sample_num
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[
                              idx + (source_seq_len - input_n):(idx + source_seq_len + output_n), :]
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    return sampled_seq, dimensions_to_ignore, dimensions_to_use



class CMU_Motion3D(Dataset):

    def __init__(self, input_n, output_n, split, actions='all'):

        opt = {'cuda_idx': 'cuda:0', 'test_sample_num': 256}

        self.path_to_data = './dataset/cmu_mocap_dataset'
        input_n = input_n
        output_n = output_n

        self.split = split
        is_all = actions
        actions = define_actions_cmu(actions)
        # actions = ['walking']
        if split == 0:
            path_to_data = self.path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = self.path_to_data + '/test/'
            is_test = False


        if not is_test:
            all_seqs, dim_ignore, dim_use = load_data_cmu_3d_all(opt, path_to_data, actions,
                                                                                             input_n, output_n,
                                                                                             is_test=is_test)
        else:
            # all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions,
            #                                                                 input_n, output_n,
            #                                                                 is_test=is_test)

            all_seqs, dim_ignore, dim_use = load_data_cmu_3d_n(opt, path_to_data, actions,
                                                                                             input_n, output_n,
                                                                                             is_test=is_test)

        self.all_seqs = all_seqs
        self.dim_used = dim_use

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        return self.all_seqs[item]
