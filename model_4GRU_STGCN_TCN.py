import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from torch.nn.utils import weight_norm


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 time_dim,
                 joints_dim,):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim,
                                                joints_dim))  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        # blocks = []
        # for i in range(block_num):
        #     blocks += [TemporalConvNet(tcn_input_size, num_channels, kernel_size=kernel_size, dropout=dropout)]
        # self.tcn1 = nn.Sequential(*blocks)

        '''
        self.prelu = nn.PReLU()

        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def forward(self, x):
        # print("x", x.shape)
        # y = x.permute(0, 3, 2, 1)
        # y = torch.squeeze(y, dim=3)
        # for i in range(len(self.tcn1)):
        #     y = self.tcn1[i](y) + y
        # y = torch.unsqueeze(y, dim=3)
        # y = y.permute(0, 3, 2, 1)

        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        #print("nctv,vtq->ncqv", x.shape)
        ## x=self.prelu(x)
        # x += y

        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        # print("nctv,tvw->nctw", x.shape)
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()

#    　origin GCN+GRU
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, padding_mode='replicate'))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, padding_mode='replicate'))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels=[25, 25, 25, 25], kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)
class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, node_num, batch_size, block_num=1):

        super(Generator, self).__init__()
        self.hidden_prev = nn.Parameter(torch.zeros(1, batch_size, hidden_size))
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, dropout=0.05, batch_first=True)
        #self.my_variable = nn.Parameter(torch.tensor(0.1))
        self.GCN = ConvTemporalGraphical(input_size, node_num)

        blocks = []
        for i in range(block_num):
            blocks += [TemporalConvNet(num_inputs=node_num, num_channels=[node_num, node_num, node_num, node_num])]
        self.tcn = nn.Sequential(*blocks)
        self.linear = nn.Linear(hidden_size, output_size)  # 输出层
        self.prelu = nn.PReLU()
        self.fc = nn.Linear(input_size * 2, input_size)


    def forward(self, x, hidden_size):
        # GCN block
        #print("x", x.shape)
        #print("1: ",x.shape) # torch.Size([16, 25, 9])
        y = x
        for i in range(len(self.tcn)):
            y = self.tcn[i](y) + y
        #print("y", y.shape)


        x = x.permute(0, 2, 1)
        #print("x", x.shape)
        # print("2: ",x.shape) # torch.Size([16, 9, 25])
        x = torch.unsqueeze(x, dim=3)
        #print("x", x.shape)
        x = x.permute(0, 3, 1, 2)

        # GCN_set = self.GCN(x)
        # x = GCN_set.permute(0, 3, 2, 1)
        # x = torch.squeeze(x, dim=3)
        # x = torch.cat([x, y], dim=2)
        # x = self.fc(x)
        GCN_set = self.GCN(x)
        GCN_set = GCN_set + x
        GCN_set = self.prelu(GCN_set)
        x = GCN_set.permute(0, 3, 2, 1)
        x = torch.squeeze(x, dim=3)
        x = torch.cat([x, y], dim=2)
        x = self.fc(x)

        #print("x", x.shape)
        out, h = self.GRU(x, self.hidden_prev)
        #print("out", out.shape)
        # print("6: ",out.size()) # torch.Size([16, 25, 128])
        out = out.reshape(-1, hidden_size)
        # print("7: ",out.size()) # torch.Size([400, 128])
        out = self.linear(out)
        # print("8: ",out.size()) # torch.Size([400, 25])
        out = out.unsqueeze(dim=0)

        return out
