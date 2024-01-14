import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.build()

    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        # 删除地理卷积层做的修改
        # self.process_coords = nn.Linear(4, 32)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)

        states = self.state_em(traj['states'].long())

        locs = torch.cat((lngs, lats, states), dim=2)
        # print(locs[0])
        # print(locs.size())

        # map the coords into 16-dim vector
        locs = F.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)

        # print(locs.size())

        # 删除地理卷积层
        # conv_locs = locs.permute(0, 2, 1)
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)
        # print(conv_locs.size())

        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'],
                                         config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim=2)


        # print(local_dist.size())

        conv_locs = torch.cat((conv_locs, local_dist), dim=2)
        # print(conv_locs.size())

        return conv_locs
