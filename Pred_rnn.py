'''

time:2020 5 7

'''
import torch
import torch.nn as nn
from GradientHighway_pytorch import GHU as ghu
from CausalLSTMCell_pytorch import CausalLSTMCell as cslstm


class RNN(nn.Module):
    def __init__(self, shape, num_layers, num_hidden, seq_length, tln=True):
        super(RNN, self).__init__()

        self.img_width = shape[-2]
        self.img_height = shape[-1]
        self.total_length = shape[1]
        self.input_length = shape[0]
        self.shape = [shape[0], shape[2], shape[3], shape[4]]
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        ghu_list = []

        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = 1
            else:
                num_hidden_in = self.num_hidden[i - 1]
            cell_list.append(cslstm('lstm_' + str(i + 1),
                                   num_hidden_in,
                                   num_hidden[i],
                                   self.shape, 1.0, tln=tln))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[-1], 1, 1, 1, 0)
        ghu_list.append(ghu('highway', self.shape, self.num_hidden[1], tln=tln))
        self.ghu_list = nn.ModuleList(ghu_list)


    def forward(self, images):
        # [batch, length, channel, width, height]

        batch = images.shape[0]
        height = images.shape[3]
        width = images.shape[4]

        next_images = []
        h_t = []
        c_t = []
        z_t = None
        m_t = None

        for i in range(self.num_layers):
            h_t.append(None)
            c_t.append(None)

        for t in range(self.total_length):
            if t < self.input_length:
                net = images[:,t]
            h_t[0], c_t[0], m_t = self.cell_list[0](net, h_t[0], c_t[0], m_t)
            z_t = self.ghu_list[0](h_t[0],z_t)
            h_t[1], c_t[1], m_t = self.cell_list[1](z_t, h_t[1], c_t[1], m_t)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], m_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_images.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_images = torch.stack(next_images, dim=1)
        out  =  next_images
        next_images = []

        return out

if __name__ == '__main__':
    a = torch.randn(2, 6, 1, 250, 350)
    shape = [2, 6, 1, 250, 350]
    numlayers = 4
    predrnn = RNN(shape, numlayers, [1,1,1,1], 6, True)
    predict = predrnn(a)
    print(predict.shape)
