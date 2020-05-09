'''
time: 2020 5 7
'''

from TensorLayerNorm_pytorch import tensor_layer_norm
import torch
import torch.nn as nn

class GHU(nn.Module):
    def __init__(self, layer_name,inputs_shape,num_features, tln=False):
        super(GHU,self).__init__()
        """Initialize the Gradient Highway Unit.
        """
        self.layer_name = layer_name
        self.num_features = num_features
        self.layer_norm = tln
        self.batch = inputs_shape[0]
        self.height = inputs_shape[3]
        self.width = inputs_shape[2]

        self.bn_z_concat = tensor_layer_norm(self.num_features*2)
        self.bn_x_concat = tensor_layer_norm(self.num_features*2)

        self.z_concat_conv = nn.Conv2d(self.num_features,self.num_features*2,5,1,2)
        self.x_concat_conv = nn.Conv2d(self.num_features,self.num_features*2,5,1,2)

    def init_state(self):
        return torch.zeros((self.batch,self.num_features,self.width,self.height), dtype=torch.float32)


    def forward(self,x,z):
        if z is None:
            z = self.init_state()
        z_concat = self.z_concat_conv(z)
        if self.layer_norm:
            z_concat = self.bn_z_concat(z_concat)

        x_concat = self.x_concat_conv(x)
        if self.layer_norm:
            x_concat = self.bn_x_concat(x_concat)

        gates = torch.add(x_concat, z_concat)
        p, u = torch.split(gates, self.num_features, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new

# if __name__ == '__main__':
#     a = torch.randn(2,32,250,350)
#     lstm = GHU("name",[2,32,250,350],32,True)
#     new_h= lstm(a,None)
#     print(new_h.shape)
