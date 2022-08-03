# 3D version of GIN in case you are using a 3D network. 3D ver. of IPA will be released soon
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace

class GradlessGCReplayNonlinBlock3D(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, init_scale = 'default', **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        self.init_scale     = init_scale
        assert requires_grad == False

    def forward(self, x_in, requires_grad = False):
        # random size of kernel
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny, nz = x_in.shape

        ker = torch.randn([self.out_channel * nb, self.in_channel , k, k, k  ], requires_grad = self.requires_grad  ).cuda()
        shift = torch.randn( [self.out_channel * nb, 1, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny, nz)
        x_conv = F.conv3d(x_in, ker, stride =1, padding = k // 2, dilation = 1, groups = nb )
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv

class GINGroupConv3D(nn.Module):
    def __init__(self, out_channel = 3, in_channel = 3, interm_channel = 2, scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', init_scale = 'default', **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv3D, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, init_scale = init_scale, layer_id = 0).cuda()
                )
        for ii in range(n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, init_scale = init_scale,layer_id = ii + 1).cuda()
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, init_scale = init_scale, layer_id = n_layer - 1, use_act = False).cuda()
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)

        nb, nc, nx, ny, nz = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None, None] # nb, 1, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1, 1).cuda() # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed

###### unit test #####
'''
if __name__ == '__main__':
    from pdb import set_trace
    xin = torch.rand([5, 3, 64, 64, 32]).cuda()
    augmenter = GINGroupConv3D().cuda()
    out = augmenter(xin)
    set_trace()
    print(out.shape)


'''
