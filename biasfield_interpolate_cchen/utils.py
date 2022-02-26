# code written by Chen Chen
# trimmed for domain generalization project
import os
import torch
from pdb import set_trace




def rescale_intensity(data,new_min=0,new_max=1,eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c , h, w = data.size(0),data.size(1),data.size(2), data.size(3)
    data = data.view(bs*c, -1)
    # pytorch 1.3
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    # co1818: in adjust to pytorch 0.4 for wtbenv
    #old_max = torch.max(data, dim=1, keepdim=True)[0]
    #old_min = torch.min(data, dim=1, keepdim=True)[0]

    new_data = (data - old_min+eps) / (old_max - old_min + eps)*(new_max-new_min)+new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data

def rescale_intensity04(data,new_min=0,new_max=1,eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c , h, w = data.size(0),data.size(1),data.size(2), data.size(3)
    data = data.view(bs*c, -1)
    # pytorch 1.3
    #old_max = torch.max(data, dim=1, keepdim=True).values
    #old_min = torch.min(data, dim=1, keepdim=True).values

    old_max = torch.max(data, dim=1, keepdim=True)[0]
    old_min = torch.min(data, dim=1, keepdim=True)[0]

    new_data = (data - old_min+eps) / (old_max - old_min + eps)*(new_max-new_min)+new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data





def check_dir(dir_path, create=False):
    '''
    check the existence of a dir, when create is True, will create the dir if it does not exist.
    dir_path: str.
    create: bool
    return:
    exists (1) or not (-1)
    '''
    if os.path.exists(dir_path):
        return 1
    else:
        if create:
            os.makedirs(dir_path)
        return -1
