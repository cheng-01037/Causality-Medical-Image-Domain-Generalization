# code written by Chen Chen
# trimmed for domain generalization project

import torch


class AdvTransformBase(torch.nn.Module):
    """
     Adv Transformer base
    """

    def __init__(self,
                 config_dict={'size':1,
                 'mean':0,
                 'std':0.1,
                 'xi':1e-6

                 },

                 use_gpu:bool = True, debug: bool = False):
        '''


        '''
        super(AdvTransformBase, self).__init__()
        self.config_dict=config_dict
        self.param=None
        self.is_training=False
        self.use_gpu = use_gpu
        self.debug = debug
        if self.use_gpu:
            self.device  = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_config(self,config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        if self.debug: print ('init base class')
        self.size = config_dict['size']
        self.mean = config_dict['mean']
        self.std = config_dict['std']
        self.xi = config_dict['xi']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        noise = torch.randn(self.size,device=self.device, dtype = torch.float32)*self.std+self.mean
        self.param=noise
        return noise

    def set_parameters(self,param):
        self.param=param.detach()


    def make_small_parameters(self):
        self.param = self.xi*self.unit_normalize(self.param)


    def get_parameters(self):
        return self.param

    def train(self):
        self.is_training = True
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def eval(self):
        self.param.requires_grad=False
        self.is_training =False

    def rescale_parameters(self):
        self.param = self.xi*unit_normalize(self.param)
        return self.param

    def optimize_parameters(self,set=False):
        grad = self.param.grad.sign()
        if self.debug: print ('grad',grad.size())
        if set:
            self.param = grad.detach()
        return self.param

    def forward(self, data):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''
        assert self.param is not None, 'init param before transform data'
        transformed_input = data+self.param
        if self.debug:
            print ('transformed',transformed_input.size())
        return transformed_input

    def backward(self,data):
        assert self.param is not None, 'init param before transform data'
        warped_back_output = data-self.param
        if self.debug:
            print ('back:',warped_back_output.size())
        return warped_back_output

    def unit_normalize(self,d, p_type='l2'):
        if p_type=='l1':
           old_size=d.size()
           d_flatten=d.view(d.size(0),-1)
           norm= d_flatten.norm(p=1, dim=1, keepdim=True)
           d_normalized = d_flatten.div(norm.expand_as(d_flatten))
           return d_normalized.view(old_size)
        elif p_type =='infinity':
            d_abs_max = torch.max(
            torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1, 1)
            # print(d_abs_max.size())
            d /= (1e-20 + d_abs_max) ## d' =d/d_max
        if p_type=='l2':
            d_abs_max = torch.max(
            torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1, 1)
            # print(d_abs_max.size())
            d /= (1e-20 + d_abs_max) ## d' =d/d_max
            d /= torch.sqrt(1e-6 + torch.sum(
                torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True)) ##d'/sqrt(d'^2)
        return d


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    images = torch.zeros((10,1)).cuda()
    # images[:,:,4:12,4:12]=1
    print ('input:',images)
    augmentor= AdvTransformBase(config_dict={'size':1,
                 'mean':0,
                 'std':0.1,
                 'xi':1e-6
                 },debug=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    recovered = augmentor.backward(transformed)
    # error = recovered-images
    # print ('sum error', torch.sum(error))

