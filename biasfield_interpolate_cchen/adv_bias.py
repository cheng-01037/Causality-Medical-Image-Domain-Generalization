# code written by Chen Chen
# trimmed for the domain generalization project
import torch
import torch.nn.functional as F
import numpy as np

from .adv_transformation_base import AdvTransformBase
from .utils import rescale_intensity
from pdb import set_trace

def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=torch.float32, device='gpu'):
    '''
    generate bspline 2D kernel matrix.
    From wiki: https://en.wikipedia.org/wiki/B-spline, Fast b-spline interpolation on a uniform sample domain can be
    done by iterative mean-filtering
    :param sigma: tuple integers, control smoothness
    :param order: the order of interpolation
    :param asTensor:
    :param dtype: data type
    :param use_gpu: bool
    :return:
    '''
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(i * padding).tolist()) / ((sigma[0] * sigma[1]))

    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()

class AdvBias(AdvTransformBase):
    """
     Adv Bias field
    """
    def __init__(self,
                 config_dict={
                'epsilon':0.3,
                 'xi':1e-6,
                 'control_point_spacing':[32,32],
                 'downscale':2,
                 'data_size':[2,1,128,128],
                 'interpolation_order':3,
                 'init_mode':'gaussian',
                 'space':'log'},
                 use_gpu:bool = True, debug: bool = False):
        super(AdvBias, self).__init__(config_dict=config_dict,use_gpu=use_gpu,debug=debug)
        self.param=None

    def init_config(self,config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        self.epsilon = config_dict['epsilon']
        self.xi = config_dict['xi']
        self.data_size = config_dict['data_size']
        self.control_point_spacing = config_dict['control_point_spacing']
        self.downscale = config_dict['downscale']
        self.interpolation_order = config_dict['interpolation_order']

        self.space=config_dict['space']
        self.init_mode=config_dict['init_mode']

        # optional params added by co1818


    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        self._device = 'cuda' if self.use_gpu else 'cpu'

        self._dim = len(self.control_point_spacing)
        self.spacing =  self.control_point_spacing
        self._dtype= torch.float32
        self.batch_size=self.data_size[0]
        self._image_size = np.array([self.data_size[2], self.data_size[3]])
        self.magnitude=self.epsilon
        self.order = self.interpolation_order
        self.downscale=self.downscale ## reduce image size to save memory

        self.use_log = True #if self.space == 'log' else False

        ## contruct and initialize control points grid with random values
        self.param,self.interp_kernel,self.bias_field = self.init_bias_field()
        return self.param



    def make_small_parameters(self):
        raise Exception('by co1818: should not call a detaching op')
        self.param = self.unit_normalize(self.param, p_type ='l2')*self.xi
        self.param = self.param.detach()
        self.param.requires_grad=True
        return self.param

    def rescale_parameters(self,power_iteration=False):
        ## restrict control points values in the 1-ball space
        self.param=self.unit_normalize(self.param, p_type ='l2')

    # def optimize_parameters(self,power_iteration=False, step_size=1): # original implementated by cc215
    # new implementation by co1818, supporting GD

    def optimize_parameters(self,power_iteration=False, step_size=1, upd_direction = 'GA'):
        '''
        Args co1818:
            upd_direction: direction of update. GA for gradient ascend in original implmentation. GD for descent, added by co1818
        '''
        if self.debug: print ('optimize bias')
        grad = self.unit_normalize(self.param.grad,p_type ='l2')
        if power_iteration:
            raise Exception('co1818: behavior unknown')
            self.param = grad.clone().detach()
        else:
            if upd_direction == 'GA':
                ## Gradient ascent by cc215
                self.param = self.param +step_size*grad.detach()
                self.param = self.param.clone().detach()
            elif upd_direction == 'GD': # added by co1818
                self.param = self.param - step_size*grad.detach()
                self.param = self.param.clone().detach()
            else:
                raise NotImplementedError(f'Unknown updating direction {upd_direction}')

        return self.param

    def set_parameters(self,param):
        raise Exception('by co1818: should not call a detached op')
        self.param=param.detach()

    def forward(self, data):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''
        assert self.param is not None, 'init param before transform data'

        bias_field = self.compute_smoothed_bias(self.param)
        bias_field = self.rescale_bias(bias_field,magnitude=self.epsilon)
        self.bias_field =bias_field
        self.diff=bias_field

        ## in case the input image is a multi-channel input.
        if bias_field.size(1)<data.size(1):
            bias_field = bias_field.expand(data.size())

        transformed_input = bias_field*data
        if self.debug:
            print ('bias transformed',transformed_input.size())
        return transformed_input

    def backward(self,data):
        if self.debug:
            print ('max magnitude',torch.max(torch.abs(self.bias_field-1)))
        return data

    def predict_forward(self,data):
        return data

    def predict_backward(self,data):
        return data

    def init_bias_field(self,init_mode=None):
        '''
        init cp points, interpolation kernel, and resulted bias field.
        :param batch_size:
        :param spacing: tuple of ints
        :param order:
        :return:bias field
        reference:
        bspline interpoplation is adapted from airlab: class _KernelTransformation(_Transformation):
https://github.com/airlab-unibas/airlab/blob/1a715766e17c812803624d95196092291fa2241d/airlab/transformation/pairwise.py
        '''
        if init_mode is None:
            mode=self.init_mode

        ## set up cpoints grid
        self._stride = np.array(self.spacing) # 32 or 24
        cp_grid = np.ceil(np.divide(self._image_size/(1.0*self.downscale), self._stride)).astype(dtype=int) # co1818: the number of control points
        # new image size after convolution
        inner_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)
        # add one control point outside each side, e.g.2 by 2 grid, requires 4 by 4 control points
        cp_grid = cp_grid + 2
        # image size with additional control points
        new_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)
        # center image between control points
        image_size_diff = inner_image_size - self._image_size/(1.0*self.downscale)
        image_size_diff_floor = np.floor((np.abs(image_size_diff)/2))*np.sign(image_size_diff)
        self._crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor
        self.cp_grid = [self.batch_size, 1] + cp_grid.tolist()

         # initialize control points parameters for optimization
        if mode == 'gaussian':
            self.param = torch.ones(*self.cp_grid).normal_(mean=0,std=1)

        # elif mode =='random':
        #     ## diff to the identity
        #     if self.use_log:
        #         raise NotImplementedError
        #     else:
        #         self.param=(torch.rand(*self.cp_grid)*2-1)*self.magnitude

        elif mode == 'identity':
            ## static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise  NotImplementedError

        self.param = self.unit_normalize(self.param, p_type ='l2')

        self.param = self.param.to(dtype=self._dtype, device=self._device)

        # convert to integer
        self._stride = self._stride.astype(dtype=int).tolist()
        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [self.batch_size, 1] + new_image_size.astype(dtype=int).tolist()
        ## initialize interpolation kernel
        self.interp_kernel=self.get_bspline_kernel(order=self.order, spacing=self.spacing)
        self.interp_kernel=self.interp_kernel.to(self.param.device)
        self.bias_field = self.compute_smoothed_bias(self.param,padding=self._padding,stride=self._stride)
        self.bias_field = self.rescale_bias(self.bias_field)
        if self.debug:
            print('initialize {} control points'.format(str(self.param.size())))

        return self.param,self.interp_kernel,self.bias_field


    # NOTE: this is added by co1818 for convenience
    def reset_bias_value(self):
        '''
        Reset param (ct point values)
        '''
        del self.param
        del self.bias_field

        mode = self.init_mode
        if mode == 'gaussian':
            self.param = torch.ones(*self.cp_grid).normal_(mean=0,std=1)

        elif mode == 'identity':
            ## static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise  NotImplementedError

        self.param = self.unit_normalize(self.param, p_type ='l2')
        self.param = self.param.to(dtype=self._dtype, device=self._device)

        self.bias_field = self.compute_smoothed_bias(self.param,padding=self._padding,stride=self._stride)
        self.bias_field = self.rescale_bias(self.bias_field)

        return self.param, self.bias_field

    def compute_smoothed_bias(self, cpoint=None,interpolation_kernel=None, padding=None,stride=None):
        '''
        generate bias field given the cppints N*1*k*l
        :return: bias field bs*1*H*W
        '''
        if interpolation_kernel is None:
            interpolation_kernel=self.interp_kernel
        if padding is None:
            padding =self._padding
        if stride is None:
            stride =self._stride
        if cpoint is None:
            cpoint=self.param

        bias_field = F.conv_transpose2d(cpoint, interpolation_kernel,
                                          padding=padding, stride=stride, groups=1)
        # crop bias
        bias_field_tmp=bias_field[:, :,
                       stride[0] + self._crop_start[0]:-stride[0] - self._crop_end[0],
                       stride[1] + self._crop_start[1]:-stride[1] - self._crop_end[1]]

        ## recover bias field to original image resolution for efficiency.
        if self.debug: print ('after intep, size:',bias_field_tmp.size())
        scale_factor = self._image_size[0] / bias_field_tmp.size(2)
        if scale_factor>1:
            upsampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                                            align_corners=True)
            diff_bias = upsampler(bias_field_tmp)
        else:
            diff_bias=bias_field_tmp

        bias_field = torch.exp(diff_bias)

        return bias_field

    def rescale_bias(self,bias_field,magnitude=None):
        """[summary]
        rescale the bias field so that it values fall in [1-magnitude, 1+magnitude]
        Args:
            bias_field ([torch 4d tensor]): [description]
            magnitude ([scalar], optional): [description]. Defaults to use predefined value.

        Returns:
            [type]: [description]
        """
        if magnitude is None:
            magnitude=self.magnitude
        assert magnitude>0

        bias_field =rescale_intensity(bias_field,1-magnitude,1+magnitude)

        if self.debug:
            print('L_infinity: max|bias-id|', torch.max(torch.abs(bias_field-1)))
        return bias_field


    def get_bspline_kernel(self,spacing, order=3):
        '''

        :param order init: bspline order, default to 3
        :param spacing tuple of int: spacing between control points along h and w.
        :return:  kernel matrix
        '''
        self._kernel = bspline_kernel_2d(spacing, order=order, asTensor=True, dtype=self._dtype, device=self._device)
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()
        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)
        return self._kernel
    def get_name(self):
        return 'bias'
    def is_geometric(self):
        return 0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import check_dir
    log_dir = "./result/log/debug/"
    check_dir(log_dir, create=True)
    images = torch.ones(2,1,128,128).cuda()
    images[:,:,::2,::2]=2.0
    images[:,:,::3,::3]=3.0
    images[:,:,::1,::1]=1.0
    images=images.float()
    images.requires_grad=False
    print ('input:',images)
    augmentor= AdvBias(debug=True,use_gpu=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    error = transformed-images
    print ('sum error', torch.sum(error))

    plt.subplot(131)
    plt.imshow(images.detach().cpu().numpy()[0,0])

    plt.subplot(132)
    plt.imshow(transformed.detach().cpu().numpy()[0,0])

    plt.subplot(133)
    plt.imshow((transformed/images).detach().cpu().numpy()[0,0])
    plt.savefig('test_bias.png')



