'''
Trainer for GIN-IPA experiments
'''
import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import my_utils.util as util
from .base_model import *
from pdb import set_trace
import numpy as np
from .smpmodels import*

from .imagefilter import GINGroupConv # gin
import models.segloss as segloss
import sys

from biasfield_interpolate_cchen.adv_bias import AdvBias # interpolating random control points, implemented by Chen Chen et al.
from biasfield_interpolate_cchen.adv_bias import rescale_intensity

class ExperimentNet(BaseModel):
    def name(self):
        return 'ExperimentNet'

    #def set_encoders_and_decoders(self, opt):
    def set_networks(self, opt):
        self.n_cls = opt.nclass
        self.gpu_ids = opt.gpu_ids
        if opt.model == 'efficient_b2_unet':
            self.netSeg = efficient_unet(nclass = self.n_cls, in_channel = 3, gpu_ids = opt.gpu_ids)
        else:
            raise NotImplementedError

        # auxillary nodes
        self.onehot_node = segloss.One_Hot(self.n_cls)
        self.softmax_node = torch.nn.Softmax(dim = 1)

        # data augmentation nodes
        if opt.exp_type == 'gin':
            self.img_transform_node = GINGroupConv(out_channel = opt.gin_out_nc, n_layer = opt.gin_nlayer, interm_channel = opt.gin_n_interm_ch, out_norm = opt.gin_norm).cuda()
        elif opt.exp_type == 'ginipa':
            # ipa
            blender_cofig = {
                    'epsilon': opt.blend_epsilon,
                    'xi': 1e-6,
                    'control_point_spacing':[opt.blend_grid_size, opt.blend_grid_size],
                    'downscale':2, #
                    'data_size':[opt.batchSize,1,opt.fineSize, opt.fineSize],
                    'interpolation_order':2,
                    'init_mode':'gaussian',
                    'space':'log'
                    }

            self.img_transform_node = GINGroupConv(out_channel = opt.gin_out_nc, n_layer = opt.gin_nlayer, interm_channel = opt.gin_n_interm_ch, out_norm = opt.gin_norm).cuda()
            self.blender_node       = AdvBias(blender_cofig) # IPA
            self.blender_node.init_parameters()

        else:
            raise NotImplementedError(f'Unknown exp type: {opt.exp_type}')

        print(f'Using image transform type {opt.exp_type}')


    def initialize(self, opt):
        ## load the model.
        BaseModel.initialize(self, opt)
        self.set_networks(opt)

        if opt.continue_train or opt.phase == 'test':
            try:
                self.load_network_by_fid(self.netSeg, opt.reload_model_fid)
            except:
                print('Cannot load the entire trainer. Trying to reload the segmenter only')
                if hasattr(self.netSeg, 'load_source_model'):
                    self.netSeg.load_source_model(opt.reload_model_fid)
                else:
                    raise Exception('Cannot reload the model')

        ## define loss functions
        self.criterionDice = segloss.SoftDiceLoss(self.n_cls).cuda(self.gpu_ids[0]) # soft dice loss
        self.ScoreDice = segloss.SoftDiceScore(self.n_cls, ignore_chan0 = True).cuda(self.gpu_ids[0]) # dice score
        self.ScoreDiceEval = segloss.Efficient_DiceScore(self.n_cls, ignore_chan0 = False).cuda(self.gpu_ids[0]) # for evaluation in 3D

        # using plain CE + Dice loss, not using WCE
        self.criterionWCE = segloss.My_CE(nclass = self.n_cls,\
                batch_size = self.opt.batchSize, weight = torch.ones(self.n_cls,)).cuda(self.gpu_ids[0])

        # consistency between conditional distributions
        if opt.consist_type == 'kld':
            self.criterionCons = torch.nn.KLDivLoss()
        else:
            raise NotImplementedError

        # initialize optimizers
        if self.opt.optimizer == 'adam':
            self.optimizer_Segmenter = torch.optim.Adam( itertools.chain(self.netSeg.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = opt.adam_weight_decay)
        else:
            raise NotImplementedError

        # register optimizers
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_Segmenter)

        # put optimizers into learning rate schedulers
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        print_network(self.netSeg)

        # register subnets which require gradients
        self.subnets = [ self.netSeg ]

        for subnet in self.subnets:
            assert next(subnet.parameters()).is_cuda == True
        print('-----------------------------------------------')

    # bypass the appearance transforms
    def set_input(self, input):
        input_img = input['img']
        input_mask = input['lb']

        if len(self.gpu_ids) > 0:
            if not isinstance(input_img, torch.FloatTensor):
                if input_img.ndims < 4:
                    input_img = input_img[np.newaxis, ...]
                input_img = torch.FloatTensor(input_img, requires_grad = False).float()
            input_img = input_img.cuda(self.gpu_ids[0])

            if not isinstance(input_mask, torch.FloatTensor):
                if input_mask.ndims < 4:
                    input_mask = input_mask[np.newaxis, ...]
                input_mask = torch.FloatTensor(input_mask, requires_grad = False).float()
            input_mask = input_mask.cuda(self.gpu_ids[0])

        self.input_img = Variable(input_img)
        self.input_mask = input_mask

    def set_input_aug(self, input):
        '''
        Applying GIN-only
        '''
        input_img = input['img']
        input_mask = input['lb']

        if len(self.gpu_ids) > 0:
            input_img = input_img.float().cuda(self.gpu_ids[0])
            input_mask = input_mask.float().cuda(self.gpu_ids[0])

        # augment appearance
        self._nb_current = input_img.shape[0] # batch size of the current batch
        input_buffer = torch.cat([  self.img_transform_node(input_img) for ii in range(3)], dim = 0) # a trick for efficient training from Xu et al.

        self.input_img_3copy = input_buffer
        self.input_mask = input_mask


    def set_input_aug_sup(self, input):
        '''
        Applying both GIN and IPA

        '''
        input_img   = input['img']
        input_mask  = input['lb']

        if len(self.gpu_ids) > 0:
            input_img = input_img.float().cuda(self.gpu_ids[0])
            input_mask = input_mask.float().cuda(self.gpu_ids[0])

        # random no-linear augmentation
        self._nb_current = input_img.shape[0] # batch size of the current batch

        # gin
        input_buffer = torch.cat([  self.img_transform_node(input_img) for ii in range(3)], dim = 0)

        if 'ipa' in self.opt.exp_type:

            self.blender_node.init_parameters()
            blend_mask = rescale_intensity(self.blender_node.bias_field).repeat(1,3,1,1)

            # spatially-variable blending
            input_cp1 = input_buffer[: self._nb_current].clone().detach() * blend_mask + input_buffer[self._nb_current: self._nb_current * 2].clone().detach() * (1.0 - blend_mask)
            input_cp2 = input_buffer[: self._nb_current] * (1 - blend_mask) + input_buffer[self._nb_current: self._nb_current * 2] *  blend_mask

            input_buffer[: self._nb_current] = input_cp1
            input_buffer[self._nb_current: self._nb_current * 2] = input_cp2

            self.blend_mask = blend_mask.data

        self.input_img_3copy = input_buffer
        self.input_mask = input_mask

    # run validation
    def validate(self):
        for subnet in self.subnets:
            subnet.eval()

        with torch.no_grad():
            img_val = self.input_img
            mask_val = self.input_mask
            pred_val, _  = self.netSeg(img_val)

            # now calculating losses!
            loss_dice_val = self.ScoreDice(pred_val, mask_val)
            loss_wce_val  = self.criterionWCE(pred_val, mask_val.long())

            self.loss_dice_val = loss_dice_val.data
            self.loss_wce_val  = loss_wce_val.data

            # visualizations!
            self.pred_val = pred_val.data
            self.gth_val  = mask_val.data

        for subnet in self.subnets:
            subnet.zero_grad()
            subnet.train()

    def get_segmentation_gpu(self, raw_logits = False):
        """
        Args:
            raw_logits: output dense masks or logits
        """
        for subnet in self.subnets:
            subnet.eval()

        with torch.no_grad():
            img_val         = self.input_img
            mask_val        = self.input_mask
            seg_val, _      = self.netSeg(img_val)
            if raw_logits != True:
                seg_val = torch.argmax(seg_val, 1)

        for subnet in self.subnets:
            subnet.zero_grad()
            subnet.train()
        return mask_val, seg_val

    def forward_seg_train(self, input_img):
        """
        run a forward segmentation in training mode
        """
        lambda_Seg  = self.opt.lambda_Seg
        lambda_wce  = self.opt.lambda_wce
        lambda_dice = self.opt.lambda_dice

        pred_all, aux_pred     = self.netSeg(input_img)
        pred = pred_all[: self._nb_current]
        loss_dice   = self.criterionDice(input = pred, target = self.input_mask)
        loss_wce    = self.criterionWCE(inputs = pred, targets = self.input_mask.long() )

        self.seg_tr         = pred.detach()
        self.loss_seg       = (loss_dice * lambda_dice + loss_wce * lambda_wce) * lambda_Seg
        self.loss_seg_tr    = self.loss_seg.data
        self.loss_dice      = loss_dice.data
        self.loss_wce       = loss_wce.data

        return pred, pred_all

    def forward_consistency(self, pred_all):
        '''
        KL-term, enforcing conditional distribution remains unchanged regardless of interventions applied
        '''
        if self.opt.consist_type == 'kld':
            lambda_consist = self.opt.lambda_consist

            pred_all_prob = F.softmax(pred_all, dim = 1)
            pred_avg = 1.0 / 3 * ( pred_all_prob[: self._nb_current] + pred_all_prob[self._nb_current : self._nb_current * 2] + pred_all_prob[self._nb_current * 2: ]) # efficient implementation inspired by Xu et al. (Randconv)
            pred_avg = torch.cat([pred_avg  for ii in range(3)], dim = 0)
            pred_all = F.log_softmax(pred_all, dim = 1) # according to pytorch 1.3 documentation, input is log_prob, target is prob
            loss_consist = self.criterionCons( pred_all, pred_avg  )
        else:
            raise NotImplementedError

        self.loss_consist = lambda_consist * loss_consist
        self.loss_consist_tr = self.loss_consist.data


    def optimize_parameters(self, **kwargs):
        self.set_requires_grad(self.subnets, True)
        pred, pred_all = self.forward_seg_train(self.input_img_3copy)
        self.forward_consistency(pred_all)
        self.optimizer_Segmenter.zero_grad()
        (self.loss_seg + self.loss_consist).backward()
        self.optimizer_Segmenter.step()
        self.set_requires_grad(self.subnets, False)

    def get_current_errors_tr(self):
        """
        Nothing
        """
        ret_errors = [ ('Dice', self.loss_dice),
                ('WCE', self.loss_wce),
                ('Consist', self.loss_consist_tr)]

        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_errors_val(self):
        ret_errors = [('loss_wce_val', self.loss_wce_val.mean()),\
                ('loss_dice_val', self.loss_dice_val.mean())]

        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_visuals_val(self):
        img_val    = t2n(self.input_img.data)
        gth_val    = t2n(self.gth_val.data)
        pred_val   = t2n( torch.argmax(self.pred_val.data, dim =1, keepdim = True ))

        ret_visuals = OrderedDict([\
                ('img_seen_val', img_val),\
                ('pred_val', pred_val * 1.0 / self.n_cls),\
                ('gth_val', gth_val * 1.0 / self.n_cls)
                ])
        return ret_visuals

    def get_current_visuals_tr(self):
        img_tr  = t2n( to01(self.input_img_3copy[: self._nb_current].data, True))
        pred_tr = t2n( torch.argmax(self.seg_tr.data, dim =1, keepdim = True )  )
        gth_tr  = t2n(self.input_mask.data )

        ret_visuals = OrderedDict([\
                ('img_seen_tr', img_tr),\
                ('seg_tr', (pred_tr + 0.01) * 1.0 / (self.n_cls + 0.01 )),\
                ('gth_seen_tr', (gth_tr + 0.01) * 1.0 / (self.n_cls + 0.01 )), \
                ])

        if hasattr(self, 'blend_mask'):
            blend_tr  = t2n(self.blend_mask[:,0:1, ...] )
            ret_visuals['blendmask'] = (blend_tr + 0.01) * 1.0 / (1 + 0.01 )

        return ret_visuals

    def plot_image_in_tb(self, writer, result_dict):
        for key, img in result_dict.items():
            writer.add_image(key, img)

    def track_scalar_in_tb(self, writer, result_dict, which_iter):
        for key, val in result_dict.items():
            writer.add_scalar(key, val, which_iter)

    # NOTE: remeber to modify this when expanding the model if more than one network component need to be stored
    def save(self, label):
        self.save_network(self.netSeg, 'Seg', label, self.gpu_ids)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    if x.is_cuda:
        x = x.data.cpu()
    else:
        x = x.data

    return np.float32(x.numpy())

def to01(x, by_channel = False):
    if not by_channel:
        out = (x - x.min()) / (x.max() - x.min())
    else:
        nb, nc, nh, nw = x.shape
        xmin = x.view(nb, nc, -1).min(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
        xmax = x.view(nb, nc, -1).max(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
        out = (x - xmin + 1e-5) / (xmax - xmin + 1e-5)
    return out

