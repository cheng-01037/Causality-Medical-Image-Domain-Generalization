# Dataloader for abdominal images
import glob
import numpy as np
import dataloaders.niftiio as nio
import dataloaders.transform_utils as trans
import torch
import random
import os
import copy
import platform
import json
import torch.utils.data as torch_data
import math
import itertools
from .abd_dataset_utils import get_normalize_op
from pdb import set_trace

hostname = platform.node()
# folder for datasets
BASEDIR = './data/abdominal/'
print(f'Running on machine {hostname}, using dataset from {BASEDIR}')
LABEL_NAME = ["bg", "liver", "rk", "lk", "spleen"]

class AbdominalDataset(torch_data.Dataset):
    def __init__(self,  mode, transforms, base_dir, domains: list,  idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, extern_norm_fn = None):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        super(AbdominalDataset, self).__init__()
        self.transforms=transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.domains = domains
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct

        self.img_pids = {}
        for _domain in self.domains: # load file names
            self.img_pids[_domain] = sorted([ fid.split("_")[-1].split(".nii.gz")[0] for fid in glob.glob(self._base_dir + "/" +  _domain  + "/processed/image_*.nii.gz") ], key = lambda x: int(x))

        self.scan_ids = self.__get_scanids(mode, idx_pct) # train val test split in terms of patient ids

        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids) # image files names according to self.scan_ids
        if self.is_train:
            self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        elif mode == 'test': # Source domain test
            self.pid_curr_load = self.scan_ids
        elif mode == 'test_all': # Choose this when being used as a target domain testing set. Liu et al.
            self.pid_curr_load = self.scan_ids
        if extern_norm_fn is None:
            self.normalize_op = get_normalize_op(self.domains[0], [ itm['img_fid'] for _, itm in self.sample_list[self.domains[0]].items() ])
            print(f'{self.phase}_{self.domains[0]}: Using fold data statistics for normalization')
        else:
            assert len(self.domains) == 1, 'for now we only support one normalization function for the entire set'
            self.normalize_op = extern_norm_fn

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids {self.pid_curr_load}')

        # load to memory
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset) # 2D

    def __get_scanids(self, mode, idx_pct):
        """
        index by domains given that we might need to load multi-domain data
        idx_pct: [0.7 0.1 0.2] for train val test. with order te val tr
        """
        tr_ids      = {}
        val_ids     = {}
        te_ids      = {}
        te_all_ids  = {}

        for _domain in self.domains:
            dset_size   = len(self.img_pids[_domain])
            tr_size     = round(dset_size * idx_pct[0])
            val_size    = math.floor(dset_size * idx_pct[1])
            te_size     = dset_size - tr_size - val_size

            te_ids[_domain]     = self.img_pids[_domain][: te_size]
            val_ids[_domain]    = self.img_pids[_domain][te_size: te_size + val_size]
            tr_ids[_domain]     = self.img_pids[_domain][te_size + val_size: ]
            te_all_ids[_domain] = list(itertools.chain(tr_ids[_domain], te_ids[_domain], val_ids[_domain]   ))

        if self.phase == 'train':
            return tr_ids
        elif self.phase == 'val':
            return val_ids
        elif self.phase == 'test':
            return te_ids
        elif self.phase == 'test_all':
            return te_all_ids

    def __search_samples(self, scan_ids):
        """search for filenames for images and masks
        """
        out_list = {}
        for _domain, id_list in scan_ids.items():
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}

                _img_fid = os.path.join(self._base_dir, _domain , 'processed'  ,f'image_{curr_id}.nii.gz')
                _lb_fid  = os.path.join(self._base_dir, _domain , 'processed', f'label_{curr_id}.nii.gz')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                out_list[_domain][str(curr_id)] = curr_dict

        return out_list


    def __read_dataset(self):
        """
        Read the dataset into memory
        """

        out_list = []
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for _domain, _sample_list in self.sample_list.items():
            for scan_id, itm in _sample_list.items():
                if scan_id not in self.pid_curr_load[_domain]:
                    continue

                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img = np.float32(img)
                img = self.normalize_op(img)

                lb = nio.read_nii_bysitk(itm["lbs_fid"])
                lb = np.float32(lb)

                img     = np.transpose(img, (1,2,0))
                lb      = np.transpose(lb, (1,2,0))

                assert img.shape[-1] == lb.shape[-1]

                # now start writing everthing in
                # write the beginning frame
                out_list.append( {"img": img[..., 0: 1],
                               "lb":lb[..., 0: 0 + 1],
                               "is_start": True,
                               "is_end": False,
                               "domain": _domain,
                               "nframe": img.shape[-1],
                               "scan_id": _domain + "_" + scan_id,
                               "z_id":0})
                glb_idx += 1

                for ii in range(1, img.shape[-1] - 1):
                    out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii + 1],
                               "is_start": False,
                               "is_end": False,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii
                               })
                    glb_idx += 1

                ii += 1 # last frame, note the is_end flag
                out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii+ 1],
                               "is_start": False,
                               "is_end": True,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii
                               })
                glb_idx += 1

        return out_list


    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        if self.is_train is True:
            comp = np.concatenate( [curr_dict["img"], curr_dict["lb"]], axis = -1 )
            if self.transforms:
                img, lb = self.transforms(comp, c_img = 1, c_label = 1, nclass = self.nclass, is_train = self.is_train, use_onehot = False)
        else:
            img = curr_dict['img']
            lb = curr_dict['lb']

        img = np.float32(img)
        lb = np.float32(lb)

        img = np.transpose(img, (2, 0, 1))
        lb  = np.transpose(lb, (2, 0, 1))

        img = torch.from_numpy( img )
        lb  = torch.from_numpy( lb )

        if self.tile_z_dim > 1:
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3

        is_start    = curr_dict["is_start"]
        is_end      = curr_dict["is_end"]
        nframe      = np.int32(curr_dict["nframe"])
        scan_id     = curr_dict["scan_id"]
        z_id        = curr_dict["z_id"]

        sample = {"img": img,
                "lb":lb,
                "is_start": is_start,
                "is_end": is_end,
                "nframe": nframe,
                "scan_id": scan_id,
                "z_id": z_id
                }
        return sample

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        return len(self.actual_dataset)

tr_func  = trans.transform_with_label(trans.tr_aug)

def get_training(modality, idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3):
    return AbdominalDataset(idx_pct = idx_pct,\
        mode = 'train',\
        domains = modality,\
        transforms = tr_func,\
        base_dir = BASEDIR,\
        extern_norm_fn = None, # normalization function is decided by domain
        tile_z_dim = tile_z_dim)

def get_validation(modality, norm_func, idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3):
     return AbdominalDataset(idx_pct = idx_pct,\
        mode = 'val',\
        transforms = None,\
        domains = modality,\
        base_dir = BASEDIR,\
        extern_norm_fn = norm_func,\
        tile_z_dim = tile_z_dim)

def get_test(modality, norm_func, tile_z_dim = 3, idx_pct = [0.7, 0.1, 0.2]):
     return AbdominalDataset(idx_pct = idx_pct,\
        mode = 'test',\
        transforms = None,\
        domains = modality,\
        extern_norm_fn = norm_func,\
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)

def get_test_all(modality, norm_func, tile_z_dim = 3, idx_pct = [0.7, 0.1, 0.2]):
     return AbdominalDataset(idx_pct = idx_pct,\
        mode = 'test_all',\
        transforms = None,\
        domains = modality,\
        extern_norm_fn = norm_func,\
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)

