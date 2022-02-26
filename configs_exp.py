# config files for experiments
import argparse
import os
from my_utils import util
import itertools
import glob

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('CAUSALDG')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './my_utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))

for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    exp_type = 'ginipa'
    name = 'myexp'
    phase = 'train'
    get_features = False
    batchSize = 20
    fineSize = 192
    gpu_ids = [0]
    nThreads = 4
    load_dir = './checkpoints'
    checkpoints_dir = './checkpoints'
    reload_model_fid = ''

    # display configs, using tensorboardX
    # display_server = "http://localhost"
    # display_port = 8097
    display_freq = 2000

    # validation configs
    print_freq = 2000
    validation_freq = 2000
    save_epoch_freq = 1000
    infer_epoch_freq = 250
    save_prediction = False

    ###### training configs ######
    data_name = 'ABDONINAL' # change to ABDOMINAL or PROSTATE
    tr_domain = 'SABSCT' # for prostate, use A B C D E or F
    te_domain = 'CHAOST2'
    exclu_domain = None # only for prostate for 1vs5 experiments, will override te_domain
    model = 'efficient_b2_unet'
    eval_fold = 0 # not in use
    nclass = 4

    continue_train = False
    epoch_count = 1
    which_epoch = 'latest'
    niter = 50
    niter_decay = 1950 # epoches for lr decay.

    optimizer = 'adam'
    beta1 = 0.5
    lr = 0.0003
    adam_weight_decay = 0.00003

    lr_policy = 'lambda' # step/ plateau
    lr_decay_iters = 50
    early_stop_epoch = -1 # some baseline method might needs early stop/less overall iterations/smaller lr. Use -1 when disable it

    lambda_Seg = 1.0
    lambda_wce = 1.0
    lambda_dice = 1.0

    lambda_consist = 10.0 # Xu et al.

    init_type = 'normal'

    # config for gin
    nb_gin = 20
    gin_out_nc = 3 # fit into the network
    gin_n_interm_ch = 2
    gin_nlayer = 4
    gin_norm = 'frob'

    # config for ipa correlation maps
    blend_grid_size = 24 # 24*2=48, 1/4 of image size
    blend_epsilon = 0.3

    # consistency
    consist_type = 'kld'

    # specific for augmentation strength. Use a rather strong photometric baseline to ensure fairness
    aug_mode = 'strongbright'

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["name"]}'
    observer = FileStorageObserver.create(os.path.join(config['checkpoints_dir'], exp_name))
    ex.observers.append(observer)
    return config
