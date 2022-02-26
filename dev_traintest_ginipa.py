# Main script for training and testing
import time
import shutil
import SimpleITK as sitk
import torch
import numpy as np
import os
import dataloaders.niftiio as nio
import pickle as pkl

from models import create_forward
from my_utils.util import AttrDict, worker_init_fn

from torch.utils.data import DataLoader
from pdb import set_trace
from tqdm import tqdm
from configs_exp import ex # configuration files
from tensorboardX import SummaryWriter

def prediction_wrapper(model, test_loader, opt, epoch, label_name, mode = 'base', save_prediction = False):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['lb'].shape[0] == 1 # enforce a batchsize of 1

            test_input = {
                    'img': batch['img'],
                    'lb': batch['lb']
                    }

            model.set_input(test_input)
            gth, pred = model.get_segmentation_gpu(raw_logits = False)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['img'][0, 1,...].numpy()
            slice_idx += 1
            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                if opt.phase == 'test':
                    recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name ), model, label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper(vol_list, nclass, model, label_name):
    """
    Evaluatation and arrange predictions
    """
    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    conf_mat_list = [] # confusion matrices
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],
                    'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices = model.ScoreDiceEval(torch.unsqueeze(pred_, 1), gth_, dense_input = True).cpu().numpy() # this includes the background class
        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f} \n, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc

    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)

    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain

    return error_dict, dsc_table, domain_names

@ex.automain
def main(_run, _config, _log):
    # configs for sacred
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        _config['run_dir'] = _run.observers[0].dir
        _config['snapshot_dir'] = f'{_run.observers[0].dir}/snapshots'
        _config['pred_dir'] = f'{_run.observers[0].dir}/interm_preds'

    tbfile_dir = os.path.join(  _run.observers[0].dir, 'tboard_file' ); os.mkdir(tbfile_dir)
    tb_writer = SummaryWriter( tbfile_dir  )

    opt = AttrDict(_config)

    if opt.data_name == 'ABDOMINAL':
        import dataloaders.AbdominalDataset as ABD
        if not isinstance(opt.tr_domain, list):
            opt.tr_domain = [opt.tr_domain]
            opt.te_domain = [opt.te_domain]

        train_set       = ABD.get_training(modality = opt.tr_domain )
        val_source_set  = ABD.get_validation(modality = opt.tr_domain, norm_func = train_set.normalize_op) # not really using it as there is no validation for target
        if opt.te_domain[0] == opt.tr_domain[0]:
            test_set        = ABD.get_test(modality = opt.te_domain, norm_func = train_set.normalize_op) # if same domain, then use the normalize op from the source
            test_source_set = test_set
        else:
            test_set        = ABD.get_test_all(modality = opt.te_domain, norm_func = None)
            test_source_set        = ABD.get_test(modality = opt.tr_domain, norm_func = train_set.normalize_op)
        label_name          = ABD.LABEL_NAME

    elif opt.data_name == 'PROSTATE':
        import dataloaders.ProstateDataset as PROS
        train_set       = PROS.get_training(modality = opt.tr_domain )
        val_source_set  = PROS.get_validation(modality = opt.tr_domain)
        if opt.exclu_domain is not None:
            test_set        = PROS.get_test_exclu(tr_modality = opt.tr_domain)
        else:
            test_set        = PROS.get_test(modality = opt.te_domain)
        test_source_set     = PROS.get_test(modality = opt.tr_domain)
        label_name      = PROS.LABEL_NAME

    else:
        raise NotImplementedError(opt.data_name)

    print(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    train_loader = DataLoader(dataset = train_set, num_workers = opt.nThreads,\
            batch_size = opt.batchSize, shuffle = True, drop_last = True, worker_init_fn = worker_init_fn, pin_memory = True)

    val_loader = iter(DataLoader(dataset = val_source_set, num_workers = 1,\
            batch_size = 1, shuffle = True, drop_last = True, pin_memory = True))

    test_loader = DataLoader(dataset = test_set, num_workers = 1,\
            batch_size = 1, shuffle = False, pin_memory = True)

    test_src_loader = DataLoader(dataset = test_source_set, num_workers = 1,\
            batch_size = 1, shuffle = False, pin_memory = True)

    if opt.exp_type == 'gin' or opt.exp_type == 'ginipa':
        model = create_forward(opt)
    elif opt.exp_type == 'erm':
        raise NotImplementedError # coming soon
    else:
        raise NotImplementedError(opt.exp_type)
    total_steps = 0
    if opt.phase == 'test':
        opt.epoch_count = 0
        opt.niter = 0
        opt.niter_decay = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        np.random.seed()
        if opt.phase == 'train':
            for i, train_batch in tqdm(enumerate(train_loader), total = train_loader.dataset.size // opt.batchSize - 1):

                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batchSize
                epoch_iter  += opt.batchSize

                # avoid batchsize issues caused by fetching last training batch
                if train_batch["img"].shape[0] != opt.batchSize:
                    continue

                train_input = {'img': train_batch["img"],
                               'lb': train_batch["lb"]}

                ## run a training step
                model.set_input_aug_sup(train_input)
                model.optimize_parameters()

                ## display training losses
                if total_steps % opt.display_freq == 0:
                    tr_viz = model.get_current_visuals_tr()
                    model.plot_image_in_tb(tb_writer, tr_viz)

                if total_steps % opt.print_freq == 0:
                    tr_error = model.get_current_errors_tr()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    model.track_scalar_in_tb(tb_writer, tr_error, total_steps)

                ## run and display validation losses
                if total_steps % opt.validation_freq == 0:
                    with torch.no_grad():
                        try:
                            val_batch = next(val_loader) # FIXME: use a nicer way
                        except:
                            val_loader = iter(DataLoader(dataset = val_source_set, num_workers = opt.nThreads,\
                                    batch_size = 1, drop_last = True, shuffle = True))
                            val_batch = next(val_loader)

                        val_input = {
                                'img': val_batch["img"],
                                'lb':  val_batch["lb"]
                                }
                        model.set_input(val_input)
                        model.validate()
                        val_errors = model.get_current_errors_val()

                    if total_steps % opt.display_freq == 0:
                        val_viz = model.get_current_visuals_val()
                        model.plot_image_in_tb(tb_writer, val_viz)

                        val_errors = model.get_current_errors_val()
                        model.track_scalar_in_tb(tb_writer, val_errors, total_steps)

                iter_data_time = time.time()

        ## test
        if (epoch % opt.infer_epoch_freq == 0):
            t0  = time.time()
            print('infering the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            with torch.no_grad():
                print(f'Starting inferring ... ')
                preds, dsc_table, error_dict, domain_list = prediction_wrapper(model, test_loader, opt, epoch, label_name, save_prediction = _config["save_prediction"])
                _run.log_scalar('rawDiceTarget', dsc_table.tolist())
                _run.log_scalar('meanDiceTarget', error_dict['overall'] )
                _run.log_scalar('meanDiceAvgTargetDomains', error_dict['overall_by_domain'] ) # for prostate dataset
                for _dm in domain_list:
                    _run.log_scalar(f'meanDice_{_dm}', error_dict[f'domain_{_dm}_overall'])
                    _run.log_scalar(f'rawDice_{_dm}', error_dict[f'domain_{_dm}_table'].tolist())

                print('test for source domain as a reference')
                _, dsc_table, error_dict, _ = prediction_wrapper(model, test_src_loader, opt, epoch, label_name, save_prediction = _config["save_prediction"])
                _run.log_scalar('source_rawDice', dsc_table.tolist())
                _run.log_scalar('source_meanDice', error_dict['overall'] )

                if _config["save_prediction"]:
                    for scan_id, comp in preds.items():
                        _pred = comp['pred']

                        itk_pred = sitk.GetImageFromArray(_pred.cpu().numpy())
                        itk_pred.SetSpacing(  test_set.info_by_scan[scan_id]["spacing"] )
                        itk_pred.SetOrigin(   test_set.info_by_scan[scan_id]["origin"] )
                        itk_pred.SetDirection(test_set.info_by_scan[scan_id]["direction"] )

                        fid = os.path.join(model.pred_dir, f'pred_{scan_id}_epoch_{epoch}.nii.gz')
                        sitk.WriteImage(itk_pred, fid, True)
                        _log.info(f'# {fid} has been saved #')

                t1 = time.time()
                print("End of model inference, which takes {} seconds".format(t1 - t0))

        if opt.phase == 'test':
            return

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        if epoch == opt.early_stop_epoch:
            return

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

