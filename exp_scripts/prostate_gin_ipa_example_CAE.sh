# GIN and IPA for prostate images
SCRIPT=dev_traintest_ginipa.py
GPUID1=0
NUM_WORKER=8
MODEL='efficient_b2_unet'
CPT='prostate_ginipa_example'

# visualization
PRINT_FREQ=50000
VAL_FREQ=50000
TEST_EPOCH=50
EXP_TYPE='ginipa'

BSIZE=20
NL_GIN=4
N_INTERM=2

LAMBDA_WCE=1.0 # not using weights, actually standard multi-class ce
LAMBDA_DICE=1.0
LAMBDA_CONSIST=10.0 # Xu et al.

SAVE_EPOCH=1000
SAVE_PRED=False # save predictions or not

DATASET='PROSTATE'
CHECKPOINTS_DIR="./my_exps/$DATASET"
NITER=50
NITER_DECAY=1950
IMG_SIZE=192

OPTM_TYPE='adam'
LR=0.0003
ADAM_L2=0.00003
TE_DOMAIN="NA" # will be override by exclu_domain

# blender config
BLEND_GRID_SIZE=24

ALL_TRS=("C" "A" "E") # repeat the experiment for different source domains. For the full set of experiments, use A B C D E F
NCLASS=2

# KL term
CONSIST_TYPE='kld'

for TR_DOMAIN in "${ALL_TRS[@]}"
do
    set -ex
    export CUDA_VISIBLE_DEVICES=$GPUID1

    NAME=${CPT}_tr${TR_DOMAIN}_exclude${TR_DOMAIN}_${MODEL}
    LOAD_DIR=$NAME

    python3 $SCRIPT with exp_type=$EXP_TYPE\
        name=$NAME\
        model=$MODEL\
        nThreads=$NUM_WORKER\
        print_freq=$PRINT_FREQ\
        validation_freq=$VAL_FREQ\
        batchSize=$BSIZE\
        lambda_wce=$LAMBDA_WCE\
        lambda_dice=$LAMBDA_DICE\
        save_epoch_freq=$SAVE_EPOCH\
        load_dir=$LOAD_DIR\
        infer_epoch_freq=$TEST_EPOCH\
        niter=$NITER\
        niter_decay=$NITER_DECAY\
        fineSize=$IMG_SIZE\
        lr=$LR\
        adam_weight_decay=$ADAM_L2\
        data_name=$DATASET\
        nclass=$NCLASS\
        tr_domain=$TR_DOMAIN\
        te_domain=$TE_DOMAIN\
        optimizer=$OPTM_TYPE\
        save_prediction=$SAVE_PRED\
        lambda_consist=$LAMBDA_CONSIST\
        blend_grid_size=$BLEND_GRID_SIZE\
        exclu_domain=$TR_DOMAIN\
        consist_type=$CONSIST_TYPE\
        display_freq=$PRINT_FREQ\
        gin_nlayer=$NL_GIN\
        gin_n_interm_ch=$N_INTERM
done
