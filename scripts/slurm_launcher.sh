#!/bin/bash

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=d:
LONGOPTS=odir:,bck:,useshade:,label:,beta_1:,beta_2:,expn:,ddir:,dsetname:,ptnc:,wd:,dpt:,ewd:,edpt:,awd:,adpt:,zdim:,seed:,num_epochs:,batch_size:,usebatchnorm:,sattr:,wdbn:,arch:,fair_coeff:,aud_steps:,adv_coeff:,gamma:,alpha:,green_yellow:,egr:,ogr:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly

! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

# ! PARSED=$(getopt --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

# DATASET ARGS
bck="blue-red"
useshade="True"
label="even-odd"
beta_1="0.5"
beta_2="0.5"
egr=0.5
ogr=0.5
green_yellow="True"
sattr="bck"

# DATASET ARGS END
homedir="/home/charanr/"
repo="$homedir/FairDeepLearning"
oroot="/scratch/charanr/fairness-project"
outputdir="$oroot/output"
mkdir -p $outputdir

# EXPERIMENT ARGS START
exp_name="slurm-demo"
data_dir="data"
dataset_name="clr-mnist"
patience=5
width=32
depth=2
awidth=32
adepth=2
ewidth=32
edepth=2
seed=3
arch="ffvae"
zdim=16
num_epochs=150
batch_size=64
odir="$outputdir"
usebatchnorm="True"
wandb_name="deepfairness-0"

fair_coeff=0.1
aud_steps=2
adv_coeff=0.1
gamma=10
alpha=10
# EXPERIMENT ARGS END

while true; do
    case "$1" in
        --odir)
            odir="$2"
            mkdir -p $odir
            shift 2
            ;;
        --bck)
            bck="$2"
            shift 2
            ;;
        --useshade)
            useshade="$2"
            shift 2
            ;;
        --label)
            label="$2"
            shift 2
            ;;
        --beta_1)
            beta_1="$2"
            shift 2
            ;;
        --beta_2)
            beta_2="$2"
            shift 2
            ;;
        --expn)
            exp_name="$2"
            shift 2
            ;;
        --ddir)
            data_dir="$2"
            shift 2
            ;;
        --dsetname)
            dataset_name="$2"
            shift 2
            ;;
        --ptnc)
            patience="$2"
            shift 2
            ;;
        --wd)
            width="$2"
            shift 2
            ;;
        --dpt)
            depth="$2"
            shift 2
            ;;
        --ewd)
            ewidth="$2"
            shift 2
            ;;
        --edpt)
            edepth="$2"
            shift 2
            ;;
        --awd)
            awidth="$2"
            shift 2
            ;;
        --adpt)
            adepth="$2"
            shift 2
            ;;
        --zdim)
            zdim="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --usebatchnorm)
            usebatchnorm="$2"
            shift 2
            ;;
        --sattr)
            sattr="$2"
            shift 2
            ;;
        --wdbn)
            wandb_name="$2"
            shift 2
            ;;
        --arch)
            arch="$2"
            shift 2
            ;;
        --fair_coeff)
            fair_coeff="$2"
            shift 2
            ;;
        --aud_steps)
            aud_steps="$2"
            shift 2
            ;;
        --adv_coeff)
            adv_coeff="$2"
            shift 2
            ;;
        --gamma)
            gamma="$2"
            shift 2
            ;;
        --alpha)
            alpha="$2"
            shift 2
            ;;
        --green_yellow)
            green_yellow="$2"
            shift 2
            ;;
        --egr)
            egr="$2"
            shift 2
            ;;
        --ogr)
            ogr="$2"
            shift 2
            ;;
        -d)
            dummy_var="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

echo "Creating output directory $odir and copying MNIST data into $data_dir"
mkdir -p $odir $data_dir

echo "Starting training"
set -o xtrace
python -u train/execute.py --wandb_name $wandb_name\
    --name $exp_name\
    --arch $arch\
    --data-dir $data_dir\
    --data $dataset_name\
    --output-dir $odir\
    --epochs $num_epochs\
    --patience $patience\
    --cwidths $width\
    --cdepth $depth\
    --awidths $awidth\
    --adepth $adepth\
    --ewidths $ewidth\
    --edepth $edepth\
    --bck $bck\
    --label_type $label\
    --beta_1 $beta_1\
    --beta_2 $beta_2\
    --egr $egr\
    --ogr $ogr\
    --sensattr $sattr\
    --seed $seed\
    --zdim $zdim\
    --fair_coeff $fair_coeff\
    --adv_coeff $adv_coeff\
    --gamma $gamma\
    --alpha $alpha\
    --ifwandb True

set +o xtrace
