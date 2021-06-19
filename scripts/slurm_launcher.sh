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
LONGOPTS=ddir:,odir:,dsetname:,beta_1:,beta_2:,green_yellow:,egr:,ogr:,sattr:,edpt:,ewd:,adpt:,awd:,cdpt:,cwd:,zdim:,seed:,arch:,fair_coeff:,aud_steps:,adv_coeff:,gamma:,alpha:,replicate:,num_epochs:,ptnc:,wdbn:

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


# Below are just default args, 
# args needed to run are taken from launcher.py files in scripts/ folder

# FOLDERS
ddir="./data"
odir="/scratch/charanr/fairness-project/output/"
mkdir -p $odir $ddir

# DATASET ARGS
dsetname="clr-mnist"
beta_1="0.5"
beta_2="0.5"
green_yellow="True"
egr=0.5
ogr=0.5
sattr="bck"

# ARCHITECTURE ARGS
edpt=32
ewd=2
adpt=32
awd=2
cdpt=32
cwd=2
zdim=16
seed=3

# MODEL ARGS
arch="ffvae"
fair_coeff=0.1
aud_steps=2
adv_coeff=0.1
gamma=10
alpha=10

# OTHERS
replicate=1
num_epochs=150
ptnc=5
wdbn="deepfairness-0"
exp_name="slurm-demo"


while true; do
    case "$1" in
        --ddir)
            ddir="$2"
            shift 2
            ;;
        --odir)
            odir="$2"
            mkdir -p $odir
            shift 2
            ;;
        --dsetname)
            dsetname="$2"
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
        --sattr)
            sattr="$2"
            shift 2
            ;;
        --edpt)
            edpt="$2"
            shift 2
            ;;
        --ewd)
            ewd="$2"
            shift 2
            ;;
        --adpt)
            adpt="$2"
            shift 2
            ;;
        --awd)
            awd="$2"
            shift 2
            ;;
        --cdpt)
            cdpt="$2"
            shift 2
            ;;
        --cwd)
            cwd="$2"
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
        --replicate)
            replicate="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --ptnc)
            ptnc="$2"
            shift 2
            ;;
        --wdbn)
            wdbn="$2"
            shift 2
            ;;
        --exp_name)
            exp_name="$2"
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

echo "Starting training"
set -o xtrace
python -u train/execute.py \
    --data-dir $ddir\
    --output-dir $odir\
    --data $dsetname\
    --beta_1 $beta_1\
    --beta_2 $beta_2\
    --green_yellow $green_yellow\
    --egr $egr\
    --ogr $ogr\
    --sensattr $sattr\
    --edepth $edpt\
    --ewidths $ewd\
    --adepth $adpt\
    --awidths $awd\
    --cdepth $cdpt\
    --cwidths $cwd\
    --zdim $zdim\
    --seed $seed\
    --arch $arch\
    --fair_coeff $fair_coeff\
    --aud_steps $aud_steps\
    --adv_coeff $adv_coeff\
    --gamma $gamma\
    --alpha $alpha\
    --replicate $replicate\
    --epochs $num_epochs\
    --patience $ptnc\
    --wandb_name $wdbn\
    --name $exp_name\
    --ifwandb True

set +o xtrace
