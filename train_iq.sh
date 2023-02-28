#!/bin/bash
## KAIROS team training script for the BASALT 2022 competition
# Usage: -e specifies environment, -f specifies offline mode, -h to collect human prefs, -x is experiment 
#        name, -m (specify 1, 2, or 3) is which VPT model to use (1x, 2x, 3x)
#        -p learning_phase
# ./train_iq.sh -e env_name -f -m 3 -x experiment_name -p learning_phase
# For example, cave online: ./train_iq.sh cave -m 3 -x test_3x_model
#              waterfall offline: ./train_iq.sh waterfall -f

# Use full logs 
export HYDRA_FULL_ERROR=1

# Set default values for all (optional) inputs:
ENV_NAME="cave"
OFFLINE=false
EXP_NAME=""
VPT_MODEL=2  # 2x model by default
HUMAN_PREFS=false
PHASE=1

print_usage() {
  printf "Usage (use -f for offline, -h to collect human prefs): ./train_iq.sh -e env_name -m vpt_model_num -x exp_name -f -h -p 2"
}

# Get env name and whether to use offline mode
while getopts e:x:m:p:fh flag
do
    case "${flag}" in
        e) ENV_NAME=${OPTARG};;
        f) OFFLINE='true';;
        x) EXP_NAME=${OPTARG};;
        m) VPT_MODEL=${OPTARG};;
        p) PHASE=${OPTARG};;
        h) HUMAN_PREFS='true';;
        *) print_usage
           exit 1 ;;
    esac
done

echo 'Training KAIROS agent for '$ENV_NAME'. Using offline mode: '$OFFLINE'. Collecting human prefs: '$HUMAN_PREFS'. Phase: '$PHASE

# Trains OpenAI VPT models using IQ-Learn algorithm
if [ $ENV_NAME = "cave" ]
then
    python iq_learn/train_iq.py agent=sac method=iq env=basalt_findcave offline=$OFFLINE human_prefs=$HUMAN_PREFS exp_name=$EXP_NAME vpt_model=$VPT_MODEL phase=$PHASE
elif [ $ENV_NAME = "waterfall" ]
then
    python iq_learn/train_iq.py agent=sac method=iq env=basalt_waterfall offline=$OFFLINE human_prefs=$HUMAN_PREFS exp_name=$EXP_NAME vpt_model=$VPT_MODEL phase=$PHASE
elif [ $ENV_NAME = "animal" ]
then
    python iq_learn/train_iq.py agent=sac method=iq env=basalt_penanimals offline=$OFFLINE human_prefs=$HUMAN_PREFS exp_name=$EXP_NAME vpt_model=$VPT_MODEL phase=$PHASE
elif [ $ENV_NAME = "house" ]
then
    python iq_learn/train_iq.py agent=sac method=iq env=basalt_buildhouse offline=$OFFLINE human_prefs=$HUMAN_PREFS exp_name=$EXP_NAME vpt_model=$VPT_MODEL phase=$PHASE
else
    echo 'ERROR: Invalid environment name.'
    exit 1
fi
