

import logging
import os

import numpy as np
import gym
import minerl

import coloredlogs

coloredlogs.install(logging.DEBUG)

# The dataset and trained models are available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def main():

    ###############################
    # PHASE 1: TRAINING
    # Train Find Cave Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_findcave offline=false human_prefs=false exp_name='' vpt_model=2 phase=1")
    # Train Make Waterfall Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_waterfall offline=false human_prefs=false exp_name='' vpt_model=2 phase=1")
    # Train Create Village Animal Pen Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_penanimals offline=false human_prefs=false exp_name='' vpt_model=2 phase=1")
    # Train Build Village House Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_buildhouse offline=false human_prefs=false exp_name='' vpt_model=2 phase=1")

    ###############################
    # PHASE 2: TRAINING
    # Train Find Cave Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_findcave offline=false human_prefs=true exp_name='' vpt_model=2 phase=2")
    # Train Make Waterfall Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_waterfall offline=false human_prefs=true exp_name='' vpt_model=2 phase=2")
    # Train Create Village Animal Pen Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_penanimals offline=false human_prefs=true exp_name='' vpt_model=2 phase=2")
    # Train Build Village House Task
    os.system("python iq_learn/train_iq.py agent=sac method=iq env=basalt_penanimals offline=false human_prefs=true exp_name='' vpt_model=2 phase=2")   


if __name__ == "__main__":
    main()