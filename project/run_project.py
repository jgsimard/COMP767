import argparse
import os
import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import gym_project.envs.object_localization as object_localization
from agent import Agent


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')
parser.add_argument('--detected_class', type=int, default=1,
                    help='class that the detector will learn to locate')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of one minibatch')
parser.add_argument('--discount_rate', type=float, default=0.99,
                    help='discount_rate')
parser.add_argument('--eps_start', type=float, default=1.0,
                    help='epsilon start')
parser.add_argument('--eps_end', type=float, default=0.1,
                    help='epsilon end')
parser.add_argument('--eps_decay', type=int, default=5,
                    help='epsilon decay')
parser.add_argument('--target_update', type=int, default=10,
                    help='number of steps between update of the target q-network')
parser.add_argument('--epoch', type=int, default=15,
                    help='number of epoch')
parser.add_argument('--memory_size', type=int, default=10000,
                    help='memory_size')
parser.add_argument('--history_action', type=int, default=10,
                    help='number of past action in a state history')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')


args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

print("\n########## Setting Up Experiment ######################")
base_path = os.path.join(f"detected_class={args.detected_class}")

def get_directory_name_with_number(base_path):
    i = 0
    while os.path.exists(base_path + "_" + str(i)):
        i += 1
    return base_path + "_" + str(i)
experiment_path = get_directory_name_with_number(base_path)

os.mkdir(experiment_path)
print("\nPutting log in %s" % experiment_path)
argsdict['save_dir'] = experiment_path
with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key + '    ' + str(argsdict[key]) + '\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("Using the CPU")
    device = torch.device("cpu")





env = object_localization.ProjectEnv(root=args.data,
                                     detected_class = args.detected_class)
agent = Agent(env,
              target_update=args.target_update,
              discout_rate=args.discount_rate,
              eps_start=args.eps_start,
              eps_end=args.eps_end,
              eps_decay=args.eps_decay,
              batch_size=args.batch_size,
              memory_size=args.memory_size,
              n_past_action_to_remember=args.history_action,
              save_path = experiment_path)
agent.train(env, nb_epoch=args.epoch)
