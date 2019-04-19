import argparse
import comet_ml
import torch
import utils

from agent import Agent
from environment import Environment

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')
parser.add_argument('--detected_class', type=int, default=1, help='class that the detector will learn to locate')
parser.add_argument('--data', type=str, default='/home/jg/MILA/COMP767-Reinforcement_Learning/COMP767/project/data/VOCtrainval_06-Nov-2007/VOCdevkit',help='location of the data')
parser.add_argument('--batch_size', type=int, default=64, help='size of one minibatch')
parser.add_argument('--discount_rate', type=float, default=0.99, help='discount_rate')
parser.add_argument('--eps_start', type=float, default=1.0, help='epsilon start')
parser.add_argument('--eps_end', type=float, default=0.1, help='epsilon end')
parser.add_argument('--eps_decay', type=int, default=5, help='epsilon decay')
parser.add_argument('--target_update', type=int, default=10, help='number of steps between update of the target q-network')
parser.add_argument('--epoch', type=int, default=15, help='number of epoch')
parser.add_argument('--memory_size', type=int, default=10000, help='memory_size')
parser.add_argument('--year', type=int, default=2007,  help='pascal voc dataset year')
parser.add_argument('--history_action', type=int, default=10, help='number of past action in a state history')
parser.add_argument('--save_dir', type=str, default='', help='sace directory')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

print("\n########## Setting Up Experiment ######################")
experiment_path, experiment = utils.setup_run_folder(args)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = utils.get_device()

###############################################################################
#
# MODEL SETUP
#
###############################################################################

env = Environment(root=args.data,
                  detected_class=args.detected_class,
                  year=args.year)
agent = Agent(env,
              target_update=args.target_update,
              discout_rate=args.discount_rate,
              eps_start=args.eps_start,
              eps_end=args.eps_end,
              eps_decay=args.eps_decay,
              batch_size=args.batch_size,
              memory_size=args.memory_size,
              n_past_action_to_remember=args.history_action,
              save_path=experiment_path)

###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################
print("########## Running Main Loop ##########################")
results = agent.train(env, nb_epoch=args.epoch)
