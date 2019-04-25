import argparse
import comet_ml

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from environment import Environment
from evaluator import Evaluator, transform
from tqdm import tqdm

from sklearn.metrics import average_precision_score

import utils
from agent import Agent

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################
parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')
parser.add_argument('--detected_class', type=int, default=1, help='class that the detector will learn to locate')
parser.add_argument('--data', type=str,
                    default='./data/VOCtrainval_06-Nov-2007/VOCdevkit',
                    help='location of the data')
parser.add_argument('--batch_size', type=int, default=64, help='size of one minibatch')
parser.add_argument('--discount_rate', type=float, default=0.9, help='discount_rate')
parser.add_argument('--eps_start', type=float, default=1.0, help='epsilon start')
parser.add_argument('--eps_end', type=float, default=0.1, help='epsilon end')
parser.add_argument('--eps_decay', type=int, default=5, help='epsilon decay')
parser.add_argument('--target_update', type=int, default=10,
                    help='number of steps between update of the target q-network')
parser.add_argument('--nb_epochs', type=int, default=30, help='number of epoch')
parser.add_argument('--memory_size', type=int, default=100000, help='memory_size')
parser.add_argument('--year', type=int, default=2007, help='pascal voc dataset year')
parser.add_argument('--history_action', type=int, default=10, help='number of past action in a state history')
parser.add_argument('--save_dir', type=str, default='', help='save directory')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--context_buffer', type=int, default=16, help='context_buffer size in pixels')
parser.add_argument('--trigger_reward_amplitude', type=float, default=3.0, help='trigger reward amplitude')
parser.add_argument('--trigger_threshold', type=float, default=0.6, help='trigger_threshold')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou threshold')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='confidence threshold')
parser.add_argument('--max_steps', type=int, default=199,
                    help='max number of steps, number of obs = number of steps + 1')
args = parser.parse_args()

print("########## Setting Up Experiment ######################")
experiment_path, experiment = utils.setup_run_folder(args, log=True)
torch.manual_seed(args.seed)
device = utils.get_device()
###############################################################################
#
# MODEL SETUP
#
###############################################################################
env = Environment(root=args.data,
                  detected_class=args.detected_class,
                  year=args.year,
                  context_buffer=args.context_buffer,
                  trigger_reward_amplitude=args.trigger_reward_amplitude,
                  trigger_threshold=args.trigger_threshold,
                  max_step=args.max_steps)
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

evaluator = Evaluator()



###############################################################################
#
# TESTING
#
###############################################################################

def testing_episode_average_precision():
    image, bounding_boxes_labels, bounding_boxes_region_proposal, trigger_indexes = agent.test_episode(env, rand=True)
    ious = [[utils.intersection_over_union(bb_region_proposal, bb_label) for bb_label in bounding_boxes_labels]
            for bb_region_proposal in bounding_boxes_region_proposal]
    ious = [max(iou) for iou in ious]
    image_proposals = [image[bb.y1:bb.y2, bb.x1:bb.x2] for bb in bounding_boxes_region_proposal]
    imgs_tensors = torch.stack([transform(img_prop) for img_prop in image_proposals]).to(utils.get_device())
    scores = evaluator.scores(imgs_tensors, args.detected_class).view(-1).detach().cpu().numpy()

    real = (np.array(ious) > args.iou_threshold).astype(float)

    def average_precision(scores, reals):
        scores_sorted_indexes = scores.argsort()
        sorted_scores = scores[scores_sorted_indexes[::-1]]
        sorted_real = reals[scores_sorted_indexes[::-1]]
        return average_precision_score(sorted_real, sorted_scores)
    try:
        ap_AAR = average_precision(scores, real)
    except:
        ap_AAR = 0.0
    try:
        ap_TR = average_precision(scores[trigger_indexes], real[trigger_indexes])
    except:
        ap_TR = 0.0

    return ap_AAR, ap_TR


def testing_mean_average_precision(nb_test=20):
    AP_AAR, AP_TR = [], []
    for test in range(nb_test):
        ap_AAR, ap_TR = testing_episode_average_precision()
        AP_AAR.append(ap_AAR)
        AP_TR.append(ap_TR)

    mAP_AAR = np.mean(np.nan_to_num(AP_AAR))
    mAP_TR = np.mean(np.nan_to_num(AP_TR))
    return mAP_AAR, mAP_TR

###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################
print("########## Running Main Loop ##########################")
for epoch in range(args.nb_epochs):
    print(f"Epoch={epoch}")
    for episode in tqdm(range(env.epoch_size)):
        episode_rewards = agent.train_episode(env)
        experiment.log_metric("Episode total reward", episode_rewards.sum().item())
        experiment.log_metric("Episode length", len(episode_rewards))
    mAP_AAR, mAP_TR = testing_mean_average_precision()
    print(mAP_AAR, mAP_TR)
    experiment.log_metric("mAP AAR", mAP_AAR)
    experiment.log_metric("mAP TR", mAP_TR)

agent_net_directory = "agent_net"
agent_net_filename = os.path.join(agent_net_directory, f"{args.detected_class}_agent_net.pt")
utils.make_dir_if_not_exist(agent_net_directory)
utils.save_model(agent.policy_q_net, agent_net_filename)
