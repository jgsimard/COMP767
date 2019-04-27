import copy

import cv2
import gluoncv
import gym
import numpy as np
from gym import spaces

from utils import BoundingBox, intersection_over_union, get_bbs


class Environment(gym.Env):
    def __init__(self,
                 root="./data/VOCtrainval_06-Nov-2007/VOCdevkit",
                 detected_class=14,  # person!
                 alpha=0.2,
                 context_buffer=16,
                 trigger_reward_amplitude=3,
                 trigger_threshold=0.6,
                 max_step=199,
                 set_name='trainval',
                 year=2007):

        self.year = year
        self.max_step = max_step
        self.context_buffer = context_buffer
        self.trigger_reward_amplitude = trigger_reward_amplitude
        self.trigger_threshold = trigger_threshold
        self.voc_dataset = gluoncv.data.VOCDetection(root=root, splits=[(year, set_name)])

        self.detected_class = detected_class
        self.detected_class_indexes = self.get_detected_class_indexes(root, set_name)
        self.epoch_size = len(self.detected_class_indexes)

        self.output_image_size = 224
        self.output_image_shape = (self.output_image_size, self.output_image_size)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.output_image_size, self.output_image_size, 3))

        self.alpha = alpha
        self.current_bounding_box = None
        self.image = None
        self.past_iou = None

        self.image_absolute_index = 0
        self.image_class_index = 0

        self.action_index_to_names = {-1:"start", 0: "right", 1: "left", 2: "up", 3: "down", 4: "bigger", 5: "smaller", 6: "fatter",
                                      7: "taller", 8: "trigger"}
        self.action_names_to_index = {v: k for k, v in self.action_index_to_names.items()}

        print(f"Environement initializatione done for class : {self.voc_dataset.classes[detected_class]}")

    def get_detected_class_indexes(self, root, set_name="trainval"):
        fname = f"{root}/VOC{self.year}/ImageSets/Main/{self.voc_dataset.classes[self.detected_class]}_{set_name}.txt"
        with open(fname) as f:
            content = f.readlines()

        content_index = [int(x.strip().split()[0]) for x in content]
        content_detected_class = [int(x.strip().split()[0]) for x in content if int(x.strip().split()[1]) == 1]
        content_detected_class_filtered = []
        i = 0
        for class_img_index in content_detected_class:
            while True:
                if content_index[i] == class_img_index and i < len(content_index):
                    content_detected_class_filtered.append(i)
                    break
                i += 1
        return content_detected_class_filtered

    def add_ior(self, img, bb, f=4):  # ior = inhibition of return
        w = bb.x2 - bb.x1
        h = bb.y2 - bb.y1
        delta_x = int(w * (1 - 1 / f) / 2)
        delta_y = int(h * (1 - 1 / f) / 2)
        cv2.rectangle(img, (bb.x1 + delta_x, bb.y1), (bb.x2 - delta_x, bb.y2), (0, 0, 0), -1)
        cv2.rectangle(img, (bb.x1, bb.y1 + delta_y), (bb.x2, bb.y2 - delta_y), (0, 0, 0), -1)

    def initial_bounding_box(self):
        return BoundingBox(self.context_buffer,
                           self.context_buffer,
                           self.output_image_size - self.context_buffer,
                           self.output_image_size - self.context_buffer)

    def restart_bounding_box(self):
        x = np.random.randint(0, 4)
        percentage = 0.75
        a = int((self.output_image_size - self.context_buffer) * (1 - np.sqrt(percentage)))
        b = int((self.output_image_size - self.context_buffer) * (np.sqrt(percentage)))
        # We create another box of area 75% of the previous one
        if x == 0:
            return BoundingBox(self.context_buffer,
                               self.context_buffer,
                               b,
                               b)
        elif x == 1:
            return BoundingBox(self.context_buffer,
                               self.context_buffer + a,
                               b,
                               self.output_image_size - self.context_buffer)
        elif x == 2:
            return BoundingBox(self.context_buffer + a,
                               self.context_buffer,
                               self.output_image_size - self.context_buffer,
                               b)
        else:
            return BoundingBox(self.context_buffer + a,
                               self.context_buffer + a,
                               self.output_image_size - self.context_buffer,
                               self.output_image_size - self.context_buffer)

    def restart(self):
        self.current_bounding_box = self.restart_bounding_box()
        self.past_iou = intersection_over_union(self.current_bounding_box, self.ground_truth_bounding_box())
        return self.get_obs()

    def get_obs(self):
        y1 = max(self.current_bounding_box.y1 - self.context_buffer, 0)
        y2 = min(self.current_bounding_box.y2 + self.context_buffer, self.output_image_size)
        x1 = max(self.current_bounding_box.x1 - self.context_buffer, 0)
        x2 = min(self.current_bounding_box.x2 + self.context_buffer, self.output_image_size)
        img = cv2.resize(self.image[y1:y2, x1:x2, :], self.output_image_shape)

        return self.image_absolute_index, img, self.current_bounding_box

    def next_bounding_box(self, bb, action):
        new_bb = copy.deepcopy(bb)

        a_w = int(self.alpha * (bb.x2 - bb.x1))
        a_h = int(self.alpha * (bb.y2 - bb.y1))

        delta_x_right = min(self.output_image_size - bb.x2, a_w)
        delta_x_left = -min(bb.x1, a_w)
        delta_y_up = -min(bb.y1, a_h)
        delta_y_down = min(self.output_image_size - bb.y2, a_h)

        # right
        if action == 0:
            new_bb.x1 += delta_x_right
            new_bb.x2 += delta_x_right
        # left
        elif action == 1:
            new_bb.x1 += delta_x_left
            new_bb.x2 += delta_x_left
        # up
        elif action == 2:
            new_bb.y1 += delta_y_up
            new_bb.y2 += delta_y_up
        # down
        elif action == 3:
            new_bb.y1 += delta_y_down
            new_bb.y2 += delta_y_down
        # bigger : expand in all direction
        elif action == 4:
            new_bb.x1 += delta_x_left
            new_bb.x2 += delta_x_right
            new_bb.y1 += delta_y_up
            new_bb.y2 += delta_y_down
        # smaller : trim all sides
        elif action == 5:
            new_bb.x1 += a_w
            new_bb.x2 -= a_w
            new_bb.y1 += a_h
            new_bb.y2 -= a_h
        # fatter : trim the height
        elif action == 6:
            new_bb.y1 += a_h
            new_bb.y2 -= a_h
        # taller : trim the sides
        elif action == 7:
            new_bb.x1 += a_w
            new_bb.x2 -= a_w
        # trigger
        else:
            new_bb = self.initial_bounding_box()

        return new_bb

    # I chose to give the grond truch box with the biggest IoU if multiple target
    def ground_truth_bounding_box(self):
        ious = [intersection_over_union(self.current_bounding_box, ground_truth_bb) for ground_truth_bb in
                self.bounding_boxes]
        return self.bounding_boxes[np.argmax(ious)]

    def transformation_action_reward(self, new_iou):
        return 1. if new_iou > self.past_iou else -1.

    def trigger_reward(self):
        above_threshold = 1 if self.past_iou > self.trigger_threshold else -1
        return self.trigger_reward_amplitude * above_threshold

    def positive_reward_actions(self):
        rewards = []
        for action in range(self.action_space.n):
            if action < self.action_space.n - 1:
                new_bb = self.next_bounding_box(self.current_bounding_box, action)
                new_iou = intersection_over_union(self.ground_truth_bounding_box(), new_bb)
                reward = self.transformation_action_reward(new_iou)
            else:
                reward = self.trigger_reward()
            rewards.append(reward)
        positive_rewards = np.where(np.array(rewards) > 0)[0]

        return positive_rewards

    def step_train(self, action, go_to_max_iter=False):
        self.t += 1
        done = False
        if self.t >= self.max_step:
            done = True

        new_bb = self.next_bounding_box(self.current_bounding_box, action)
        if action < self.action_space.n - 1:
            new_iou = intersection_over_union(self.ground_truth_bounding_box(), new_bb)
            reward = self.transformation_action_reward(new_iou)
            self.past_iou = new_iou
        else:
            # print(self.past_iou)  # TODO : TO REMOVE
            reward = self.trigger_reward()
            self.add_ior(self.image, self.current_bounding_box)
            if go_to_max_iter and len(self.bounding_boxes) == 0:
                self.past_iou = -1
            else:
                self.bounding_boxes.remove(self.ground_truth_bounding_box())
                if len(self.bounding_boxes) == 0:
                    done = True
                else:
                    self.past_iou = intersection_over_union(new_bb, self.ground_truth_bounding_box())
        self.current_bounding_box = new_bb
        obs = self.get_obs()

        return obs, reward, done, {}

    def step_testing(self, action):
        self.t += 1

        new_bb = self.next_bounding_box(self.current_bounding_box, action)
        if action == self.action_space.n - 1:
            self.add_ior(self.image, self.current_bounding_box)
        self.current_bounding_box = new_bb

        obs = self.get_obs()
        reward = 0
        done = False if self.t < self.max_step else True

        return obs, reward, done, {}

    # TODO
    def step(self, action, train=True, go_to_max_iter=False):
        if train:
            return self.step_train(action, go_to_max_iter)
        else:
            return self.step_testing(action)

    # TODO
    def reset(self, rand=False):
        self.t = 0
        if rand:
            self.image_absolute_index = np.random.choice(self.detected_class_indexes)
        else:
            self.image_absolute_index = self.detected_class_indexes[self.image_class_index]
            self.image_class_index = (self.image_class_index + 1) % len(self.detected_class_indexes)
        image, label = self.voc_dataset[self.image_absolute_index]

        self.image = cv2.resize(image.asnumpy(), self.output_image_shape)
        self.bounding_boxes = get_bbs(image.shape, self.output_image_shape, label, self.detected_class)
        self.current_bounding_box = self.initial_bounding_box()

        self.past_iou = intersection_over_union(self.current_bounding_box, self.ground_truth_bounding_box())
        return self.get_obs()
