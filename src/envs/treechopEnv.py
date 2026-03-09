import gym
import minerl
import numpy as np
import cv2
from collections import deque

class TreechopEnv:
    """
    we:
    - create the MineRLTreechop-v0 env
    - convert the MineRL action dictionary into a small discrete action space
    - preprocess observations into grayscale 64x64 frames
    - add light reward shaping based on log gains (might want to move this into a different file)
    """

    def __init__(
            self,
            image_size = 64,
            frame_stack = 4,
            max_episode_steps = 1000,
            extra_log_reward = 20.0,
            time_penalty = 0.01,
    ):
        
        self.env = gym.make("MineRLTreechop-v0")

        self.image_size = image_size
        self.frame_stack = frame_stack 

        # reward shaping settings
        self.max_episode_steps = max_episode_steps
        self.extra_log_reward = extra_log_reward
        self.time_penalty = time_penalty

        self.current_step = 0
        self.previous_log_count = 0

        self.discrete_actions = self.buildDiscreteActions()
        self.num_actions = len(self.discrete_actions)

        # create a frame buffer for stacking recent observations
        self.stacked_frames = deque(maxlen=self.frame_stack)

        # store latest raw pov frame for debug
        self.latest_raw_frame = None
    
    def buildDiscreteActions(self):
        """
        ADD ANY NEW ACTIONS HERE
        """
        discrete_actions = []

        # action 0: do nothing
        discrete_actions.append(self.makeNoopAction())

        # action 1: move forward
        action = self.makeNoopAction()
        action["forward"] = 1
        discrete_actions.append(action)

        # action 2: turn camera left
        action = self.makeNoopAction()
        action["camera"] = [-10, 0]
        discrete_actions.append(action)

        # action 3: turn camera right
        action = self.makeNoopAction()
        action["camera"] = [10, 0]
        discrete_actions.append(action)

        # action 4: look up
        action = self.makeNoopAction()
        action["camera"] = [0, -10]
        discrete_actions.append(action)

        # action 5: look down
        action = self.makeNoopAction()
        action["camera"] = [0, 10]
        discrete_actions.append(action)

        # action 6: attack
        action = self.makeNoopAction()
        action["attack"] = 1
        discrete_actions.append(action)

        # action 7: move forward while attacking
        action = self.makeNoopAction()
        action["forward"] = 1
        action["attack"] = 1
        discrete_actions.append(action)

        return discrete_actions

    def makeNoopAction(self):
        return self.env.action_space.noop()

    def extractPovImage(self, observation):
        return observation["pov"]
    
    def preprocessFrame(self, observation):
        """
        preprocess a single pov frame.
        """

        # get the raw pov image from the observation
        pov_image = self.extractPovImage(observation)

        # keep a copy of the latest raw frame for optional video saving
        self.latest_raw_frame = pov_image.copy()

        # resize to a smaller image for faster training
        resized_image = cv2.resize(
            pov_image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        # convert the RGB image to grayscale
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

        # normalize pixel values to [0, 1]
        grayscale_image = grayscale_image.astype(np.float32) / 255.0

        # add a channel dimension so the final shape is (1, H, W)
        grayscale_image = np.expand_dims(grayscale_image, axis=0)

        return grayscale_image
    
    def initFrameStack(self, processed_frame):
        """
        fill the frame stack with the first frame after reset.
        """
        self.stacked_frames.clear()

        for _ in range(self.frame_stack):
            self.stacked_frames.append(processed_frame)

    def getStackedObservation(self):
        """
        combine the most recent grayscale frames into one stacked observation.

        shape:
        (frame_stack, H, W)
        """
        return np.concatenate(list(self.stacked_frames), axis=0)

    def getLogCount(self, observation):
        """
        after investigation, treechop doesnt reveal inventory so lets just return 0 for now
        """
        return 0

    def computeShapedReward(self, env_reward):
        """
        compute the shaped reward.
        """
        shaped_reward = float(env_reward)
        shaped_reward -= self.time_penalty
        return shaped_reward

    def reset(self):
        """
        reset the environment and return the first stacked observation.
        """
        raw_observation = self.env.reset()

        self.current_step = 0
        self.previous_log_count = 0

        processed_frame = self.preprocessFrame(raw_observation)
        self.initFrameStack(processed_frame)

        return self.getStackedObservation()

    def step(self, action_index):
        """
        take one step in the environment
        """
        action_dictionary = self.discrete_actions[action_index]

        raw_observation, env_reward, done, info = self.env.step(action_dictionary)

        self.current_step += 1

        processed_frame = self.preprocessFrame(raw_observation)
        self.stacked_frames.append(processed_frame)
        stacked_observation = self.getStackedObservation()

        shaped_reward = self.computeShapedReward(env_reward)

        if self.current_step >= self.max_episode_steps:
            done = True

        info["env_reward"] = float(env_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["episode_step"] = int(self.current_step)

        return stacked_observation, shaped_reward, done, info

    def getCurrentFrame(self):
        """
        return the latest raw RGB pov frame.
        """
        return self.latest_raw_frame

    def sampleRandomAction(self):
        """
        sample a random action
        """
        return np.random.randint(self.num_actions)

    def close(self):
        """
        close the environment
        """
        self.env.close()