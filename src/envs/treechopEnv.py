import os
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
    - track how often the agent appears to be looking at a tree
    - record episode frames so the training script can choose which runs to save
    """

    def __init__(
        self,
        image_size=64,
        frame_stack=4,
        max_episode_steps=1000,
        extra_log_reward=50.0,
        tree_in_view_bonus=0.005,
        centered_log_bonus=0.02,
        attack_on_target_bonus=0.04,
        forward_attack_bonus=0.02,
        pickup_followthrough_bonus=0.25,
        time_penalty=0.005,
        break_streak_threshold=20,
        break_window_length=16,
        save_videos=True,
        video_fps=30,
        video_scale=20,
    ):
        self.env = gym.make("MineRLTreechop-v0")

        self.image_size = image_size
        self.frame_stack = frame_stack

        # reward shaping settings
        self.max_episode_steps = max_episode_steps
        self.extra_log_reward = extra_log_reward
        self.tree_in_view_bonus = tree_in_view_bonus
        self.centered_log_bonus = centered_log_bonus
        self.attack_on_target_bonus = attack_on_target_bonus
        self.forward_attack_bonus = forward_attack_bonus
        self.pickup_followthrough_bonus = pickup_followthrough_bonus
        self.time_penalty = time_penalty
        self.break_streak_threshold = break_streak_threshold
        self.break_window_length = break_window_length

        self.current_step = 0

        # track sustained attack while centered on target
        self.centered_attack_streak = 0

        # track short window after we think the log may have broken
        self.recent_break_window = 0

        # keep previous streak so we can detect a likely "break happened" transition
        self.previous_centered_attack_streak = 0

        # tree-view tracking metrics
        self.tree_view_step_count = 0
        self.tree_view_fraction = 0.0

        # hysteresis state for target lock
        self.log_centered_last_step = False

        self.discrete_actions = self.buildDiscreteActions()
        self.num_actions = len(self.discrete_actions)

        # create a frame buffer for stacking recent observations
        self.stacked_frames = deque(maxlen=self.frame_stack)

        # store latest raw pov frame for debug
        self.latest_raw_frame = None

        # sparse reward environment problems...
        self.ever_likely_broke_log = False
        self.ever_likely_broke_but_no_reward = False

        # video settings
        self.save_videos = save_videos
        self.video_fps = video_fps
        self.video_scale = video_scale
        self.episode_video_frames = []
        self.episode_video_index = 0

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        self.video_directory = os.path.join(current_file_directory, "videos")

        if self.save_videos:
            os.makedirs(self.video_directory, exist_ok=True)

    def buildDiscreteActions(self):
        """
        use a smaller action space so exploration is less wasteful
        """
        discrete_actions = []

        # action 0: move forward
        action = self.makeNoopAction()
        action["forward"] = 1
        discrete_actions.append(action)

        # action 1: turn camera left
        action = self.makeNoopAction()
        action["camera"] = [-10, 0]
        discrete_actions.append(action)

        # action 2: turn camera right
        action = self.makeNoopAction()
        action["camera"] = [10, 0]
        discrete_actions.append(action)

        # action 3: look down
        action = self.makeNoopAction()
        action["camera"] = [0, 10]
        discrete_actions.append(action)

        # action 4: repeated attack macro
        action = self.makeNoopAction()
        action["attack"] = 1
        action["_repeat"] = 20
        discrete_actions.append(action)

        # action 5: repeated forward + attack macro
        action = self.makeNoopAction()
        action["forward"] = 1
        action["attack"] = 1
        action["_repeat"] = 20
        discrete_actions.append(action)

        # action 6: repeated forward-only pickup macro
        action = self.makeNoopAction()
        action["forward"] = 1
        action["_repeat"] = 20
        discrete_actions.append(action)

        return discrete_actions

    def makeNoopAction(self):
        return self.env.action_space.noop()

    def extractPovImage(self, observation):
        return observation["pov"]

    def preprocessFrame(self, observation):
        """
        preprocess a single pov frame
        """
        pov_image = self.extractPovImage(observation)

        # keep a copy of the latest raw frame for debugging and tree detection
        self.latest_raw_frame = pov_image.copy()

        # resize to a smaller image for faster training
        resized_image = cv2.resize(
            pov_image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        # convert the rgb image to grayscale
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

        # normalize pixel values to [0, 1]
        grayscale_image = grayscale_image.astype(np.float32) / 255.0

        # add a channel dimension so the final shape is (1, h, w)
        grayscale_image = np.expand_dims(grayscale_image, axis=0)

        return grayscale_image

    def initFrameStack(self, processed_frame):
        """
        fill the frame stack with the first frame after reset
        """
        self.stacked_frames.clear()

        for _ in range(self.frame_stack):
            self.stacked_frames.append(processed_frame)

    def getStackedObservation(self):
        """
        combine the most recent grayscale frames into one stacked observation

        shape:
        (frame_stack, h, w)
        """
        return np.concatenate(list(self.stacked_frames), axis=0)

    def getWoodMask(self, frame):
        """
        detect likely trunk-colored pixels using a color heuristic
        """
        red_channel = frame[:, :, 0]
        green_channel = frame[:, :, 1]
        blue_channel = frame[:, :, 2]

        brown_mask = (
            (red_channel > 60)
            & (red_channel < 170)
            & (green_channel > 40)
            & (green_channel < 120)
            & (blue_channel > 20)
            & (blue_channel < 100)
            & (red_channel > green_channel)
            & (green_channel >= blue_channel)
        )

        return brown_mask

    def isTreeLikelyInCenter(self):
        """
        is the tree in the center of the view?
        """
        if self.latest_raw_frame is None:
            return False

        frame = self.latest_raw_frame
        frame_height, frame_width, _ = frame.shape

        center_y = frame_height // 2
        center_x = frame_width // 2

        patch_half_size = 12

        y_min = max(center_y - patch_half_size, 0)
        y_max = min(center_y + patch_half_size, frame_height)
        x_min = max(center_x - patch_half_size, 0)
        x_max = min(center_x + patch_half_size, frame_width)

        center_patch = frame[y_min:y_max, x_min:x_max]
        brown_mask = self.getWoodMask(center_patch)
        brown_fraction = np.mean(brown_mask)

        return brown_fraction > 0.08

    def isLogCentered(self):
        """
        stricter check: is a likely log/trunk very close to the crosshair?

        we use hysteresis:
        - entering centered state uses a stricter threshold
        - staying centered uses a looser threshold
        """
        if self.latest_raw_frame is None:
            return False

        frame = self.latest_raw_frame
        frame_height, frame_width, _ = frame.shape

        center_y = frame_height // 2
        center_x = frame_width // 2

        patch_half_size = 7

        y_min = max(center_y - patch_half_size, 0)
        y_max = min(center_y + patch_half_size, frame_height)
        x_min = max(center_x - patch_half_size, 0)
        x_max = min(center_x + patch_half_size, frame_width)

        center_patch = frame[y_min:y_max, x_min:x_max]
        brown_mask = self.getWoodMask(center_patch)
        brown_fraction = np.mean(brown_mask)

        if self.log_centered_last_step:
            log_centered = brown_fraction > 0.16
        else:
            log_centered = brown_fraction > 0.22

        self.log_centered_last_step = log_centered
        return log_centered

    def updateTreeViewMetrics(self):
        """
        update counters for how often the agent appears to be looking at a tree
        """
        tree_in_view = self.isTreeLikelyInCenter()

        if tree_in_view:
            self.tree_view_step_count += 1

        if self.current_step > 0:
            self.tree_view_fraction = self.tree_view_step_count / self.current_step
        else:
            self.tree_view_fraction = 0.0

        return tree_in_view

    def updateCenteredAttackStreak(self, action_dictionary, log_centered):
        """
        keep track of consecutive attack steps while the log stays centered
        """
        self.previous_centered_attack_streak = self.centered_attack_streak

        if action_dictionary["attack"] == 1 and log_centered:
            self.centered_attack_streak += 1
        else:
            self.centered_attack_streak = 0

        return self.centered_attack_streak

    def updateRecentBreakWindow(self, action_dictionary, log_centered):
        """
        open a short pickup window after a likely break event
        """
        likely_just_broke = (
            self.previous_centered_attack_streak >= self.break_streak_threshold
            and (
                not log_centered
                or action_dictionary["attack"] != 1
            )
        )

        if likely_just_broke:
            self.recent_break_window = self.break_window_length
            self.ever_likely_broke_log = True
        elif self.recent_break_window > 0:
            self.recent_break_window -= 1

        return likely_just_broke

    def computeShapedReward(
        self,
        env_reward,
        action_dictionary,
        tree_in_view,
        log_centered,
        centered_attack_streak,
    ):
        """
        compute reward considering our reward shaping
        """
        shaped_reward = 0.0

        # keep the original reward
        shaped_reward += float(env_reward)

        # amplify true environment reward heavily since task reward is sparse
        shaped_reward += self.extra_log_reward * float(env_reward)

        # small encouragement for finding a tree at all
        if tree_in_view:
            shaped_reward += self.tree_in_view_bonus

        # stronger encouragement for aligning crosshair with likely log
        if log_centered:
            shaped_reward += self.centered_log_bonus

        # reward attack only when the target is actually centered
        if action_dictionary["attack"] == 1 and log_centered:
            shaped_reward += self.attack_on_target_bonus

        # encourage sustaining attack on the centered target
        if centered_attack_streak >= 2:
            shaped_reward += 0.02
        if centered_attack_streak >= 4:
            shaped_reward += 0.03
        if centered_attack_streak >= 8:
            shaped_reward += 0.05
        if centered_attack_streak >= 12:
            shaped_reward += 0.08
        if centered_attack_streak >= 16:
            shaped_reward += 0.10
        if centered_attack_streak >= 20:
            shaped_reward += 0.14
        if centered_attack_streak >= 24:
            shaped_reward += 0.18

        # small bonus for moving into the tree while attacking and centered
        if (
            action_dictionary["attack"] == 1
            and action_dictionary["forward"] == 1
            and log_centered
        ):
            shaped_reward += self.forward_attack_bonus

        # if we likely just broke a log, encourage moving forward to collect it
        if self.recent_break_window > 0 and action_dictionary["forward"] == 1:
            shaped_reward += self.pickup_followthrough_bonus

        # discourage wasting the break window by continuing to throw punches
        if self.recent_break_window > 0 and action_dictionary["attack"] == 1:
            shaped_reward -= 0.03

        shaped_reward -= self.time_penalty

        return shaped_reward

    # video code to help figure out why no env reward -------

    def recordVideoFrame(self, observation):
        """
        save the raw pov frame for episode video generation
        """
        if not self.save_videos:
            return

        raw_frame = observation["pov"].copy()
        self.episode_video_frames.append(raw_frame)

    def saveEpisodeVideo(self):
        """
        write the current episode frames to an mp4 file
        """
        if not self.save_videos:
            return

        if len(self.episode_video_frames) == 0:
            return

        first_frame = self.episode_video_frames[0]
        frame_height, frame_width, _ = first_frame.shape

        scaled_width = frame_width * self.video_scale
        scaled_height = frame_height * self.video_scale

        video_path = os.path.join(
            self.video_directory,
            f"treechop_episode_{self.episode_video_index:04d}.mp4",
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            float(self.video_fps),
            (scaled_width, scaled_height),
        )

        print(f"original frame size: {frame_width}x{frame_height}")
        print(f"scaled video size: {scaled_width}x{scaled_height}")
        print(f"saved episode video to {video_path}")

        for rgb_frame in self.episode_video_frames:
            enlarged_rgb_frame = cv2.resize(
                rgb_frame,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_NEAREST,
            )

            bgr_frame = cv2.cvtColor(enlarged_rgb_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

        video_writer.release()

        self.episode_video_frames = []
        self.episode_video_index += 1

    def discardEpisodeVideo(self):
        """
        discard recorded frames for the current episode
        """
        self.episode_video_frames = []

    # video code to help figure out why no env reward -------

    def reset(self):
        """
        reset the environment and return the first stacked observation
        """
        raw_observation = self.env.reset()

        self.current_step = 0
        self.centered_attack_streak = 0
        self.previous_centered_attack_streak = 0
        self.log_centered_last_step = False
        self.recent_break_window = 0

        # reset tree-view tracking metrics
        self.tree_view_step_count = 0
        self.tree_view_fraction = 0.0

        self.ever_likely_broke_log = False
        self.ever_likely_broke_but_no_reward = False

        # reset video buffer for the new episode and record the first frame
        self.episode_video_frames = []
        self.recordVideoFrame(raw_observation)

        processed_frame = self.preprocessFrame(raw_observation)
        self.initFrameStack(processed_frame)

        return self.getStackedObservation()

    def step(self, action_index):
        """
        take one step in the environment

        some discrete actions are macro actions that repeat internally
        """
        action_dictionary = self.discrete_actions[action_index].copy()
        repeat_count = action_dictionary.pop("_repeat", 1)

        total_env_reward = 0.0
        total_shaped_reward = 0.0
        done = False
        info = {}

        tree_in_view = False
        log_centered = False
        centered_attack_streak = 0

        likely_broke_this_step = False

        for _ in range(repeat_count):
            raw_observation, env_reward, done, info = self.env.step(action_dictionary)

            self.current_step += 1
            total_env_reward += env_reward

            self.recordVideoFrame(raw_observation)

            processed_frame = self.preprocessFrame(raw_observation)
            self.stacked_frames.append(processed_frame)

            tree_in_view = self.updateTreeViewMetrics()
            log_centered = self.isLogCentered()

            centered_attack_streak = self.updateCenteredAttackStreak(
                action_dictionary,
                log_centered,
            )

            likely_just_broke = self.updateRecentBreakWindow(
                action_dictionary,
                log_centered,
            )

            if likely_just_broke:
                likely_broke_this_step = True
                self.ever_likely_broke_log = True

            step_shaped_reward = self.computeShapedReward(
                env_reward,
                action_dictionary,
                tree_in_view,
                log_centered,
                centered_attack_streak,
            )

            total_shaped_reward += step_shaped_reward

            if done or self.current_step >= self.max_episode_steps:
                done = True
                break

        stacked_observation = self.getStackedObservation()

        likely_broke_but_no_reward = (
            self.ever_likely_broke_log
            and total_env_reward <= 0.0
        )

        self.ever_likely_broke_but_no_reward = likely_broke_but_no_reward

        info["env_reward"] = float(total_env_reward)
        info["shaped_reward"] = float(total_shaped_reward)
        info["episode_step"] = int(self.current_step)

        # tree / targeting metrics
        info["tree_in_view"] = bool(tree_in_view)
        info["log_centered"] = bool(log_centered)
        info["centered_attack_streak"] = int(centered_attack_streak)
        info["tree_view_step_count"] = int(self.tree_view_step_count)
        info["tree_view_fraction"] = float(self.tree_view_fraction)
        info["recent_break_window"] = int(self.recent_break_window)
        info["previous_centered_attack_streak"] = int(self.previous_centered_attack_streak)

        # sparse world problems, extra metrics to help
        info["likely_broke_this_step"] = bool(likely_broke_this_step)
        info["ever_likely_broke_log"] = bool(self.ever_likely_broke_log)
        info["likely_broke_but_no_reward"] = bool(likely_broke_but_no_reward)

        return stacked_observation, total_shaped_reward, done, info

    def getCurrentFrame(self):
        """
        return the latest raw rgb pov frame
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
        self.discardEpisodeVideo()
        self.env.close()