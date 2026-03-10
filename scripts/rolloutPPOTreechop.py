import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
from src.envs.treechopEnv import TreechopEnv
from src.agents.PPOAgent import PPOAgent


def main():
    checkpoint_path = "checkpoints/ppo_treechop_final.pt"

    env = TreechopEnv()

    agent = PPOAgent(
        state_channels=env.frame_stack,
        num_actions=env.num_actions,
        device="cpu",
    )

    if os.path.exists(checkpoint_path):
        agent.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        agent.network.eval()
        print(f"loaded checkpoint from {checkpoint_path}")
    else:
        print("no checkpoint found, running with random weights")

    observation = env.reset()
    print("rollout started")
    print("observation shape:", observation.shape)
    print("number of actions:", env.num_actions)

    done = False
    step_index = 0
    total_env_reward = 0.0
    total_shaped_reward = 0.0
    frames = []

    while not done:
        action, log_prob, value = agent.selectAction(observation)
        observation, reward, done, info = env.step(action)

        total_shaped_reward += reward
        total_env_reward += info["env_reward"]

        # grab the raw rgb frame for video
        frame = env.getCurrentFrame()
        if frame is not None:
            frames.append(frame.copy())

        print(
            f"step {step_index:04d} | "
            f"action {action} | "
            f"env_reward {info['env_reward']:.2f} | "
            f"shaped_reward {info['shaped_reward']:.2f} | "
            f"value {value:.3f} | "
            f"log_prob {log_prob:.3f}"
        )
        step_index += 1

    print("\nepisode finished")
    print("episode length:", step_index)
    print("total env reward:", total_env_reward)
    print("total shaped reward:", total_shaped_reward)

    # save video
    if frames:
        os.makedirs("videos", exist_ok=True)
        video_path = "videos/ppo_rollout2.mp4"
        h, w = frames[0].shape[0], frames[0].shape[1]
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (w, h),
        )
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"saved video to {video_path}")
    else:
        print("no frames captured, video not saved")

    env.close()


if __name__ == "__main__":
    main()