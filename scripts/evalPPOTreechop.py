import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from src.envs.treechopEnv import TreechopEnv
from src.agents.PPOAgent import PPOAgent


def run_episode(env, agent, video_path=None):
    observation = env.reset()
    done = False
    total_shaped_reward = 0.0
    total_env_reward = 0.0
    steps = 0
    max_streak = 0
    ever_break_window = False
    frames = []

    while not done:
        action, _, _ = agent.selectAction(observation)
        observation, reward, done, info = env.step(action)
        total_shaped_reward += reward
        total_env_reward += info["env_reward"]
        steps += 1
        max_streak = max(max_streak, info["centered_attack_streak"])
        if info["recent_break_window"] > 0:
            ever_break_window = True

        frame = env.getCurrentFrame()
        if frame is not None:
            frames.append(frame.copy())

    if video_path and frames:
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

    return {
        "shaped_reward": total_shaped_reward,
        "env_reward": total_env_reward,
        "steps": steps,
        "max_streak": max_streak,
        "ever_break_window": ever_break_window,
        "tree_view_fraction": info["tree_view_fraction"],
    }


def main():
    checkpoint_path = "checkpoints/ppo_treechop_final.pt"
    num_eval_episodes = 10
    video_dir = "videos/eval"
    os.makedirs(video_dir, exist_ok=True)

    env = TreechopEnv()

    agent = PPOAgent(
        state_channels=env.frame_stack,
        num_actions=env.num_actions,
        device="cpu",
    )

    if os.path.exists(checkpoint_path):
        agent.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        agent.network.eval()
        print(f"loaded checkpoint: {checkpoint_path}")
    else:
        print("no checkpoint found, evaluating random policy")

    print(f"running {num_eval_episodes} evaluation episodes...\n")

    results = []
    for i in range(num_eval_episodes):
        video_path = os.path.join(video_dir, f"episode_{i:03d}.mp4")
        result = run_episode(env, agent, video_path=video_path)
        results.append(result)
        print(
            f"ep {i:03d} | "
            f"shaped {result['shaped_reward']:7.2f} | "
            f"steps {result['steps']:4d} | "
            f"streak {result['max_streak']:3d} | "
            f"tree_view {result['tree_view_fraction']:.2f} | "
            f"break_window {result['ever_break_window']} | "
            f"video saved to {video_path}"
        )

    print("\n--- evaluation summary ---")
    shaped_rewards = [r["shaped_reward"] for r in results]
    print(f"shaped reward:     mean={np.mean(shaped_rewards):.2f}  std={np.std(shaped_rewards):.2f}  min={np.min(shaped_rewards):.2f}  max={np.max(shaped_rewards):.2f}")
    print(f"avg steps:         {np.mean([r['steps'] for r in results]):.1f}")
    print(f"avg max streak:    {np.mean([r['max_streak'] for r in results]):.1f}")
    print(f"avg tree_view:     {np.mean([r['tree_view_fraction'] for r in results]):.3f}")
    print(f"break window rate: {np.mean([r['ever_break_window'] for r in results]):.2f}")

    env.close()


if __name__ == "__main__":
    main()