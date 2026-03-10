import sys
import os
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2

from src.envs.treechopEnv import TreechopEnv
from src.agents.DQNAgent import DQNAgent


def appendEpisodeMetrics(csv_path, episode_metrics):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_metrics.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(episode_metrics)


def writeSummaryCsv(csv_path, summary_metrics):
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary_metrics.keys()))
        writer.writeheader()
        writer.writerow(summary_metrics)


def selectGreedyAction(agent, observation, device="cpu"):
    """
    lets choose the best action
    """
    state_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.q_network(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

    return action


def runEpisode(env, agent, device="cpu", video_path=None):
    observation = env.reset()
    done = False

    total_shaped_reward = 0.0
    total_env_reward = 0.0
    steps = 0
    max_streak = 0
    ever_break_window = False
    frames = []

    tree_view_steps = 0
    tree_view_fraction = 0.0
    ever_log_centered = False
    ever_likely_broke_log = False
    ever_likely_broke_but_no_reward = False
    recent_break_window_final = 0

    while not done:
        action = selectGreedyAction(agent, observation, device=device)
        observation, reward, done, info = env.step(action)

        total_shaped_reward += reward
        total_env_reward += info["env_reward"]
        steps += 1

        tree_view_steps = info["tree_view_step_count"]
        tree_view_fraction = info["tree_view_fraction"]
        max_streak = max(max_streak, info["centered_attack_streak"])
        recent_break_window_final = info["recent_break_window"]

        ever_log_centered = ever_log_centered or info["log_centered"]
        ever_likely_broke_log = ever_likely_broke_log or info["ever_likely_broke_log"]
        ever_likely_broke_but_no_reward = (
            ever_likely_broke_but_no_reward or info["likely_broke_but_no_reward"]
        )

        if info["recent_break_window"] > 0:
            ever_break_window = True

        frame = env.getCurrentFrame()
        if frame is not None:
            frames.append(frame.copy())

    if video_path and frames:
        height = frames[0].shape[0]
        width = frames[0].shape[1]

        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (width, height),
        )

        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

    episode_metrics = {
        "steps": steps,
        "total_shaped_reward": total_shaped_reward,
        "total_env_reward": total_env_reward,
        "tree_view_steps": tree_view_steps,
        "tree_view_fraction": tree_view_fraction,
        "ever_log_centered": int(ever_log_centered),
        "max_centered_attack_streak": max_streak,
        "ever_break_window": int(ever_break_window),
        "recent_break_window_final": recent_break_window_final,
        "ever_likely_broke_log": int(ever_likely_broke_log),
        "likely_broke_but_no_reward": int(ever_likely_broke_but_no_reward),
    }

    return episode_metrics


def main():
    checkpoint_path = "checkpoints/dqn_treechop_final.pt"
    num_eval_episodes = 20
    video_dir = "videos/eval_dqn"
    log_dir = "logs"
    device = "cpu"

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    episode_csv_path = os.path.join(log_dir, "eval_dqn_treechop_episodes.csv")
    summary_csv_path = os.path.join(log_dir, "eval_dqn_treechop_summary.csv")

    # remove old episode csv so this run starts fresh
    if os.path.exists(episode_csv_path):
        os.remove(episode_csv_path)

    env = TreechopEnv()

    agent = DQNAgent(
        state_channels=env.frame_stack,
        num_actions=env.num_actions,
        device=device,
    )

    if os.path.exists(checkpoint_path):
        agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.q_network.eval()
        print(f"loaded checkpoint: {checkpoint_path}")
    else:
        print("no checkpoint found") 

    print(f"running {num_eval_episodes} evaluation episodes...")

    results = []

    try:
        for episode_index in range(num_eval_episodes):
            video_path = os.path.join(video_dir, f"episode_{episode_index:03d}.mp4")

            result = runEpisode(
                env=env,
                agent=agent,
                device=device,
                video_path=video_path,
            )

            result["episode"] = episode_index
            result["video_path"] = video_path

            results.append(result)
            appendEpisodeMetrics(episode_csv_path, result)

            print(
                f"episode {episode_index:03d} | "
                f"shaped_reward={result['total_shaped_reward']:.2f} | "
                f"env_reward={result['total_env_reward']:.2f} | "
                f"steps={result['steps']}"
            )

        shaped_rewards = [result["total_shaped_reward"] for result in results]
        env_rewards = [result["total_env_reward"] for result in results]
        steps_list = [result["steps"] for result in results]
        streaks = [result["max_centered_attack_streak"] for result in results]
        tree_view_fractions = [result["tree_view_fraction"] for result in results]
        break_window_flags = [result["ever_break_window"] for result in results]
        ever_log_centered_flags = [result["ever_log_centered"] for result in results]
        likely_broke_flags = [result["ever_likely_broke_log"] for result in results]

        summary_metrics = {
            "num_eval_episodes": num_eval_episodes,
            "mean_shaped_reward": np.mean(shaped_rewards),
            "std_shaped_reward": np.std(shaped_rewards),
            "min_shaped_reward": np.min(shaped_rewards),
            "max_shaped_reward": np.max(shaped_rewards),
            "mean_env_reward": np.mean(env_rewards),
            "std_env_reward": np.std(env_rewards),
            "min_env_reward": np.min(env_rewards),
            "max_env_reward": np.max(env_rewards),
            "avg_steps": np.mean(steps_list),
            "avg_max_centered_attack_streak": np.mean(streaks),
            "avg_tree_view_fraction": np.mean(tree_view_fractions),
            "break_window_rate": np.mean(break_window_flags),
            "log_centered_rate": np.mean(ever_log_centered_flags),
            "likely_broke_log_rate": np.mean(likely_broke_flags),
            "success_rate_env_reward_positive": np.mean([reward > 0 for reward in env_rewards]),
        }

        writeSummaryCsv(summary_csv_path, summary_metrics)

        print("\nfinished evaluation")
        print(f"saved episode metrics csv to {episode_csv_path}")
        print(f"saved summary csv to {summary_csv_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()