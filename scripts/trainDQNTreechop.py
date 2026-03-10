import os
import csv
import torch
import numpy as np

from src.envs.treechopEnv import TreechopEnv
from src.agents.DQNAgent import DQNAgent


def saveCheckpoint(agent, checkpoint_path):
    """
    save the online q-network parameters, so we dont waste lots of time like before :(
    """
    checkpoint_directory = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_directory, exist_ok=True)

    torch.save(agent.q_network.state_dict(), checkpoint_path)


def appendEpisodeMetrics(csv_path, episode_metrics):
    """
    append one episode's metrics to a csv file
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_metrics.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(episode_metrics)


def main():
    """
    train a dqn agent on MineRL treechop
    """

    # training hyperparameters
    num_episodes = 200
    max_steps_per_episode = 1000
    checkpoint_every = 10

    env = TreechopEnv(
        max_episode_steps=max_steps_per_episode,
        save_videos=False, # toggle depending on what we need
        video_fps=30,
        video_scale=20,
    )

    # create agent
    agent = DQNAgent(
        state_channels=env.frame_stack,
        num_actions=env.num_actions,
        learning_rate=1e-4,
        gamma=0.99,
        replay_buffer_capacity=100000,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=30000,
        target_update_frequency=1000,
    )

    # keep track of total environment interaction steps
    global_step_count = 0

    # keep track of the best run so far for video saving
    best_env_reward = float("-inf")
    best_centered_attack_streak = float("-inf")
    best_tree_view_fraction = float("-inf")

    # save progress code
    checkpoint_directory = "checkpoints"
    os.makedirs(checkpoint_directory, exist_ok=True)

    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    metrics_csv_path = os.path.join(log_directory, "dqn_treechop_metrics-longer.csv")

    try:
        for episode_index in range(num_episodes):
            # reset the environment at the start of each episode
            state = env.reset()

            done = False
            episode_step_count = 0

            total_env_reward = 0.0
            total_shaped_reward = 0.0
            episode_loss_values = []

            # tree / chop / pickup metrics
            tree_view_steps = 0
            tree_view_fraction = 0.0
            ever_log_centered = False
            max_centered_attack_streak = 0

            ever_break_window = False
            break_window_active_steps = 0
            max_recent_break_window = 0
            ever_reached_break_threshold = False

            # likely break diagnostics
            ever_likely_broke_log = False
            ever_likely_broke_but_no_reward = False
            likely_broke_step_count = 0

            positive_env_reward_steps = 0
            positive_env_reward_total = 0.0

            # action usage tracking
            action_counts = np.zeros(env.num_actions, dtype=np.int32)

            while not done and episode_step_count < max_steps_per_episode:
                # choose an action using epsilon-greedy exploration
                action = agent.selectAction(state, global_step_count)
                action_counts[action] += 1

                # take one step in the environment
                next_state, shaped_reward, done, info = env.step(action)

                # get the original environment reward for logging
                env_reward = info["env_reward"]

                # update tree-view metrics from the latest environment info
                tree_view_steps = info["tree_view_step_count"]
                tree_view_fraction = info["tree_view_fraction"]
                ever_log_centered = ever_log_centered or info["log_centered"]
                max_centered_attack_streak = max(
                    max_centered_attack_streak,
                    info["centered_attack_streak"],
                )

                # update break / pickup metrics
                recent_break_window = info["recent_break_window"]
                max_recent_break_window = max(
                    max_recent_break_window,
                    recent_break_window,
                )

                if recent_break_window > 0:
                    ever_break_window = True
                    break_window_active_steps += 1

                if info["centered_attack_streak"] >= env.break_streak_threshold:
                    ever_reached_break_threshold = True

                # update likely break diagnostics
                ever_likely_broke_log = (
                    ever_likely_broke_log or info["ever_likely_broke_log"]
                )
                ever_likely_broke_but_no_reward = (
                    ever_likely_broke_but_no_reward
                    or info["likely_broke_but_no_reward"]
                )

                if info["likely_broke_this_step"]:
                    likely_broke_step_count += 1

                # track sparse env reward events
                if env_reward > 0:
                    positive_env_reward_steps += 1
                    positive_env_reward_total += env_reward

                # store the transition in replay memory
                agent.storeTransition(
                    state,
                    action,
                    shaped_reward,
                    next_state,
                    done,
                )

                # run one dqn training update
                loss_value = agent.updateNetwork()
                if loss_value is not None:
                    episode_loss_values.append(loss_value)

                # update running totals
                total_env_reward += env_reward
                total_shaped_reward += shaped_reward

                # move to the next state
                state = next_state

                # update counters
                episode_step_count += 1
                global_step_count += 1

            # compute average loss for the episode if updates happened
            if len(episode_loss_values) > 0:
                average_episode_loss = sum(episode_loss_values) / len(episode_loss_values)
            else:
                average_episode_loss = None

            current_epsilon = agent.getEpsilon(global_step_count)

            most_used_action = int(np.argmax(action_counts))
            most_used_action_count = int(action_counts[most_used_action])

            # save video only if this is the best run so far
            is_best_run = False

            if total_env_reward > best_env_reward:
                is_best_run = True
            elif total_env_reward == best_env_reward:
                if max_centered_attack_streak > best_centered_attack_streak:
                    is_best_run = True
                elif max_centered_attack_streak == best_centered_attack_streak:
                    if tree_view_fraction > best_tree_view_fraction:
                        is_best_run = True

            if is_best_run:
                best_env_reward = total_env_reward
                best_centered_attack_streak = max_centered_attack_streak
                best_tree_view_fraction = tree_view_fraction

                env.saveEpisodeVideo()
                print("    saved video for new best run")
            else:
                env.discardEpisodeVideo()

            # better logging aesthetic
            print(f"episode {episode_index:04d}")
            print(f"    steps: {episode_step_count}")
            print(f"    total env reward: {total_env_reward:.2f}")
            print(f"    total shaped reward: {total_shaped_reward:.2f}")
            print(f"    epsilon: {current_epsilon:.4f}")
            print(f"    tree view steps: {tree_view_steps}")
            print(f"    tree view fraction: {tree_view_fraction:.4f}")
            print(f"    ever log centered: {ever_log_centered}")
            print(f"    max centered attack streak: {max_centered_attack_streak}")
            print(f"    ever reached break threshold: {ever_reached_break_threshold}")
            print(f"    ever break window: {ever_break_window}")
            print(f"    break window active steps: {break_window_active_steps}")
            print(f"    max recent break window: {max_recent_break_window}")
            print(f"    ever likely broke log: {ever_likely_broke_log}")
            print(f"    ever likely broke but no reward: {ever_likely_broke_but_no_reward}")
            print(f"    likely broke step count: {likely_broke_step_count}")
            print(f"    positive env reward steps: {positive_env_reward_steps}")
            print(f"    positive env reward total: {positive_env_reward_total:.2f}")
            print(f"    most used action: {most_used_action}")
            print(f"    most used action count: {most_used_action_count}")
            print(f"    action counts: {action_counts.tolist()}")

            if average_episode_loss is not None:
                print(f"    average loss: {average_episode_loss:.6f}")
            else:
                print(f"    average loss: none")

            # save episode metrics to csv
            episode_metrics = {
                "episode": episode_index,
                "global_step_count": global_step_count,
                "steps": episode_step_count,
                "total_env_reward": total_env_reward,
                "total_shaped_reward": total_shaped_reward,
                "epsilon": current_epsilon,
                "tree_view_steps": tree_view_steps,
                "tree_view_fraction": tree_view_fraction,
                "ever_log_centered": int(ever_log_centered),
                "max_centered_attack_streak": max_centered_attack_streak,
                "ever_reached_break_threshold": int(ever_reached_break_threshold),
                "ever_break_window": int(ever_break_window),
                "break_window_active_steps": break_window_active_steps,
                "max_recent_break_window": max_recent_break_window,
                "ever_likely_broke_log": int(ever_likely_broke_log),
                "ever_likely_broke_but_no_reward": int(ever_likely_broke_but_no_reward),
                "likely_broke_step_count": likely_broke_step_count,
                "positive_env_reward_steps": positive_env_reward_steps,
                "positive_env_reward_total": positive_env_reward_total,
                "most_used_action": most_used_action,
                "most_used_action_count": most_used_action_count,
                "average_loss": average_episode_loss if average_episode_loss is not None else "",
                "is_best_run": int(is_best_run),
            }
            appendEpisodeMetrics(metrics_csv_path, episode_metrics)

            # periodically save checkpoints
            if (episode_index + 1) % checkpoint_every == 0:
                checkpoint_path = os.path.join(
                    checkpoint_directory,
                    f"dqn_treechop_episode_{episode_index + 1}.pt",
                )
                saveCheckpoint(agent, checkpoint_path)
                print(f"    saved checkpoint to {checkpoint_path}")

        # save last checkpoint regardless
        final_checkpoint_path = os.path.join(
            checkpoint_directory,
            "dqn_treechop_final.pt",
        )
        saveCheckpoint(agent, final_checkpoint_path)
        print(f"saved final checkpoint to {final_checkpoint_path}")
        print(f"saved metrics csv to {metrics_csv_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()