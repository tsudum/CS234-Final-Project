import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.envs.treechopEnv import TreechopEnv
from src.agents.PPOAgent import PPOAgent



def saveCheckpoint(agent, checkpoint_path):
    """
    save the network parameters
    """
    checkpoint_directory = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_directory, exist_ok=True)

    torch.save(agent.network.state_dict(), checkpoint_path)


def main():
    """
    train a PPO agent on MineRL treechop
    """

    # training hyperparameters
    num_episodes = 20
    max_steps_per_episode = 1000
    rollout_length = 256 
    checkpoint_every = 10

    env = TreechopEnv(max_episode_steps=max_steps_per_episode)

    agent = PPOAgent(
        state_channels=env.frame_stack,
        num_actions=env.num_actions,
        device="cpu",
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        update_epochs=4,
        batch_size=64,
    )

    global_step_count = 0

    checkpoint_directory = "checkpoints"
    os.makedirs(checkpoint_directory, exist_ok=True)

    try:
        for episode_index in range(num_episodes):
            state = env.reset()
            done = False
            episode_step_count = 0

            total_env_reward = 0.0
            total_shaped_reward = 0.0
            episode_policy_losses = []
            episode_value_losses = []
            episode_entropies = []

            tree_view_steps = 0
            tree_view_fraction = 0.0
            ever_log_centered = False
            max_centered_attack_streak = 0
            ever_break_window = False
            break_window_active_steps = 0
            max_recent_break_window = 0
            ever_reached_break_threshold = False
            positive_env_reward_steps = 0
            positive_env_reward_total = 0.0

            action_counts = np.zeros(env.num_actions, dtype=np.int32)

            steps_since_update = 0

            while not done and episode_step_count < max_steps_per_episode:
                action, log_prob, value = agent.selectAction(state)
                action_counts[action] += 1

                next_state, shaped_reward, done, info = env.step(action)

                env_reward = info["env_reward"]

                # update metrics
                tree_view_steps = info["tree_view_step_count"]
                tree_view_fraction = info["tree_view_fraction"]
                ever_log_centered = ever_log_centered or info["log_centered"]
                max_centered_attack_streak = max(
                    max_centered_attack_streak,
                    info["centered_attack_streak"],
                )

                recent_break_window = info["recent_break_window"]
                max_recent_break_window = max(max_recent_break_window, recent_break_window)

                if recent_break_window > 0:
                    ever_break_window = True
                    break_window_active_steps += 1

                if info["centered_attack_streak"] >= env.break_streak_threshold:
                    ever_reached_break_threshold = True

                if env_reward > 0:
                    positive_env_reward_steps += 1
                    positive_env_reward_total += env_reward

                agent.storeTransition(
                    state,
                    action,
                    log_prob,
                    shaped_reward,
                    value,
                    done,
                )

                total_env_reward += env_reward
                total_shaped_reward += shaped_reward
                state = next_state
                episode_step_count += 1
                global_step_count += 1
                steps_since_update += 1

                if steps_since_update >= rollout_length or done:
                    if done:
                        last_value = 0.0
                    else:
                        last_state_tensor = torch.tensor(
                            state, dtype=torch.float32, device=agent.device
                        ).unsqueeze(0)
                        with torch.no_grad():
                            last_value = agent.network.getValue(last_state_tensor).item()

                    policy_loss, value_loss, entropy = agent.updatePolicy(last_value)
                    episode_policy_losses.append(policy_loss)
                    episode_value_losses.append(value_loss)
                    episode_entropies.append(entropy)
                    steps_since_update = 0

            # compute episode averages
            avg_policy_loss = np.mean(episode_policy_losses) if episode_policy_losses else None
            avg_value_loss = np.mean(episode_value_losses) if episode_value_losses else None
            avg_entropy = np.mean(episode_entropies) if episode_entropies else None

            most_used_action = int(np.argmax(action_counts))
            most_used_action_count = int(action_counts[most_used_action])

            print(f"episode {episode_index:04d}")
            print(f"  steps: {episode_step_count}")
            print(f"  total env reward: {total_env_reward:.2f}")
            print(f"  total shaped reward: {total_shaped_reward:.2f}")
            print(f"  tree view steps: {tree_view_steps}")
            print(f"  tree view fraction: {tree_view_fraction:.4f}")
            print(f"  ever log centered: {ever_log_centered}")
            print(f"  max centered attack streak: {max_centered_attack_streak}")
            print(f"  ever reached break threshold: {ever_reached_break_threshold}")
            print(f"  ever break window: {ever_break_window}")
            print(f"  break window active steps: {break_window_active_steps}")
            print(f"  max recent break window: {max_recent_break_window}")
            print(f"  positive env reward steps: {positive_env_reward_steps}")
            print(f"  positive env reward total: {positive_env_reward_total:.2f}")
            print(f"  most used action: {most_used_action}")
            print(f"  most used action count: {most_used_action_count}")
            print(f"  action counts: {action_counts.tolist()}")

            if avg_policy_loss is not None:
                print(f"  avg policy loss: {avg_policy_loss:.6f}")
                print(f"  avg value loss: {avg_value_loss:.6f}")
                print(f"  avg entropy: {avg_entropy:.6f}")
            else:
                print("  no updates this episode")

            if (episode_index + 1) % checkpoint_every == 0:
                checkpoint_path = os.path.join(
                    checkpoint_directory,
                    f"ppo_treechop_episode_{episode_index + 1}.pt",
                )
                saveCheckpoint(agent, checkpoint_path)
                print(f"  saved checkpoint to {checkpoint_path}")

        final_checkpoint_path = os.path.join(
            checkpoint_directory,
            "ppo_treechop_final.pt",
        )
        saveCheckpoint(agent, final_checkpoint_path)
        print(f"saved final checkpoint to {final_checkpoint_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()