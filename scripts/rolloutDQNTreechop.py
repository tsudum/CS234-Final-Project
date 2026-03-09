import numpy as np
from src.envs.treechopEnv import TreechopEnv


def chooseAction(treechop_env, step_index):
    """
    simple policy used only for debugging rollouts
    """

    if step_index % 40 == 10:
        return 2  # turn left

    if step_index % 40 == 30:
        return 3  # turn right

    
    if step_index % 60 == 20:
        return 5  # look down

    # default action: move forward while attacking
    return 7


def main():
    """
    run one rollout episode to verify the environment works correctly
    """

    env = TreechopEnv()

    observation = env.reset()

    print("rollout started")
    print("observation shape:", observation.shape)
    print("number of actions:", env.num_actions)

    done = False
    step_index = 0

    total_env_reward = 0.0
    total_shaped_reward = 0.0

    while not done:

        action_index = chooseAction(env, step_index)

        observation, reward, done, info = env.step(action_index)

        total_shaped_reward += reward
        total_env_reward += info["env_reward"]

        print(
            f"step {step_index:04d} | "
            f"action {action_index} | "
            f"env_reward {info['env_reward']:.2f} | "
            f"shaped_reward {info['shaped_reward']:.2f}"
        )

        step_index += 1

    print("\nepisode finished")
    print("episode length:", step_index)
    print("total env reward:", total_env_reward)
    print("total shaped reward:", total_shaped_reward)

    env.close()


if __name__ == "__main__":
    main()