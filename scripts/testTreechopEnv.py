from src.envs.treechopEnv import TreechopEnv


def main():
    env = TreechopEnv()

    observation = env.reset()

    print("env reset successful")
    print("observation shape:", observation.shape)
    print("num actions:", env.num_actions)

    total_reward = 0
    done = False

    while not done:
        action = env.sampleRandomAction()

        observation, reward, done, info = env.step(action)
        print("info keys: ", info.keys())
        total_reward += reward

        print(
            "step:", info["episode_step"],
            "| reward:", reward
        )

    print("\nepisode finished")
    print("total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()