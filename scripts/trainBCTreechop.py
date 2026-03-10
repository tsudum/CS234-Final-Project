import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.treechopEnv import TreechopEnv
from src.agents.PPONetwork import PPONetwork
from src.agents.BCDataset import BCDataset
from src.agents.BCTrainer import BCTrainer


def main():
    env = TreechopEnv()
    num_actions = env.num_actions
    frame_stack = env.frame_stack
    env.close()

    print(f"num_actions: {num_actions}, frame_stack: {frame_stack}")

    # load dataset
    dataset = BCDataset(
        data_dir="data/MineRLTreechop-v0",
        image_size=64,
        frame_stack=frame_stack,
        max_demos=None,       # use all 210 demos
        max_samples=200000,   # cap at 200k transitions
    )

    # build network — same architecture as PPO
    network = PPONetwork(
        input_channels=frame_stack,
        num_actions=num_actions,
    )

    # trainer = BCTrainer(
    #     network=network,
    #     device="cpu",
    #     learning_rate=1e-4,
    #     batch_size=64,
    # )

    # trainer.train(
    #     dataset=dataset,
    #     num_epochs=5,
    #     save_path="checkpoints/bc_pretrained.pt",
    # )

    trainer = BCTrainer(
        network=network,
        dataset=dataset,
        device="cpu",
        learning_rate=1e-4,
        batch_size=64,
        num_actions=num_actions,
    )

    trainer.train(
        dataset=dataset,
        num_epochs=5,
        save_path="checkpoints/bc_pretrained.pt",
    )
    
    print("BC pretraining done. now run trainPPOTreechop.py with --bc-init to fine-tune.")


if __name__ == "__main__":
    main()