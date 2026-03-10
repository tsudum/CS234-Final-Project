import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # paths to the two evaluation csv files
    ppo_csv_path = "logs/eval_ppo_treechop_episodes.csv"
    dqn_csv_path = "logs/eval_dqn_treechop_episodes.csv"

    # directory where we will save the plots
    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)

    # load the csv files
    ppo_df = pd.read_csv(ppo_csv_path)
    dqn_df = pd.read_csv(dqn_csv_path)

    # sort by episode just to make sure the lines are drawn in the correct order
    ppo_df = ppo_df.sort_values("episode")
    dqn_df = dqn_df.sort_values("episode")

    # graph 1: tree_view_fraction vs episode
    plt.figure(figsize=(10, 6))

    plt.plot(
        ppo_df["episode"],
        ppo_df["tree_view_fraction"],
        marker="o",
        linewidth=2,
        label="ppo",
    )

    plt.plot(
        dqn_df["episode"],
        dqn_df["tree_view_fraction"],
        marker="o",
        linewidth=2,
        label="dqn",
    )

    plt.xlabel("episode")
    plt.ylabel("tree view fraction")
    plt.title("tree view fraction vs episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    tree_view_plot_path = os.path.join(output_dir, "tree_view_fraction_comparison.png")
    plt.savefig(tree_view_plot_path, dpi=300)
    plt.close()

    # graph 2: max_centered_attack_streak vs episode
    plt.figure(figsize=(10, 6))

    # clip for better presentation
    ppo_streak = ppo_df["max_centered_attack_streak"].clip(upper=50)
    dqn_streak = dqn_df["max_centered_attack_streak"].clip(upper=50)

    plt.plot(
        ppo_df["episode"],
        ppo_streak,
        marker="o",
        linewidth=2,
        label="ppo",
    )

    plt.plot(
        dqn_df["episode"],
        dqn_streak,
        marker="o",
        linewidth=2,
        label="dqn",
    )

    plt.xlabel("episode")
    plt.ylabel("max centered attack streak (clipped at 50)")
    plt.title("max centered attack streak vs episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    streak_plot_path = os.path.join(output_dir, "max_centered_attack_streak_comparison.png")
    plt.savefig(streak_plot_path, dpi=300)
    plt.close()

    print("saved plots:")
    print(tree_view_plot_path)
    print(streak_plot_path)


if __name__ == "__main__":
    main()