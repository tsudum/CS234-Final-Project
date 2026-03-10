import numpy as np
import torch
import torch.optim as optim

from src.agents.PPONetwork import PPONetwork


class PPOAgent:
    """
    proximal policy optimization agent for MineRL treechop

    uses a clipped surrogate objective with GAE advantage estimation
    and an entropy bonus to encourage exploration
    """

    def __init__(
        self,
        state_channels=4,
        num_actions=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        update_epochs=4,
        batch_size=64,
        device=None,
    ):
        self.state_channels = state_channels
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.training_step_count = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = PPONetwork(
            input_channels=self.state_channels,
            num_actions=self.num_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # rollout buffer — cleared after each policy update
        self.clearRolloutBuffer()

    def clearRolloutBuffer(self):
        """
        wipe the rollout buffer at the start of each collection phase
        """
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_dones = []

    def selectAction(self, state):
        """
        sample an action from the policy and return it along with its log prob and value

        Args:
            state: np.array of shape [channels, H, W]
        Returns:
            action:   int
            log_prob: float
            value:    float
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            distribution = self.network.getActionDistribution(state_tensor)
            value = self.network.getValue(state_tensor)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def storeTransition(self, state, action, log_prob, reward, value, done):
        """
        add one transition to the rollout buffer
        """
        self.rollout_states.append(state)
        self.rollout_actions.append(action)
        self.rollout_log_probs.append(log_prob)
        self.rollout_rewards.append(reward)
        self.rollout_values.append(value)
        self.rollout_dones.append(done)

    def computeGAE(self, last_value):
        """
        compute generalized advantage estimates and discounted returns

        Args:
            last_value: float, V(s_T) for bootstrapping if episode did not end
        Returns:
            advantages: np.array of shape [T]
            returns:    np.array of shape [T]
        """
        rewards = np.array(self.rollout_rewards, dtype=np.float32)
        values = np.array(self.rollout_values, dtype=np.float32)
        dones = np.array(self.rollout_dones, dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)

        # bootstrap from last value if the episode is not done
        next_value = last_value
        gae = 0.0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def updatePolicy(self, last_value):
        """
        run PPO update epochs over the collected rollout

        Args:
            last_value: float, V(s_T) for GAE bootstrapping
        Returns:
            mean_policy_loss: float
            mean_value_loss:  float
            mean_entropy:     float
        """
        advantages, returns = self.computeGAE(last_value)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert rollout data to tensors
        states_tensor = torch.tensor(
            np.array(self.rollout_states), dtype=torch.float32, device=self.device
        )
        actions_tensor = torch.tensor(
            self.rollout_actions, dtype=torch.long, device=self.device
        )
        old_log_probs_tensor = torch.tensor(
            self.rollout_log_probs, dtype=torch.float32, device=self.device
        )
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        T = states_tensor.size(0)

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.update_epochs):
            # shuffle indices for mini-batch updates
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # forward pass
                distribution = self.network.getActionDistribution(batch_states)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()
                values = self.network.getValue(batch_states)

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # value function loss
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)

                # combined loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self.training_step_count += 1
        self.clearRolloutBuffer()

        return (
            float(np.mean(policy_losses)),
            float(np.mean(value_losses)),
            float(np.mean(entropies)),
        )