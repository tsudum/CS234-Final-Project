import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import os


class BCTrainer:
    def __init__(
        self,
        network,
        dataset,
        device="cpu",
        learning_rate=1e-4,
        batch_size=64,
        num_actions=7,
    ):
        self.network = network.to(device)
        self.device = device
        self.batch_size = batch_size

        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        # compute class weights inversely proportional to frequency
        counts = Counter(dataset.actions)
        total = len(dataset.actions)
        weights = torch.FloatTensor([
            total / (num_actions * counts.get(i, 1)) for i in range(num_actions)
        ]).to(device)

        print("class weights:")
        labels = ["forward", "look_up", "look_down", "turn_right", "attack", "fwd+attack", "fwd_pickup"]
        for i, (label, w) in enumerate(zip(labels, weights)):
            print(f"  action {i} ({label}): weight={w:.3f}, count={counts.get(i, 0)}")

        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def train(self, dataset, num_epochs=5, save_path="checkpoints/bc_pretrained.pt"):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        print(f"training BC for {num_epochs} epochs on {len(dataset)} samples...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            self.network.train()
            for batch_idx, (obs, actions) in enumerate(loader):
                obs = obs.to(self.device)
                actions = actions.squeeze(1).to(self.device)

                # get action logits from actor head
                action_logits, _ = self.network(obs)

                loss = self.criterion(action_logits, actions)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                correct += (action_logits.argmax(1) == actions).sum().item()
                total += len(actions)

                if batch_idx % 100 == 0:
                    print(f"  epoch {epoch+1}/{num_epochs} | batch {batch_idx}/{len(loader)} | loss {loss.item():.4f}")

            acc = correct / total * 100
            avg_loss = total_loss / len(loader)
            print(f"epoch {epoch+1}/{num_epochs} | avg loss {avg_loss:.4f} | accuracy {acc:.1f}%")

        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.network.state_dict(), save_path)
        print(f"saved BC pretrained weights to {save_path}")