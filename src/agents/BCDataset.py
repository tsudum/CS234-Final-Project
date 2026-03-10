import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


# maps expert continuous actions to our 7 discrete actions
# action 0: forward
# action 1: camera pitch=-10 (look up)
# action 2: camera pitch=+10 (look down)
# action 3: camera yaw=+10 (turn right)
# action 4: attack
# action 5: forward + attack
# action 6: forward (pickup macro)
def mapExpertActionToDiscrete(forward, attack, camera):
    pitch = camera[0]
    yaw = camera[1]

    if attack == 1 and forward == 1:
        return 5
    if attack == 1:
        return 4
    if abs(yaw) > abs(pitch):
        if yaw > 5:
            return 3   # turn right
        if yaw < -5:
            return 1   # turn left (reuse action 1)
    if pitch > 5:
        return 2       # look down
    if pitch < -5:
        return 1       # look up
    if forward == 1:
        return 0
    return 0           # default forward


def preprocessFrame(frame_bgr, image_size=64):
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    return normalized  # shape (64, 64)


class BCDataset(Dataset):
    def __init__(
        self,
        data_dir="data/MineRLTreechop-v0",
        image_size=64,
        frame_stack=4,
        max_demos=None,
        max_samples=100000,
    ):
        self.image_size = image_size
        self.frame_stack = frame_stack

        self.observations = []  # list of np arrays (frame_stack, H, W)
        self.actions = []       # list of ints

        demo_dirs = sorted(os.listdir(data_dir))
        if max_demos is not None:
            demo_dirs = demo_dirs[:max_demos]

        print(f"loading {len(demo_dirs)} demos from {data_dir}...")

        total_samples = 0
        for demo_name in demo_dirs:
            if total_samples >= max_samples:
                break

            demo_path = os.path.join(data_dir, demo_name)
            npz_path = os.path.join(demo_path, "rendered.npz")
            mp4_path = os.path.join(demo_path, "recording.mp4")

            if not os.path.exists(npz_path) or not os.path.exists(mp4_path):
                continue

            try:
                npz = np.load(npz_path, allow_pickle=True)
                forwards = npz["action$forward"]
                attacks = npz["action$attack"]
                cameras = npz["action$camera"]
                num_steps = len(forwards)

                cap = cv2.VideoCapture(mp4_path)
                if not cap.isOpened():
                    continue

                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(preprocessFrame(frame, image_size))
                cap.release()

                if len(frames) < frame_stack:
                    continue

                # align frames with actions (use min length)
                usable = min(len(frames), num_steps)

                # build frame stack for each timestep
                frame_buffer = [frames[0]] * frame_stack
                for t in range(usable):
                    if total_samples >= max_samples:
                        break

                    frame_buffer.pop(0)
                    frame_buffer.append(frames[t])

                    stacked = np.stack(frame_buffer, axis=0)  # (4, 64, 64)

                    action = mapExpertActionToDiscrete(
                        int(forwards[t]),
                        int(attacks[t]),
                        cameras[t],
                    )

                    self.observations.append(stacked.copy())
                    self.actions.append(action)
                    total_samples += 1

            except Exception as e:
                print(f"  skipping {demo_name}: {e}")
                continue

        print(f"loaded {len(self.observations)} transitions")

        # print action distribution
        from collections import Counter
        labels = ["forward", "look_up", "look_down", "turn_right", "attack", "fwd+attack", "fwd_pickup"]
        counts = Counter(self.actions)
        for i, label in enumerate(labels):
            print(f"  action {i} ({label}): {counts.get(i, 0)}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.FloatTensor(self.observations[idx])  # (4, 64, 64)
        act = torch.LongTensor([self.actions[idx]])
        return obs, act