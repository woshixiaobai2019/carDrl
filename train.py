import os
import sys
import random

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import RobotEnv
import torch as th 
import torch.nn as nn
from stable_baselines3.common.policies import BaseFeaturesExtractor
# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space,  features_dim=3136):
        
#         super().__init__(observation_space, features_dim)
        
#         # 处理4通道输入
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#     def forward(self, observations):
#         x = observations.permute(0, 3, 1, 2)
#         return self.cnn(x)
NUM_ENV = 5
LOG_DIR = "logs-robot"
os.makedirs(LOG_DIR, exist_ok=True)
# policy_kwargs = policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
# )

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def make_env(seed=0):
    def _init():
        env = RobotEnv()
        env = ActionMasker(env, RobotEnv.get_action_mask)
        env = Monitor(env)
        return env

    return _init


def main():
    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # Instantiate a PPO agent
    model = MaskablePPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
        device="cuda",
    )
    # model = MaskablePPO.load("trained_models_CNN/ppo_snake_40000000_steps.zip", env=env)
    # Set the save directory
    save_dir = "trained_models_CNN"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 5000  # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_robot")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(200 * 10e4),
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_robot_final.zip"))

if __name__ == "__main__":
    main()
