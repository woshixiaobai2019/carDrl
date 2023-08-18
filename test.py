import time
import random

from sb3_contrib import MaskablePPO
import imageio
from env import RobotEnv
MODEL_PATH = r"trained_models_CNN/ppo_robot_1875000_steps"

NUM_EPISODE = 20

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

env = RobotEnv()

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

for episode in range(10):
    obs = env.reset()
    episode_reward = 0
    done = False
    num_step = 0
    print(f"=================== Episode {episode + 1} ==================")
    step_counter = 0
    trace = []
    while True:
        trace.append(obs[:,:-1])
        mask = env.get_action_mask()
        action, _ = model.predict(obs, action_masks=mask)
        num_step += 1
        obs, reward, done, info = env.step(int(action))
        info["action"] = action
        if done:
            print(info)
            break
        episode_reward += reward
    imageio.mimsave(f"test_video/epo-{episode}.mp4",trace,format="mp4",fps=10)

    print(f"total_reward:{episode_reward}")


