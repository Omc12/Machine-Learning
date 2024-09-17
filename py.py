from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os
import time

# Set Tesseract OCR path if not set in environment variables
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Custom Environment for Web Game
class WebGame(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 700, 'height': 100}
        self.done_location = {'top': 405, 'left': 630, 'width': 660, 'height': 70}
        self.total_episodes = 0
        self.successful_episodes = 0
        self.current_frame = None

    def step(self, action):
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pydirectinput.press(action_map[action])

        time.sleep(0.1)  # Allow some time for the action to take effect

        done, done_cap = self.get_done()
        observation = self.get_observation()
        reward = 1 if not done else -100  # Negative reward for game over

        if done:
            self.total_episodes += 1
            if done_cap is not None:
                self.successful_episodes += 1

        accuracy = self.successful_episodes / max(1, self.total_episodes)  # Calculate accuracy
        info = {'accuracy': accuracy}
        return observation, reward, done, info

    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()

    def render(self):
        if self.current_frame is not None:
            cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        channel = np.reshape(resized, (1, 83, 100))
        self.current_frame = gray  # Update current frame for rendering
        return channel

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))
        done_strings = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done, done_cap

# Custom Callback for Logging and Saving Model
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
 
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.num_timesteps}_steps.zip')
            self.model.save(model_path)

            # Log accuracy
            if hasattr(self.model.env, 'total_episodes') and hasattr(self.model.env, 'successful_episodes'):
                accuracy = self.model.env.successful_episodes / max(1, self.model.env.total_episodes)
                print(f'Step: {self.num_timesteps}, Accuracy: {accuracy:.2f}')

        return True  # Return True to continue training

# Directories for saving checkpoints and logs
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Create WebGame environment
env = WebGame()

# Define and train DQN model
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=10000, learning_starts=1000, exploration_fraction=0.1, exploration_final_eps=0.01, target_update_interval=500, train_freq=4)

# Define callback for logging and saving checkpoints
callback = TrainAndLoggingCallback(check_freq=300, save_path=CHECKPOINT_DIR)

# Train the model
model.learn(total_timesteps=3000, callback=callback)
 