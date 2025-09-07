import ctypes
import mmap
import time
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torchvision.transforms as T

class PPOConfig:
    """A class to hold all hyperparameters for the PPO agent."""
    def __init__(self):
        self.mode = 'train'  # 'train' or 'test'
        self.total_timesteps = 1_000_000
        self.learning_rate = 2.5e-4
        self.n_steps = 2048  # Steps to collect per environment before an update
        self.batch_size = 64
        self.n_epochs = 10  # Number of times to iterate over the collected buffer
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # Factor for Generalized Advantage Estimation
        self.clip_coef = 0.2  # PPO clipping coefficient
        self.ent_coef = 0.01  # Entropy bonus coefficient
        self.vf_coef = 0.5  # Value function loss coefficient
        self.max_grad_norm = 0.5  # Max gradient norm for clipping
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = (84, 84) # Resize screenshot for the CNN
        self.model_path = "ppo_godot_agent.pth"
        self.num_test_episodes = 20
        
        # --- Dimensions inferred from C++ structs ---
        self.action_dim = 4 # idle, left, right, jump
        self.vector_obs_dim = 4 # pos_x, pos_y, vel_x, vel_y

# --- PPO Algorithm Implementation ---

class ActorCritic(nn.Module):
    """
    A combined Actor-Critic network that processes both image and vector observations.
    - A CNN processes the image (screenshot).
    - An MLP processes the vector data (player position, velocity).
    - The features are combined and fed into actor (policy) and critic (value) heads.
    """
    def __init__(self, vector_obs_dim, action_dim, image_size=(84, 84)):
        super().__init__()
        
        # 1. CNN for image processing (inspired by Nature DQN paper)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *image_size)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        # 2. Shared MLP for combined features
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + vector_obs_dim, 512),
            nn.ReLU(),
        )
        
        # 3. Actor head: outputs action logits
        self.actor_head = nn.Linear(512, action_dim)
        
        # 4. Critic head: outputs a state value
        self.critic_head = nn.Linear(512, 1)

    def get_value(self, image_obs, vector_obs):
        """Returns the state value V(s)."""
        cnn_features = self.cnn(image_obs)
        combined_features = torch.cat([cnn_features, vector_obs], dim=1)
        shared_features = self.mlp(combined_features)
        return self.critic_head(shared_features)

    def get_action_and_value(self, image_obs, vector_obs, action=None):
        """
        Returns the action, its log probability, entropy, and the state value.
        - If action is None, it samples a new action from the policy.
        """
        cnn_features = self.cnn(image_obs)
        combined_features = torch.cat([cnn_features, vector_obs], dim=1)
        shared_features = self.mlp(combined_features)
        
        logits = self.actor_head(shared_features)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(shared_features)

class RolloutBuffer:
    """Stores trajectories (s, a, r, d, log_p, v) and calculates advantages."""
    def __init__(self, n_steps, vector_obs_dim, image_size, device):
        self.n_steps = n_steps
        self.device = device
        
        # Store images as uint8 to save GPU memory
        self.image_obs = torch.zeros((n_steps, 3, *image_size), dtype=torch.uint8, device=device)
        self.vector_obs = torch.zeros((n_steps, vector_obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((n_steps,), dtype=torch.int64, device=device)
        self.logprobs = torch.zeros((n_steps,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_steps,), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps,), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, image_obs, vector_obs, action, logprob, reward, done, value):
        """Add a new transition to the buffer."""
        self.image_obs[self.ptr] = image_obs
        self.vector_obs[self.ptr] = vector_obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.flatten()
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, done, gamma, gae_lambda):
        """Calculate advantages using GAE after a rollout is complete."""
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Returns an iterator over mini-batches from the buffer."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)
        
        # Normalize advantages across the whole batch
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
        for start in range(0, self.n_steps, batch_size):
            batch_indices = indices[start : start + batch_size]
            
            # Convert images to float and normalize on-the-fly for the batch
            image_obs_batch = self.image_obs[batch_indices].float() / 255.0

            yield (
                image_obs_batch,
                self.vector_obs[batch_indices],
                self.actions[batch_indices],
                self.logprobs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )

class PPOAgent:
    """The main PPO Agent class that orchestrates the training process."""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        self.agent = ActorCritic(config.vector_obs_dim, config.action_dim, config.image_size).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(config.n_steps, config.vector_obs_dim, config.image_size, self.device)

    def learn(self, next_image_obs, next_vector_obs, done):
        """Perform the PPO update step using the data in the rollout buffer."""
        # 1. Calculate advantages and returns
        with torch.no_grad():
            next_value = self.agent.get_value(next_image_obs, next_vector_obs).reshape(1, -1)
            self.buffer.compute_returns_and_advantages(next_value, done, self.config.gamma, self.config.gae_lambda)
        
        # 2. Optimize policy and value network for K epochs
        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                image_batch, vector_batch, actions_batch, logprobs_batch, advantages_batch, returns_batch = batch

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(image_batch, vector_batch, actions_batch)
                logratio = newlogprob - logprobs_batch
                ratio = logratio.exp()

                # Policy loss (clipped surrogate objective)
                pg_loss1 = -advantages_batch * ratio
                pg_loss2 = -advantages_batch * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns_batch) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        # 3. Clear the buffer for the next rollout
        self.buffer.ptr = 0

    def save_model(self, path):
        """Saves the model's state dictionary."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads a model from a state dictionary."""
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

# --- Godot Communication and Environment Wrapper ---

# Constants from the C++ code
SHARED_MEMORY_META = "Local\\Godot_AI_Shared_Memory_Meta_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
SHARED_MEMORY_ACTION = "Local\\Godot_AI_Shared_Memory_Action_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
SHARED_MEMORY_SCREENSHOT = "Local\\Godot_AI_Shared_Memory_Screenshot_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
SHARED_MEMORY_OBSERVATION = "Local\\Godot_AI_Shared_Memory_Observation_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
SEMAPHORE_PYTHON = "Local\\Godot_AI_Semaphore_Python_96c7e30b-4c86-4484-a18e-dacbffde8d72"
SEMAPHORE_GODOT = "Local\\Godot_AI_Semaphore_Godot_96c7e30b-4c86-4484-a18e-dacbffde8d72"
MAX_SCREENSHOT_BUFFER_SIZE = 33177600
TIMEOUT_MS = 60000

# Win32 API constants
SYNCHRONIZE = 0x00100000
SEMAPHORE_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0x00000000

# Mirror the C++ structs in Python using ctypes
class Observation(ctypes.Structure):
    _fields_ = [("player_position_x", ctypes.c_float), ("player_position_y", ctypes.c_float),
                ("velocity_x", ctypes.c_float), ("velocity_y", ctypes.c_float)]

class Meta(ctypes.Structure):
    _fields_ = [("screenshot_width", ctypes.c_uint32), ("screenshot_height", ctypes.c_uint32),
                ("screenshot_format", ctypes.c_int32)]

class Action(ctypes.Structure):
    _fields_ = [("action", ctypes.c_int32), ("reward", ctypes.c_int32),
                ("done", ctypes.c_int8), ("_padding", ctypes.c_int8 * 3)]

class RLServer:
    """
    Manages communication with the Godot client and acts as an environment wrapper.
    """
    def __init__(self, config):
        self.config = config
        self.meta_shm = self.action_shm = self.observation_shm = self.screenshot_shm = None
        self.meta_map = self.action_map = self.observation_map = self.screenshot_map = None
        self.python_semaphore = self.godot_semaphore = None
        self.meta_data = None
        
        # Image transformation pipeline
        self.image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.config.image_size),
            T.ToTensor(), # Converts to [0,1] float and C,H,W
            lambda x: (x * 255).byte() # Convert to uint8 [0,255] for efficient buffer storage
        ])

    def connect(self):
        """Initializes shared memory and semaphores."""
        print("Python server attempting to connect...")
        try:
            self.meta_shm = mmap.mmap(-1, ctypes.sizeof(Meta), SHARED_MEMORY_META)
            self.action_shm = mmap.mmap(-1, ctypes.sizeof(Action), SHARED_MEMORY_ACTION)
            self.observation_shm = mmap.mmap(-1, ctypes.sizeof(Observation), SHARED_MEMORY_OBSERVATION)
            self.screenshot_shm = mmap.mmap(-1, MAX_SCREENSHOT_BUFFER_SIZE, SHARED_MEMORY_SCREENSHOT)
            
            self.meta_map = Meta.from_buffer(self.meta_shm)
            self.action_map = Action.from_buffer(self.action_shm)
            self.observation_map = Observation.from_buffer(self.observation_shm)
            self.screenshot_map = (ctypes.c_ubyte * MAX_SCREENSHOT_BUFFER_SIZE).from_buffer(self.screenshot_shm)

            self.python_semaphore = ctypes.windll.kernel32.OpenSemaphoreW(
                SEMAPHORE_MODIFY_STATE | SYNCHRONIZE, False, SEMAPHORE_PYTHON)
            self.godot_semaphore = ctypes.windll.kernel32.OpenSemaphoreW(
                SEMAPHORE_MODIFY_STATE | SYNCHRONIZE, False, SEMAPHORE_GODOT)

            if not all([self.meta_shm, self.action_shm, self.observation_shm, self.screenshot_shm, self.python_semaphore, self.godot_semaphore]):
                raise ConnectionError("Failed to open one or more shared memory objects or semaphores.")

            print("Python server connected successfully.")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.disconnect()
            return False

    def wait_for_godot(self):
        """Waits for a signal from the Godot semaphore."""
        result = ctypes.windll.kernel32.WaitForSingleObject(self.python_semaphore, TIMEOUT_MS)
        if result != WAIT_OBJECT_0:
            raise TimeoutError("Godot client may have disconnected or timed out.")

    def signal_godot(self):
        """Sends a signal to the Godot semaphore."""
        ctypes.windll.kernel32.ReleaseSemaphore(self.godot_semaphore, 1, None)
        
    def receive_meta_data(self):
        """Waits for and receives the initial metadata from Godot."""
        print("Waiting for metadata from Godot...")
        self.wait_for_godot()
        self.meta_data = {
            "width": self.meta_map.screenshot_width,
            "height": self.meta_map.screenshot_height,
            "format": self.meta_map.screenshot_format,
        }
        print(f"Received metadata: {self.meta_data}")
        self.signal_godot() # Acknowledge receipt
        return True

    def get_state(self):
        """Reads the current state from shared memory and preprocesses it."""
        # 1. Vector observation
        obs_struct = self.observation_map
        vector_obs = np.array([
            obs_struct.player_position_x, obs_struct.player_position_y,
            obs_struct.velocity_x, obs_struct.velocity_y
        ], dtype=np.float32)
        vector_obs_tensor = torch.tensor(vector_obs, dtype=torch.float32, device=self.config.device)

        # 2. Image observation
        if self.meta_data and self.meta_data['format'] == 4: # Assuming Godot's Image.FORMAT_RGB8
            bytes_per_pixel = 3
            w, h = self.meta_data['width'], self.meta_data['height']
            img_size = w * h * bytes_per_pixel
            
            img_np = np.frombuffer(self.screenshot_map, dtype=np.uint8, count=img_size).reshape((h, w, bytes_per_pixel))
            image_obs_tensor = self.image_transform(img_np).to(self.config.device)
        else:
            image_obs_tensor = torch.zeros((3, *self.config.image_size), dtype=torch.uint8, device=self.config.device)

        return image_obs_tensor, vector_obs_tensor

    def step(self, action: int):
        """Sends an action to Godot and returns (next_state, reward, done)."""
        self.action_map.action = action
        self.signal_godot()
        self.wait_for_godot()
        
        reward = self.action_map.reward
        done = self.action_map.done == 1
        
        next_image_obs, next_vector_obs = self.get_state()
        
        return next_image_obs, next_vector_obs, reward, done

    def disconnect(self):
        """Cleans up resources."""
        print("Disconnecting Python server...")
        if self.python_semaphore: ctypes.windll.kernel32.CloseHandle(self.python_semaphore)
        if self.godot_semaphore: ctypes.windll.kernel32.CloseHandle(self.godot_semaphore)
        if self.meta_shm: self.meta_shm.close()
        if self.action_shm: self.action_shm.close()
        if self.observation_shm: self.observation_shm.close()
        if self.screenshot_shm: self.screenshot_shm.close()

# --- Main Training and Testing Loops ---

def train(config, server, agent):
    """The main training loop."""
    print("--- Starting Training ---")
    
    # Wait for the first signal and get the initial state from Godot
    server.wait_for_godot()
    next_image_obs, next_vector_obs = server.get_state()
    next_done = False
    
    num_updates = config.total_timesteps // config.n_steps
    for update in range(1, num_updates + 1):
        start_time = time.time()
        for step in range(config.n_steps):
            global_step = (update - 1) * config.n_steps + step
            
            image_obs, vector_obs = next_image_obs, next_vector_obs
            
            with torch.no_grad():
                # Add batch dimension and normalize image for network
                img_in = image_obs.unsqueeze(0).float() / 255.0
                vec_in = vector_obs.unsqueeze(0)
                action_tensor, logprob, _, value = agent.agent.get_action_and_value(img_in, vec_in)
            
            action = action_tensor.cpu().item()
            next_image_obs, next_vector_obs, reward, next_done = server.step(action)
            
            # Store experience (store the memory-efficient uint8 image tensor)
            agent.buffer.add(image_obs, vector_obs, action_tensor, logprob, reward, next_done, value)

        # After collecting n_steps, perform learning
        agent.learn(
            next_image_obs.unsqueeze(0).float() / 255.0, # Add batch dim and normalize for value prediction
            next_vector_obs.unsqueeze(0),
            next_done
        )
        
        sps = int(config.n_steps / (time.time() - start_time))
        print(f"Update {update}/{num_updates}, Global Step: {global_step+1}, SPS: {sps}")
        
        # Save model periodically
        if update % 20 == 0:
            agent.save_model(config.model_path)
            
    agent.save_model(config.model_path)
    print("--- Training Finished ---")

def test(config, server, agent):
    """The main testing/inference loop."""
    print("--- Starting Testing ---")
    try:
        agent.load_model(config.model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.model_path}. Cannot run test.")
        return
        
    agent.agent.eval() # Set model to evaluation mode (e.g., disables dropout)

    total_rewards = []
    
    # Wait for the initial connection from Godot
    server.wait_for_godot()

    for episode in range(config.num_test_episodes):
        episode_reward = 0
        done = False
        
        image_obs, vector_obs = server.get_state()
        
        while not done:
            with torch.no_grad():
                # For testing, we take the most likely action (deterministic)
                img_in = image_obs.unsqueeze(0).float() / 255.0
                vec_in = vector_obs.unsqueeze(0)
                logits = agent.agent.actor_head(agent.agent.mlp(torch.cat([agent.agent.cnn(img_in), vec_in], dim=1)))
                action = torch.argmax(logits, dim=1).cpu().item()

            next_image_obs, next_vector_obs, reward, done = server.step(action)
            episode_reward += reward
            image_obs, vector_obs = next_image_obs, next_vector_obs

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{config.num_test_episodes}, Reward: {episode_reward}")
        
    print(f"\nTesting finished. Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Agent for Godot")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Set the agent to training or testing mode.')
    args = parser.parse_args()

    config = PPOConfig()
    config.mode = args.mode
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    print(f"Using device: {config.device}")
    
    server = RLServer(config)
    if server.connect():
        if server.receive_meta_data():
            agent = PPOAgent(config)
            try:
                if config.mode == 'train':
                    train(config, server, agent)
                else: # test mode
                    test(config, server, agent)
            except (KeyboardInterrupt, TimeoutError) as e:
                print(f"Loop interrupted: {e}")
            finally:
                server.disconnect()
        else:
            server.disconnect()