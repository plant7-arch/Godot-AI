import ctypes
import mmap
import time
import numpy as np
from PIL import Image

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
WAIT_TIMEOUT = 0x00000102

# Mirror the C++ structs in Python using ctypes
class Observation(ctypes.Structure):
    _fields_ = [
        ("player_position_x", ctypes.c_float),
        ("player_position_y", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
    ]

class Meta(ctypes.Structure):
    _fields_ = [
        ("screenshot_width", ctypes.c_uint32),
        ("screenshot_height", ctypes.c_uint32),
        ("screenshot_format", ctypes.c_int32),
    ]

class Action(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_int32),
        ("reward", ctypes.c_int32),
        ("done", ctypes.c_int8),
        ("_padding", ctypes.c_int8 * 3),
    ]

class RLServer:
    """
    The Python server that connects to the Godot client via shared memory.
    """
    def __init__(self):
        self.meta_shm = None
        self.action_shm = None
        self.observation_shm = None
        self.screenshot_shm = None
        
        self.meta_map = None
        self.action_map = None
        self.observation_map = None
        self.screenshot_map = None

        self.python_semaphore = None
        self.godot_semaphore = None

        self.meta_data = None

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
                SEMAPHORE_MODIFY_STATE | SYNCHRONIZE, False, SEMAPHORE_PYTHON
            )
            self.godot_semaphore = ctypes.windll.kernel32.OpenSemaphoreW(
                SEMAPHORE_MODIFY_STATE | SYNCHRONIZE, False, SEMAPHORE_GODOT
            )

            if not all([self.meta_shm, self.action_shm, self.observation_shm, self.screenshot_shm, self.python_semaphore, self.godot_semaphore]):
                 raise ConnectionError("Failed to open one or more shared memory objects or semaphores.")

            print("Python server connected successfully.")
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            self.disconnect()
            return False

    def receive_meta_data(self):
        """Waits for and receives the initial metadata from Godot."""
        print("Waiting for metadata from Godot...")
        result = ctypes.windll.kernel32.WaitForSingleObject(self.python_semaphore, TIMEOUT_MS)
        if result != WAIT_OBJECT_0:
            print("Failed to receive metadata. Timeout or error.")
            return False

        self.meta_data = {
            "width": self.meta_map.screenshot_width,
            "height": self.meta_map.screenshot_height,
            "format": self.meta_map.screenshot_format,
        }
        print(f"Received metadata: {self.meta_data}")
        ctypes.windll.kernel32.ReleaseSemaphore(self.godot_semaphore, 1, None)
        return True

    def calculate_next_action(self, observation, screenshot_array):
        """This is where your RL agent's logic goes."""
        "0 == idle"
        "1 == move_left"
        "2 == move_right"
        "3 == jump"
        return np.random.randint(0, 4)

    def run_loop(self, max_episodes=5):
        """The main loop for receiving observations and sending actions."""
        if max_episodes <= 0:
            print("Starting main RL loop to run indefinitely.")
        else:
            print(f"Starting main RL loop. Will run for {max_episodes} episodes.")

        episode_count = 0
        while True:
            result = ctypes.windll.kernel32.WaitForSingleObject(self.python_semaphore, TIMEOUT_MS)
            if result != WAIT_OBJECT_0:
                print("Godot client may have disconnected or timed out. Exiting.")
                break

            # --- MODIFIED LOGIC ---
            # Check if the episode has ended
            if self.action_map.done == 1:
                episode_count += 1
                if max_episodes > 0:
                    print(f"Episode {episode_count}/{max_episodes} finished.")
                    if episode_count >= max_episodes:
                        print("Maximum number of episodes reached. Shutting down.")
                        # This break will cause the disconnect() method to be called.
                        # Godot will timeout on its next wait and should handle it.
                        break
                else:
                    print(f"Episode {episode_count} finished. Continuing indefinitely.")
            # --- END MODIFIED LOGIC ---

            observation = self.observation_map
            
            if self.meta_data and self.meta_data['format'] == 4:
                bytes_per_pixel = 3
                img_size = self.meta_data['width'] * self.meta_data['height'] * bytes_per_pixel
                screenshot_data = self.screenshot_map[:img_size]
            else:
                screenshot_data = bytearray() 

            action = self.calculate_next_action(observation, bytes(screenshot_data))
            self.action_map.action = action
            ctypes.windll.kernel32.ReleaseSemaphore(self.godot_semaphore, 1, None)

    def disconnect(self):
        """Cleans up resources."""
        print("Disconnecting Python server...")

        self.meta_map = None
        self.action_map = None
        self.observation_map = None
        self.screenshot_map = None

        if self.python_semaphore:
            ctypes.windll.kernel32.CloseHandle(self.python_semaphore)
        if self.godot_semaphore:
            ctypes.windll.kernel32.CloseHandle(self.godot_semaphore)
        
        if self.meta_shm: self.meta_shm.close()
        if self.action_shm: self.action_shm.close()
        if self.observation_shm: self.observation_shm.close()
        if self.screenshot_shm: self.screenshot_shm.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set the number of episodes to run.
    # Set to 0 or a negative number to run forever until you stop it manually (Ctrl+C).
    NUM_EPISODES_TO_RUN = 10
    
    server = RLServer()
    if server.connect():
        if server.receive_meta_data():
            try:
                server.run_loop(max_episodes=NUM_EPISODES_TO_RUN)
            except KeyboardInterrupt:
                print("Server stopped by user.")
            finally:
                server.disconnect()
        else:
            server.disconnect()

