
#!/usr/bin/env python3 - This is a line that tells the operating system to run this script using Python 3.

import datetime             # Used to manipulate date and time, particularly to create a timestamp for logging purposes.
import numpy as np          # A package for numerical computing in Python. It is used to work with arrays and matrices efficiently.
from mpi4py import MPI      # A Python wrapper for MPI (Message Passing Interface), which allows parallel processing and inter-process communication. 
                            # This is used to distribute the computations across multiple processors for efficiency.

from deep_rl_for_swarms.common import logger, cmd_util                 # These are modules from the deep_rl_for_swarms package, which handle logging and command utility functions, 
                                                                       # likely related to the setup and debugging of the training environment.
                                                                       
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy      # This is the policy function (Multilayer Perceptron, MLP) used by agents in reinforcement 
                                                                       # learning to take actions based on observations.
                                                                       
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi               # The Trust Region Policy Optimization (TRPO) algorithm implementation that works with MPI
                                                                       # for distributed computation.

from ma_envs.envs.point_envs import rendezvous      # This is an environment from deep_rl_for_swarms.ma_envs where agents (drones in this case) interact. 
                                                                       # It simulates drones or agents interacting in a "rendezvous" task, which is a swarm behavior where agents 
                                                                       # meet or align.
#from deep_rl_for_swarms.ma_envs.envs.point_envs import pursuit_evasion

# ------------------------------------------------------------------------------- #
# --------------- MK code - getting state, action, etc data --------------------- #
# ------------------------------------------------------------------------------- #
import json            # Used to handle data in JSON format, making it easy to save logs and state information.
import numpy as np
import gym             # A toolkit for developing and comparing reinforcement learning algorithms. This class wraps the environment to add extra functionality like logging.
import capstone_parameter_file as cpf

import pygame
import random
import threading   # run the training and the visualizer in parallel threads to ensure both work simultaneously without blocking each other.

# The DroneVisualizer class will manage the pygame window and display the drones' positions during each step.
class DroneVisualizer:
    def __init__(self, window_width=1280, window_height=960, drone_radius=5, num_drones=20):
        # Pygame setup
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.drone_radius = drone_radius  # Radius is half of the diameter (5 pixels)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Drone Swarm Visualization")
        self.font = pygame.font.SysFont(None, 16)
        self.clock = pygame.time.Clock()
        self.fps = 30  # Frames per second

        # Initialize the target location
        self.target_location = TargetLocation(window_width, window_height)
        # All drones will be blue
        self.drone_color = (0, 0, 255)  # RGB for blue
        

    def draw_drones(self, drone_positions):
        # Clear the screen with white background
        self.screen.fill((255, 255, 255))

        # Draw the target location
        self.target_location.draw_target(self.screen)
        
        for i, pos in enumerate(drone_positions):
            # Extract x and y from the position
            if len(pos) >= 2:
                x, y = pos[:2]  # Get the first two elements (x, y)
            else:
                raise ValueError(f"Invalid drone position: {pos}")

            # Convert drone positions from environment coordinates to window coordinates
            x = int((x / 100) * self.window_width)  # Assume the environment size is 100x100
            y = int((y / 100) * self.window_height)

            # Draw the drone as a blue circle
            pygame.draw.circle(self.screen, self.drone_color, (x, y), self.drone_radius)

            # Label the drone with its ID
            label = self.font.render(f"Drone {i}", True, (0, 0, 0))  # Black text for ID label
            self.screen.blit(label, (x + 10, y))  # Offset the label slightly to the right of the drone

        pygame.display.update()

    def check_for_quit(self):
        # Allow the user to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def update(self, drone_positions):
        # Check for quit events
        self.check_for_quit()

        # Draw the drones in their current positions
        self.draw_drones(drone_positions)

        # Limit the frame rate
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
        
    # This resets the target location for another run.
    def reset_target(self):
        self.target_location.reset_target()

# This class will manage the generation and display of random target locations.
class TargetLocation:
    def __init__(self, window_width=1280, window_height=960):
        self.window_width = window_width
        self.window_height = window_height
        self.target_position = self.generate_random_target()

    def generate_random_target(self):
        # Generates a random target position within the window.
        x = random.randint(0, self.window_width)
        y = random.randint(0, self.window_height)
        return (x, y)
        
        # Extend the target position to a 5-tuple; fill the remaining with zeros or relevant values
        #return (x, y, 0, 0, 0)

    def draw_target(self, screen):
        # Draws the target as a red circle on the screen.
        target_radius = 10
        pygame.draw.circle(screen, (255, 0, 0), self.target_position, target_radius)

    def reset_target(self):
        # Resets the target to a new random location.
        self.target_position = self.generate_random_target()

# This class extends the gym.Wrapper class to add custom logging and tracking of the environment's state, actions, and rewards. 
# It logs these interactions at each step during training and writes them into a JSON file.
class CustomMonitor(gym.Wrapper):
    # __init__(): Constructor for the CustomMonitor class. It takes the environment (env) and two filenames to store logs and initial state information. 
    # It initializes internal lists for rewards, episode length, and the log data
    def __init__(self, env, log_file='gym_log_data.json', initial_state_file='initial_state.json', num_drones=20):
        # super(CustomMonitor, self).__init__(env): Calls the parent class (gym.Wrapper) constructor to set up the environment.
        super(CustomMonitor, self).__init__(env)
        self.log_file = log_file
        self.initial_state_file = initial_state_file
        self.episode_rewards = []
        self.episode_length = 0
        self.log_data = []            # List to store all log entries
        self.initial_state_data = []  # List to store initial state information

        # Initialize the visualizer
        self.visualizer = DroneVisualizer(num_drones=num_drones)
        
    # reset(): Resets the environment to start a new episode. It also logs the initial state, agent states, and other matrices (like distance_matrix and angle_matrix).
    # If these attributes are not available, it samples from the observation space.
    def reset(self, **kwargs):
        # Resets the environment and initializes logging for a new episode.
        self.episode_rewards = []
        self.episode_length = 0

        # Collect initial state information
        initial_state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        agent_states = self.world.agent_states if hasattr(self.world, 'agent_states') else None
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None
          
        # self.initial_state_data.append(): Adds the initial state data (after conversion from numpy to JSON serializable format) to the list initial_state_data.
        self.initial_state_data.append({
            "initial_state": self.convert_ndarray(initial_state),
            "agent_states": self.convert_ndarray(agent_states),
            "distance_matrix": self.convert_ndarray(distance_matrix),
            "angle_matrix": self.convert_ndarray(distance_matrix)
        })

        return self.env.reset(**kwargs)

# -------------------------------------------------------------------------------------------------------- #
    # Compute a simple direction-based action towards the target.
    def compute_action_towards_target(self, drone_pos, target_pos):
        # Directly subtract the two (5,) tuples/lists element-wise
        #direction_vector = [target_pos[i] - drone_pos[i] for i in range(len(drone_pos))]
        #norm = sum([direction_vector[i] ** 2 for i in range(2)]) ** 0.5  # Only normalize using x, y components
              
        drone_pos_2d = drone_pos[:2]  # Extract the (x, y) coordinates of the drone
        
        # Simple proportional controller towards the target
        direction_vector = np.array(target_pos) - np.array(drone_pos_2d)
        norm = np.linalg.norm(direction_vector)
       
        
        if norm > 1e-3:  # Avoid division by zero and tiny movements
            direction_vector = direction_vector / norm  # Normalize the direction vector
            # Normalize only x, y components, keep others as is
            #direction_vector = [direction_vector[i] / norm if i < 2 else direction_vector[i] for i in range(5)]
        return direction_vector  # This will guide the drone towards the target
# -------------------------------------------------------------------------------------------------------- #

    # step(action): Executes a single step in the environment using the provided action. 
    # It logs the current state, next state, reward, and other environment-specific information like distance and angle matrices. 
    # This information is appended to log_data, which tracks the whole episode.
    def step(self, action):
        # Logs state, action, reward, next state, and done.
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)
        
        # Convert actions list to numpy array before passing it to the environment
        #actions_array = np.array(action)
        #next_state, reward, done, info = self.env.step(actions_array)
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None

# --------------------------------------------------------------------------------------------------------- #
        # Get drone positions (this assumes environment has a get_drone_positions method)
        drone_positions = self.env.get_drone_positions()
             
        # Get the target position
        target_pos = self.visualizer.target_location.target_position
        
        # Compute actions for all drones to move toward the target
        actions = []
        for pos in drone_positions:
            action_to_target = self.compute_action_towards_target(pos, target_pos)
            actions.append(action_to_target)

        # Step the environment with the new actions
        next_state, reward, done, info = self.env.step(actions)
        
        # Update the visualizer with the current drone positions
        self.visualizer.update(drone_positions)
        
        # Log distances to the target
        distances_to_target = []
        for pos in drone_positions:
            x, y = int((pos[0] / 100) * self.visualizer.window_width), int((pos[1] / 100) * self.visualizer.window_height)
            distance = np.linalg.norm(np.array([x, y]) - np.array(target_pos))
            distances_to_target.append(distance)
                               
        # Prepare log entry
        log_entry = {
            "step": self.episode_length,
            "state_sampled": self.convert_ndarray(state),
            "action": self.convert_ndarray(action),
            "reward": reward,
            "next_state_sampled": self.convert_ndarray(next_state),
            "done": done,
            "info": info,
            "distance_matrix": distance_matrix,
            "angle_matrix": angle_matrix,
            "distances_to_target": distances_to_target
        }

        # Append log entry to the list
        self.log_data.append(log_entry)

        # Check if all drones are close enough to the target, then reset the target
        if all(dist < 5 for dist in distances_to_target):  # 5 pixels as threshold
            self.visualizer.reset_target()

        # Update the visualizer with the current drone positions
        self.visualizer.update(drone_positions)

        self.episode_length += 1             # Increments the episode step count.
        self.episode_rewards.append(reward)  # Logs the reward obtained in this step.

        return next_state, reward, done, info

    # Recursively converts numpy arrays to Python lists, which can be stored in JSON format. This ensures all logged data is JSON serializable.
    def convert_ndarray(self, item):
        # Recursively convert numpy arrays to lists.
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: self.convert_ndarray(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self.convert_ndarray(i) for i in item]
        else:
            return item

    # Finalizes and writes all log data and initial state data to JSON files before closing the environment. 
    # This ensures that all data collected during the episode is safely stored for further analysis.
    def close(self):
        # Writes all log entries and initial state data to JSON files and closes the environment.
        # Close the visualizer window
        self.visualizer.close()        
        
        # Convert the log data to be JSON serializable
        serializable_log_data = self.convert_ndarray(self.log_data)
        serializable_initial_state_data = self.convert_ndarray(self.initial_state_data)

        # Print for debugging
        print("Writing log data to", self.log_file)
        print("Writing initial state data to", self.initial_state_file)
        print("Initial state data:", serializable_initial_state_data)

        # Write log data to the log file
        with open(self.log_file, 'w') as f:
            json.dump(serializable_log_data, f, indent=4)

        # Write initial state data to a separate file
        with open(self.initial_state_file, 'w') as f:
            json.dump(serializable_initial_state_data, f, indent=4)

        # Close the environment
        super(CustomMonitor, self).close()


# ------------------------------------------------------------------------------- #
# -------------------------- Return to base code -------------------------------- #
# ------------------------------------------------------------------------------- #

# This function handles the setup and execution of the TRPO (Trust Region Policy Optimization) algorithm to train the agents in the environment.
def train(num_timesteps, log_dir, num_drones=20):
    import deep_rl_for_swarms.common.tf_util as U
    
    # Initializes a TensorFlow session with a single thread to manage the computations.
    sess = U.single_threaded_session()
    # Enters the session context to ensure TensorFlow operations run within the session.
    sess.__enter__()

    # Gets the rank of the current process in the MPI setup. If the rank is 0, it means this is the master process, responsible for logging and managing outputs.
    # logger.configure(): Configures logging for the master process (rank 0). Logs are written in CSV format to the directory log_dir. 
                        # Non-master processes have logging disabled to avoid redundant outputs.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(format_strs=['csv'], dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    # Defines the policy function used by the agents. It creates an MLP-based policy (neural network with hidden layers of size 64). 
    # This policy determines the actions taken by agents based on their observations from the environment.
    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                   hid_size=[64], feat_size=[64])

    # Initializes the environment with 20 agents (drones). The environment simulates a world of size 100x100, with each agent observing summed accelerations (sum_obs_acc). 
    # Other parameters define the communication radius, the grid resolution for distance and bearing, and the movement dynamics (unicycle dynamics).
    env = rendezvous.RendezvousEnv(#nr_agents=20,
                                   nr_agents=num_drones,
                                   obs_mode='sum_obs_acc',
                                   comm_radius=100 * np.sqrt(2),
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle_acc')

    # ------------------------------------------------------------------------------- #
    # ----------------------- MK code - adding custom logger ------------------------ #
    # ------------------------------------------------------------------------------- #
    
    # Initializes the CustomMonitor to wrap the environment for logging purposes. 
    # The log files (gym_log_data.json and initial_state.json) are created in the logging directory
    custom_logger = CustomMonitor(env, log_file=logger.get_dir() + '/gym_log_data.json',
                                  initial_state_file=logger.get_dir() + '/initial_state.json',
                                  num_drones=num_drones)
    # ------------------------------------------------------------------------------- #
    # -------------------------- Return to base code -------------------------------- #
    # ------------------------------------------------------------------------------- #

    # This is the core of the training loop (main agent training loop) where the Trust Region Policy Optimization (TRPO) algorithm is applied. 
    # The learning process iterates over batches of timesteps (timesteps_per_batch=10), optimizing the agents' policies. Key hyperparameters:
        # max_kl=0.01: Maximum allowed KL-divergence between old and new policies (to ensure stable updates).
        # cg_iters=10: Number of conjugate gradient iterations for solving linear systems.
        # gamma=0.99: Discount factor for future rewards.
        # lam=0.98: GAE (Generalized Advantage Estimation) discount factor.
        # vf_iters=5 and vf_stepsize=1e-3: Parameters for optimizing the value function.
    trpo_mpi.learn(custom_logger, policy_fn, timesteps_per_batch=10, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    # Closes the logger and the environment after training is complete, ensuring that all data is saved.
    custom_logger.close()
    env.close()

# This is the main entry point of the script. It generates a unique logging directory based on the current time and starts the training process by calling train().
def main():
    # Get the number of drones from the user
    try:
        num_drones = int(input("Enter the number of drones to run in the rendezvous mission: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return    
    
    env_id = "Rendezvous-v0"
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = cpf.USER_OUTPUT_PATH + env_id + '_' + dstr
    
    #  Start the training process with the user-specified number of drones
    train(num_timesteps=10, log_dir=log_dir, num_drones=num_drones)
    
if __name__ == '__main__':
    main()
