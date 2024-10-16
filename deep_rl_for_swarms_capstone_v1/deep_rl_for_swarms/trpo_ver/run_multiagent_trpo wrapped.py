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

from ma_envs.envs.point_envs import rendezvous2      # This is an environment from deep_rl_for_swarms.ma_envs where agents (drones in this case) interact. 
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


# This class extends the gym.Wrapper class to add custom logging and tracking of the environment's state, actions, and rewards. 
# It logs these interactions at each step during training and writes them into a JSON file.
class CustomMonitor(gym.Wrapper):
    # __init__(): Constructor for the CustomMonitor class. It takes the environment (env) and two filenames to store logs and initial state information. 
    # It initializes internal lists for rewards, episode length, and the log data
    def __init__(self, env, log_file='gym_log_data.json', initial_state_file='initial_state.json'):
        # super(CustomMonitor, self).__init__(env): Calls the parent class (gym.Wrapper) constructor to set up the environment.
        super(CustomMonitor, self).__init__(env)
        self.log_file = log_file
        self.initial_state_file = initial_state_file
        self.episode_rewards = []
        self.episode_length = 0
        self.log_data = []            # List to store all log entries
        self.initial_state_data = []  # List to store initial state information

    # reset(): Resets the environment to start a new episode. It also logs the initial state, agent states, and other matrices (like distance_matrix and angle_matrix).
    # If these attributes are not available, it samples from the observation space.
    def reset(self, **kwargs):
        """Resets the environment and initializes logging for a new episode."""
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


    # step(action): Executes a single step in the environment using the provided action. 
    # It logs the current state, next state, reward, and other environment-specific information like distance and angle matrices. 
    # This information is appended to log_data, which tracks the whole episode.
    def step(self, action):
        """Logs state, action, reward, next state, and done."""
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None


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
            "angle_matrix": angle_matrix
        }

        # Append log entry to the list
        self.log_data.append(log_entry)

        self.episode_length += 1             # Increments the episode step count.
        self.episode_rewards.append(reward)  # Logs the reward obtained in this step.

        return next_state, reward, done, info

    # Recursively converts numpy arrays to Python lists, which can be stored in JSON format. This ensures all logged data is JSON serializable.
    def convert_ndarray(self, item):
        """Recursively convert numpy arrays to lists."""
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
        """Writes all log entries and initial state data to JSON files and closes the environment."""
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
def train(num_timesteps, log_dir):
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
    env = rendezvous2.RendezvousEnv(nr_agents=20,
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
                                  initial_state_file=logger.get_dir() + '/initial_state.json')
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
    env_id = "Rendezvous-v0"
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = cpf.USER_OUTPUT_PATH + env_id + '_' + dstr
    train(num_timesteps=10, log_dir=log_dir)


if __name__ == '__main__':
    main()