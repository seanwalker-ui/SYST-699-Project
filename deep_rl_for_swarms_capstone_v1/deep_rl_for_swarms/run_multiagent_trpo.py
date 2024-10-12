#!/usr/bin/env python3
import datetime
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from deep_rl_for_swarms.common import logger, cmd_util
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous
#from deep_rl_for_swarms.ma_envs.envs.point_envs import pursuit_evasion


# --------------- SW code - Initialize Drone RAI data --------------------------- #
# Input RAI features
#from responsible_features import ResponsibleAI
# Initialize the ResponsibleAI instance
#responsible_ai = ResponsibleAI(min_safe_distance=5, alert_distance=2)

# Input Grid matrix
#from virtualmap import GridEnvironment
# --------------- ----------------------------------- --------------------------- #

# ------------------------------------------------------------------------------- #
# --------------- MK code - getting state, action, etc data --------------------- #
# ------------------------------------------------------------------------------- #
import json
import numpy as np
import gym
import csv
import capstone_parameter_file as cpf

class CustomMonitor(gym.Wrapper):
    def __init__(self, env, log_file='gym_log_data.json', initial_state_file='initial_state.json'):
        super(CustomMonitor, self).__init__(env)
        self.log_file = log_file
        self.initial_state_file = initial_state_file
        self.episode_rewards = []
        self.episode_length = 0
        self.log_data = []  # List to store all log entries
        self.initial_state_data = []  # List to store initial state information
        self.responsible_ai = ResponsibleAI(min_safe_distance=5, alert_distance=2)  # List to store all RAI info

    def reset(self, **kwargs):
        """Resets the environment and initializes logging for a new episode."""
        self.episode_rewards = []
        self.episode_length = 0

        # Collect initial state information
        initial_state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        agent_states = self.world.agent_states if hasattr(self.world, 'agent_states') else None
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None
        self.initial_state_data.append({
            "initial_state": self.convert_ndarray(initial_state),
            "agent_states": self.convert_ndarray(agent_states),
            "distance_matrix": self.convert_ndarray(distance_matrix),
            "angle_matrix": self.convert_ndarray(distance_matrix)
        })

        return self.env.reset(**kwargs)

    def step(self, action):
        """Logs state, action, reward, next state, and done."""
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None

        # ------------------------------------------------------------------------------- #
        # --------------- SW code - get Drone RAI data -----------------------------------#
        # ------------------------------------------------------------------------------- #
        # Check for responsibility features here
        drone_positions = self.env.get_drone_positions()  # Created this method in rendezvous.py
        obstacle_positions = self.env.get_obstacle_positions()  # Created this method in rendezvous.py

        # Use ResponsibleAI to check for safety
        self.responsible_ai.avoid_collisions(drone_positions, obstacle_positions)
        self.responsible_ai.ensure_ethics(drone_positions)

        # Log decisions
        for drone_id in range(len(drone_positions)):
            decision = f"Drone {drone_id} moved to position {drone_positions[drone_id]}"
            self.responsible_ai.log_decision(drone_id, decision)

        # ------------------------------Back to MK Code -------------------------------- #
        
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

        self.episode_length += 1
        self.episode_rewards.append(reward)

        return next_state, reward, done, info

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

def train(num_timesteps, log_dir):
    import deep_rl_for_swarms.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

# ------------------------------------ SW code - Initialize variables before Tensorflow runs -------- #
    # Initialize all variables
    #sess.run(tf.global_variables_initializer())
# ------------------------------------ Return to base code ------------------------------------------ #

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(format_strs=['csv'], dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    # The policy_fn is defined to create the agents' policy using mlp_mean_embedding_policy.MlpPolicy. 
    # This policy guides how the agents select actions given observations from the environment.
    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                   hid_size=[64], feat_size=[64])
        
# ------------------------------------ SW code - Update Policy ----------------------------------------------- #
        """
        policy = mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=[64], feat_size=[64])
                
        # Return a function that directly calls get_action() on the policy object. This hides the creation of the policy instance within the policy_fn 
        # and allowing ythe code to call get_action() directly.
        def get_action_fn(observation, stochastic = True):
            #return policy.get_action(observation)  # Call get_action directly on the policy
            #action, value, neglogp = policy.act(observation)  # Call act() directly on the policy
            observation = np.expand_dims(observation, axis=0)  # Add batch dimension
            action, value, neglogp = policy.act(stochastic, observation)
            return action, value, neglogp
        return get_action_fn  # Return the function
        """

# ------------------------------------ Return to Base Code ----------------------------------------------- #  
  
    # The environment is created using the rendezvous.RendezvousEnv class and wrapped in a CustomMonitor, which logs data for each step (state, action, reward, etc.).
    env = rendezvous.RendezvousEnv(nr_agents=20,
                                   obs_mode='sum_obs_acc',
                                   comm_radius=100 * np.sqrt(2),
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle_acc')

    # ------------------------------------ SW code - Initialize Grid Environment ------------------------- #

    # Create an instance of GridEnvironment
    grid_env = GridEnvironment(size=100, obstacle_probability=0.1, num_drones=env.nr_agents, policy=policy_fn, env=env)
    
    # Generate the grid environment with obstacles
    grid_env.generate_environment()
    
    # Visualize the drone movements
    grid_env.visualize_grid()
    # ------------------------------------ Return to base code -------------------------------------------- #
    
    # ------------------------------------------------------------------------------- #
    # ----------------------- SW code - No Fly Zone Check --------------------------- #
    # ------------------------------------------------------------------------------- #
    """
    # Start main training loop
    for episode in range(num_episodes):
        done = False
        state = env.reset()  # Initial state
        drone_positions = env.get_drone_positions()  # Get initial drone positions
        ## Remove if necessary
        action_fn = policy_fn('policy', env.observation_space, env.action_space)  # Get the action function
        
        while not done:
            for drone_id in range(num_drones):
                # Get action from the policy
                #action = policy_fn.get_action(state[drone_id])
                action = action_fn(state[drone_id])
                
                # Execute action in the environment
                next_state, reward, done, _ = env.step(action)

                # Check if the drone is in a no-fly zone and adjust path if necessary
                env.avoid_no_fly_zones(drone_positions)

                # Store the next state
                state[drone_id] = next_state
"""
    # ------------------------------------------------------------------------------- #
    # ----------------------- MK code - adding custom logger ------------------------ #
    # ------------------------------------------------------------------------------- #
    custom_logger = CustomMonitor(env, log_file=logger.get_dir() + '/gym_log_data.json',
                                  initial_state_file=logger.get_dir() + '/initial_state.json')
    # ------------------------------------------------------------------------------- #
    # -------------------------- Return to base code -------------------------------- #
    # ------------------------------------------------------------------------------- #

    # Main agent decision loop. responsible for the core reinforcement learning loop, where the agents (drones) interact with the environment, 
    # make decisions (actions), and update their policies based on the rewards they receive.
    
    # The actual decision-making and learning loop is abstracted inside trpo_mpi.learn. It is responsible for iterating through timesteps, selecting actions, applying them
    # to the environment, and updating the policy based on rewards and observations. This is where the decisions happen, but this function is part of the TRPO
    # (Trust Region Policy Optimization) algorithm.
    trpo_mpi.learn(custom_logger, policy_fn, timesteps_per_batch=10, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    # Visualize the grid after training
    grid_env.visualize_grid()
    custom_logger.close()
    env.close()
    
    


def main():
    env_id = "Rendezvous-v0"
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = cpf.USER_OUTPUT_PATH + env_id + '_' + dstr
    
# -------------------------------- SW - Hardcoded Episodes and Drone Number for No Fly Zones ----------------------------------------------- #
    #num_episodes = 100
    #num_drones = 20
    #train(num_timesteps=10, log_dir=log_dir, num_episodes=num_episodes, num_drones=num_drones)
# ------------------------------------------------------------------------------------------------------------------------- #    
    #train(num_timesteps=1e7, log_dir=log_dir)
    train(num_timesteps=10, log_dir=log_dir)

if __name__ == '__main__':
    main()
