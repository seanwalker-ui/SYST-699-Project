#!/usr/bin/env python3
import datetime
import numpy as np
import csv
import pygame  # Import pygame for visualization
from mpi4py import MPI
from deep_rl_for_swarms.common import logger
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous


# Pygame window dimensions
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960
BACKGROUND_COLOR = (0, 0, 0)  # Black background
DRONE_COLOR = (0, 0, 255)  # Blue for drones
LABEL_COLOR = (255, 255, 255)  # White for labels

# Function to map world position (x, y) to screen coordinates
def map_position_to_screen(x, y, world_size):
    scale_x = WINDOW_WIDTH / world_size
    scale_y = WINDOW_HEIGHT / world_size
    screen_x = int(x * scale_x)
    screen_y = int(y * scale_y)
    return screen_x, screen_y

# Function to draw drones on pygame window
def draw_drones(screen, drone_positions, world_size, font):
    screen.fill(BACKGROUND_COLOR)
    for i, pos in enumerate(drone_positions):
        x, y = pos
        screen_x, screen_y = map_position_to_screen(x, y, world_size)
        pygame.draw.circle(screen, DRONE_COLOR, (screen_x, screen_y), 5)
        
        # Render the drone ID next to the drone
        label = font.render(str(i), True, LABEL_COLOR)
        screen.blit(label, (screen_x + 10, screen_y - 10))  # Offset the label slightly
        
    pygame.display.flip()

# Step 1: Extend RendezvousEnv to include position tracking
class CustomRendezvousEnv(rendezvous.RendezvousEnv):
    def __init__(self, *args, **kwargs):
        super(CustomRendezvousEnv, self).__init__(*args, **kwargs)
        # Add a position tracking attribute for each agent
        self.agent_positions = np.zeros((self.nr_agents, 2))  # Store (x, y) positions

    def reset(self):
        obs = super(CustomRendezvousEnv, self).reset()  # Call parent reset method
        # Initialize agent positions randomly in the world space (for example)
        self.agent_positions = np.random.uniform(low=0, high=self.world_size, size=(self.nr_agents, 2))
        return obs

    def step(self, actions):
        obs, reward, done, info = super(CustomRendezvousEnv, self).step(actions)
        # Update agent positions based on their dynamics (this is a simple example)
        for i, agent in enumerate(self.agents):
            # Update position based on velocity (for simplicity, assume p_vel is the velocity)
            if agent.state.p_vel is not None:
                # Ensure self.agent_positions[i] is always an iterable of length 2 (x, y)
                self.agent_positions[i] += agent.state.p_vel  # Simple position update based on velocity
        return obs, reward, done, info

# Step 2: Modify the logging function to include position tracking
def log_drone_states(env, csv_writer, screen, world_size, font):
    """
    Logs the state of each drone in the environment to a CSV file.
    """
    drone_positions = []
    for i, agent in enumerate(env.agents):
        try:
            # Get position (x, y) from the custom environment's agent_positions
            x, y = env.agent_positions[i]
            drone_positions.append((x, y))  # Store position for pygame rendering

            # Check if velocity (p_vel) is None
            if agent.state.p_vel is not None:
                vx, vy = agent.state.p_vel  # Get the x, y velocity from p_vel
            else:
                vx, vy = None, None  # Handle missing velocity
            
            accel = agent.accel  # Acceleration is stored in 'accel'
            orientation = agent.state.p_orientation  # Get orientation from p_orientation

            # Write the data to CSV, handling None values appropriately
            csv_writer.writerow([i, x, y, vx, vy, accel, orientation])

        except AttributeError as e:
            print(f"Error logging state for agent {i}: {e}")
    
    # Update pygame window with drone positions
    draw_drones(screen, drone_positions, world_size, font)
                

def train(num_timesteps, log_dir, csv_path):
    import deep_rl_for_swarms.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(format_strs=['csv'], dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                   hid_size=[64], feat_size=[64])
    """
    env = rendezvous.RendezvousEnv(nr_agents=20,
                                   obs_mode='sum_obs_acc',
                                   comm_radius=100 * np.sqrt(2),
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle_acc')
    """
    
    # Step 3: Use the custom environment
    env = CustomRendezvousEnv(nr_agents=20,
                              obs_mode='sum_obs_acc',
                              comm_radius=100 * np.sqrt(2),
                              world_size=100,
                              distance_bins=8,
                              bearing_bins=8,
                              torus=False,
                              dynamics='unicycle_acc')
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Drone Swarm Visualization")
    
    # Initialize font for labeling drones
    font = pygame.font.SysFont(None, 16)
    
    # Step 1: Open the CSV file before training
    file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['DroneID', 'X', 'Y', 'Velocity_X', 'Velocity_Y', 'Acceleration', 'Orientation'])
    
    """
    # Open CSV file for logging drone states
    with open(csv_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write header for CSV
        csv_writer.writerow(['DroneID', 'X', 'Y', 'Velocity_X', 'Velocity_Y', 'Acceleration', 'Orientation'])
    """
    
    # Modified TRPO learning loop to log states at each step
    def callback(locals, globals):
        log_drone_states(env, csv_writer, screen, env.world_size, font)
        return True  # Keep the training going

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=10, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, callback=callback)
        
    # Step 4: Close the CSV file after training
    file.close()
    env.close()

    # Quit pygame after training
    pygame.quit()
    
def main():
    env_id = "Rendezvous-v0"
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = '/tmp/baselines/trpo_test/rendezvous/' + dstr
    csv_path = 'testlog.csv'
    #train(num_timesteps=1e7, log_dir=log_dir, csv_path=csv_path)
    train(num_timesteps=10, log_dir=log_dir, csv_path=csv_path)


if __name__ == '__main__':
    main()