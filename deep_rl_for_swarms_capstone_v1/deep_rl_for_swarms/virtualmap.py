import numpy as np
import random
import csv
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from ma_envs.envs.point_envs.rendezvous2 import RendezvousEnv
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.policies.mlp_policy import MlpPolicy  # Import the MLP policy
from deep_rl_for_swarms.common import logger

# FuncAnimation may have issues if  using a non-interactive backend like Agg. I switched to a more suitable backend like TkAgg.
matplotlib.use('TkAgg') 

class GridEnvironment:
    def __init__(self, size=100, obstacle_probability=0.1, num_drones=20, policy=None, env=None):
        """
        Initialize the grid environment and rendezvous environment.
        """
        self.size = size
        self.obstacle_probability = obstacle_probability
        self.num_drones = num_drones

        # Define obstacle types in the environment that the drone swarm needs to avoid
        self.EMPTY = 0
        self.TREE = 1
        self.PERSON = 2
        self.ANIMAL = 3
        self.HAZARD = 4
        self.DRONE = 5

        # Initialize grid
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Initialize drone positions randomly within the grid
        #self.drone_positions = np.array([[random.randint(0, self.size-1), random.randint(0, self.size-1)] for _ in range(self.num_drones)])
        
        # Calculate the center of the grid
        center_x, center_y = self.size // 2, self.size // 2

        # Group the drones around the center of the grid
        self.drone_positions = []
        offset = [-1, 0, 1]  # Offset to create a group around the center
        for i in range(self.num_drones):
            x_offset = random.choice(offset)
            y_offset = random.choice(offset)
            drone_position = [center_x + x_offset, center_y + y_offset]
            self.drone_positions.append(drone_position)

        self.drone_positions = np.array(self.drone_positions)
        
                
        # Initialize the Rendezvous Environment (passed from `run_multiagent_trpo.py`)
        self.env = env  # Assuming the environment is passed here during initialization
        self.policy = policy  # The TRPO policy that will control drone movements
        # Animation
        self.animation = None
        
        # Open New CSV file and set up for logging
        self.csv_file = open('C:\\Users\\sawal\\Documents\\VS_Code\\Output\\vmap_drone_log.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Frame', 'Drone ID', 'X Position', 'Y Position'])  # Header for CSV

    def place_obstacles(self, obstacle_type, probability):
        """
        Places obstacles of a certain type randomly in the grid with a given probability.
        """
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < probability:
                    self.grid[i, j] = obstacle_type
    
    def generate_environment(self):
        """
        Generates the grid environment by placing all of the different types of obstacles.
        """
        self.place_obstacles(self.TREE, self.obstacle_probability)
        self.place_obstacles(self.PERSON, self.obstacle_probability / 2)
        self.place_obstacles(self.ANIMAL, self.obstacle_probability / 2)
        self.place_obstacles(self.HAZARD, self.obstacle_probability / 3)
    
    def move_drones(self):
        """
        Moves the drones using the learned policy from the TRPO training process.
        Update the drone positions on the grid.
        """
        # Get the current observation (environment state)
        observation = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        
        for i in range(self.num_drones):
            # Get action for each drone using the policy (MLPPolicy.act)
            action, _ = self.policy.act(stochastic=True, ob=observation)
            
            # Convert action to a movement (This line of code is assuming that a helper function is used to interpret the action)
            move = self.convert_action_to_movement(action)
            
            # Compute/Solve the new position based on the action
            new_position = self.drone_positions[i] + np.array(move)
            
            # Ensure the new position is within bounds and not colliding with obstacles
            if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size and self.grid[new_position[0], new_position[1]] == self.EMPTY:
                self.drone_positions[i] = new_position
    
    def convert_action_to_movement(self, action):
        """
        Converts the action output from the policy into grid movements.
        This is a placeholder function - map continuous actions to discrete grid movements.
        """
        # Placeholder logic: Convert action (continuous) into discrete grid moves
        # Example in Simple Terms: interpret action as (dx, dy) displacement on the grid
        dx = np.clip(int(action[0]), -1, 1)
        dy = np.clip(int(action[1]), -1, 1)
        return np.array([dx, dy])
    
    def get_grid_with_drones(self):
        """
        Returns a copy of the grid with drone positions marked in it.
        """
        grid_with_drones = np.copy(self.grid)
        for drone_pos in self.drone_positions:
            grid_with_drones[drone_pos[0], drone_pos[1]] = self.DRONE
        return grid_with_drones
    
    def log_drone_positions(self, frame):
        """
        Log the current positions of all drones to the CSV file for the given frame.
        """
        for i, drone_pos in enumerate(self.drone_positions):
            # Log drone ID, X position, and Y position in the CSV file
            self.csv_writer.writerow([frame, i, drone_pos[0], drone_pos[1]])
    
    def visualize_grid(self):
        """
        Visualizes the grid environment using matplotlib.
        Different colors represent different obstacles and the drones.
        """
        cmap = ListedColormap(['white', 'green', 'red', 'blue', 'yellow', 'black'])

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        grid_with_drones = self.get_grid_with_drones()
        mat = ax.matshow(grid_with_drones, cmap=cmap)
        
        # Initialize function to set up background for animation
        def init():
            mat.set_data(np.zeros((self.size, self.size)))  # Set an empty grid
            return [mat]
        
        def update(frame):
            """
            Animation update function. Moves the drones and updates the grid visualization.
            """
            # Print the frame for now to use it
            print(f"Frame: {frame}")
            
            self.move_drones()  # Move drones based on the policy
            self.log_drone_positions(frame)  # Log drone positions for this frame
            grid_with_drones = self.get_grid_with_drones()
            mat.set_data(grid_with_drones)  # Update grid with new drone positions
            return [mat]
        
        # Animate the grid using FuncAnimation - Code is still not working. Needs continued revisions
        #self.animation = FuncAnimation(fig, update, frames=200, interval=200, blit=True)
        self.animation = FuncAnimation(fig, update, frames=200, init_func=init, interval=200, blit=False)
        
        # Create a color bar for the grid visualization
        cbar = plt.colorbar(mat)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['Empty', 'Tree', 'Person', 'Animal', 'Hazard', 'Drone'])

        plt.title("Real-Time Drone Swarm Movement")
        plt.show()
        
    def close(self):
        """
        Closes CSV file after logging is done.
        """
        self.csv_file.close()

# This helps me visualize the example of being used in run_multiagent_trpo.py
def main():
    # Initialize the TRPO environment and policy
    env = RendezvousEnv(nr_agents=20, obs_mode='sum_obs_acc', comm_radius=100 * np.sqrt(2),
                        world_size=100, distance_bins=8, bearing_bins=8, torus=False, dynamics='unicycle_acc')
    
    # Initialize the policy (MLP-based policy function)
    #policy = MlpPolicy("pi", env.observation_space, env.action_space, hid_size=[64])
    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                   hid_size=[64], feat_size=[64])
    
    # Create an instance of GridEnvironment
    grid_env = GridEnvironment(size=100, obstacle_probability=0.1, num_drones=20, policy=policy_fn, env=env)
    
    # Generate the grid environment with obstacles
    grid_env.generate_environment()
    
    # Visualize the drone movements
    grid_env.visualize_grid()

if __name__ == "__main__":
    main()