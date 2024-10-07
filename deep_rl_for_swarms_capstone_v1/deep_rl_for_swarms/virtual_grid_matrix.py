import numpy as np
import random
import tensorflow as tf
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from ma_envs.envs.point_envs.rendezvous import RendezvousEnv
from deep_rl_for_swarms.policies.mlp_policy import MlpPolicy  # Import the MLP policy
from deep_rl_for_swarms.common import logger, cmd_util
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi


# This is a class for a 100x100 sized grid matrix.
class GridEnvironment:
    def __init__(self, size=100, obstacle_probability=0.1, num_drones=20):
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
        # Initialize all of the drone positions. Note: Double-check if drones are all initialized in the rendezvous.py file.
        self.drone_positions = np.array([[random.randint(0, self.size-1), random.randint(0, self.size-1)] for _ in range(self.num_drones)])
        
        # Initialize the Rendezvous Environment
        self.env = RendezvousEnv(nr_agents=self.num_drones, obs_mode='sum_obs_acc', comm_radius=100 * np.sqrt(2),
                                 world_size=self.size, distance_bins=8, bearing_bins=8, torus=False, dynamics='unicycle_acc')
        
        # Initialize the policy
        self.policy = MlpPolicy("pi", self.env.observation_space, self.env.action_space, hid_size=[64])

    
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
    
    def move_drones(self, policy_fn, env_state):
        """
        Moves the drones using the learned policy from rendezvous.py.
        """
        for i, drone_pos in enumerate(self.drone_positions):
            # Use the policy to get the next action (movement) for the drone
            action = policy_fn(env_state)  # Get action using the policy
            
            # Translate the action into grid movements (assume action gives direction)
            move = self.convert_action_to_movement(action)

            # Compute new position based on the action
            new_position = drone_pos + np.array(move)

            # Ensure new position is within bounds and not on an obstacle
            if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size and self.grid[new_position[0], new_position[1]] == self.EMPTY:
                self.drone_positions[i] = new_position
        
               
    def print_grid(self):
        """
        Prints the grid for visualization. Optional for debugging purposes.
        """
        for row in self.grid:
            print(' '.join(map(str, row)))

    def get_grid_with_drones(self):
        """
        Returns a copy of the grid with drone positions marked in it.
        """
        grid_with_drones = np.copy(self.grid)
        for drone_pos in self.drone_positions:
            grid_with_drones[drone_pos[0], drone_pos[1]] = self.DRONE
        return grid_with_drones

    
    def visualize_grid(self):
        """
        Visualizes the grid environment using matplotlib.
        Different colors represent different obstacles.
        """
        # Define a custom color map for different obstacles
        cmap = ListedColormap(['white', 'green', 'red', 'blue', 'yellow'])

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        grid_with_drones = self.get_grid_with_drones()
        mat = ax.matshow(grid_with_drones, cmap=cmap)
        
        """
        # Update function for animation
        def update(frame):
            self.move_drones()  # Move drones at each frame
            grid_with_drones = self.get_grid_with_drones()
            mat.set_data(grid_with_drones)  # Update grid with new drone positions
            return [mat]
        """
        def update(frame):
            # Assuming `self.policy` is the policy function and `self.env` is the environment state.
            self.move_drones(self.policy.act, self.env.state)  # Pass the policy function and environment state
            grid_with_drones = self.get_grid_with_drones()
            mat.set_data(grid_with_drones)  # Update grid with new drone positions
            return [mat]
        
        # Animate the grid using FuncAnimation
        animation = FuncAnimation(fig, update, frames=200, interval=200, blit=True)

         # Create a color bar for the grid visualization
        cbar = plt.colorbar(mat)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['Empty', 'Tree', 'Person', 'Animal', 'Hazard', 'Drone'])

        plt.title("Real-Time Drone Swarm Movement")
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create an instance of GridEnvironment
    environment = GridEnvironment(size=100, obstacle_probability=0.1, num_drones=20)
    
    # Generate the grid environment with obstacles
    environment.generate_environment()
    
    # Print the grid to visualize what is happening.
    environment.visualize_grid()

    # Access the grid matrix for further use
    grid_matrix = environment.get_grid_with_drones()