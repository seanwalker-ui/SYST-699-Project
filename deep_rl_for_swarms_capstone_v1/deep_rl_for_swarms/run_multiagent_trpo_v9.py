#!/usr/bin/env python3 - This is a line that tells the operating system to run this script using Python 3.

import datetime  # Used to manipulate date and time, particularly to create a timestamp for logging purposes.
from mpi4py import \
    MPI  # A Python wrapper for MPI (Message Passing Interface), which allows parallel processing and inter-process communication.
# This is used to distribute the computations across multiple processors for efficiency.

from deep_rl_for_swarms.common import logger, \
    cmd_util  # These are modules from the deep_rl_for_swarms package, which handle logging and command utility functions,
# likely related to the setup and debugging of the training environment.

from deep_rl_for_swarms.policies import \
    mlp_mean_embedding_policy  # This is the policy function (Multilayer Perceptron, MLP) used by agents in reinforcement
# learning to take actions based on observations.

from deep_rl_for_swarms.rl_algo.trpo_mpi import \
    trpo_mpi  # The Trust Region Policy Optimization (TRPO) algorithm implementation that works with MPI
# for distributed computation.

from ma_envs.envs.point_envs import \
    rendezvous  # This is an environment from deep_rl_for_swarms.ma_envs where agents (drones in this case) interact.
# It simulates drones or agents interacting in a "rendezvous" task, which is a swarm behavior where agents
# meet or align.
# from deep_rl_for_swarms.ma_envs.envs.point_envs import pursuit_evasion

# ------------------------------------------------------------------------------- #
# --------------- MK code - getting state, action, etc data --------------------- #
# ------------------------------------------------------------------------------- #
import json  # Used to handle data in JSON format, making it easy to save logs and state information.
import numpy as np  # A package for numerical computing in Python. It is used to work with arrays and matrices efficiently.
import gym  # A toolkit for developing and comparing reinforcement learning algorithms. This class wraps the environment to add extra functionality like logging.
import capstone_parameter_file as cpf

import pygame
import math
import csv
import random
import threading  # Run training and visualizer in parallel threads to ensure both work simultaneously without blocking each other.
import time  # Additional imports for pausing and resuming functionality
import uuid  # Generate unique run_id

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from collections import deque

# This is a class to handle pausing, resuming, and restarting of the training and display.
class TrainingController:
    def __init__(self, log_file="C:\\Users\\sawal\\Documents\\VS_Code\\Results\\DRL_training_controller_log.csv"):
        self.pause_event = threading.Event()  # Event for pausing
        self.pause_event.set()  # Initially sets the running state to True
        self.stop_event = threading.Event()  # Event to stop the training
        self.restart_event = threading.Event()  # Event to restart the training
        self.log_file = log_file
        self.init_log_file()

    # Initialize log file with headers
    def init_log_file(self):
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Event"])

    # Log actions to the CSV file
    def log_action(self, action):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, action])

    def pause(self):
        # Pauses the training and display.
        print("Pausing training and display...")
        self.pause_event.clear()  # Clear event to pause

    def resume(self):
        # Resumes the training and display.
        print("Resuming training and display...")
        self.pause_event.set()  # Set event to resume

    def restart(self):
        # Restarts the training and display.
        print("Restarting training and display...")
        self.restart_event.set()  # Set event to restart

    def stop(self):
        # Stops the training and display.
        print("Stopping training and display...")
        self.stop_event.set()  # Set event to stop both


class RunLogger:
    def __init__(self, log_file="C:\\Users\\sawal\\Documents\\VS_Code\\Results\\DRL_run_log.csv"):
        self.log_file = log_file
        self.run_id = str(uuid.uuid4())  # Unique identifier for each DRL run
        self.init_log_file()

    # Initialize the log file with headers
    def init_log_file(self):
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["run_id", "episode_id", "drone_id", "x_coord", "y_coord",
                             "orientation_unscaled", "linear_velocity", "angular_velocity",
                             "avoidance_action", "obstacle_type", "obstacle_id", "timestamp"])

    # Log data for each drone at each step
    def log_step(self, episode_id, drone_id, x_coord, y_coord, orientation_unscaled,
                 linear_velocity, angular_velocity, avoidance_action=None, obstacle_type=None,
                 obstacle_id=None, trajectory_changed=False):

        timestamp = time.strftime("%M:%S")  # Current timestamp
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.run_id, episode_id, drone_id, x_coord, y_coord,
                             orientation_unscaled, linear_velocity, angular_velocity,
                             avoidance_action, obstacle_type, obstacle_id, trajectory_changed,
                             timestamp])

    # New method to log collision avoidance details with trajectory change info
    def log_collision_avoidance(self, episode_id, drone_id, x_coord, y_coord, orientation_unscaled,
                                linear_velocity, angular_velocity, obstacle_type, obstacle_id, trajectory_changed):
        self.log_step(
            episode_id=episode_id,
            drone_id=drone_id,
            x_coord=x_coord,
            y_coord=y_coord,
            orientation_unscaled=orientation_unscaled,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            avoidance_action="Corrected Trajectory" if trajectory_changed else "No Change",
            obstacle_type=obstacle_type,
            obstacle_id=obstacle_id,
            trajectory_changed=trajectory_changed
        )

# This class contains the drone features for avoiding collisions.
class CollisionAvoidance:
    def __init__(self, min_distance=5, log_file="C:\\Users\\sawal\\Documents\\VS_Code\\Results\\DRL_collision_log.csv",
                 run_logger=None):
        self.min_distance = min_distance  # Minimum safe distance from obstacles and no-fly zones
        self.log_file = log_file
        self.init_log_file()
        self.run_logger = run_logger  # Reference to RunLogger for logging avoidance actions

    # This method initializes the log file with headers
    def init_log_file(self):
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Drone ID", "Object Type", "Distance", "Correction X", "Correction Y", "Collision Log"])

    # This computes the Euclidean distance between the two points.
    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def avoid_collisions(self, drones, obstacles, no_fly_zones, swarm_center, episode_id):
        for i, drone in enumerate(drones):
            drone_x, drone_y = drone[:2]  # Extract drone's x, y coordinates

            # Adjust trajectory based on obstacle or no-fly zone proximity
            corrected_trajectory, obstacle_type, obstacle_id = self.correct_trajectory(drone, obstacles, no_fly_zones, swarm_center, i)

            trajectory_changed = corrected_trajectory is not None
            # If collision avoidance changed trajectory, update drone's position
            if corrected_trajectory:
                # drone[2:] = corrected_trajectory  # Update the drone's 5D trajectory tuple
                drone[2:5] = corrected_trajectory[:3]  # Only update the relevant dimensions (e.g., 3D)

                # Log the avoidance action
                if self.run_logger:
                    self.run_logger.log_step(
                        episode_id=episode_id,
                        drone_id=i,
                        x_coord=drone_x,
                        y_coord=drone_y,
                        orientation_unscaled=drone[2],  # Assuming orientation is part of the trajectory
                        linear_velocity=drone[3],
                        angular_velocity=drone[4],
                        avoidance_action="Corrected Trajectory",
                        obstacle_type=obstacle_type,
                        obstacle_id=obstacle_id,
                        trajectory_changed=trajectory_changed
                    )

    # Adjust the drone's trajectory if it's near an obstacle or no-fly zone. Pass a 5D tuple representing the trajectory.
    def correct_trajectory(self, drone, obstacles, no_fly_zones, swarm_center, drone_id):
        drone_x, drone_y = drone[:2]
        trajectory_changed = False
        #obstacle_type = None

        for obstacle in obstacles:
            obstacle_x, obstacle_y = obstacle["position"]
            obstacle_id = obstacle.get("id", "Unknown")
            distance = self.calculate_distance(drone_x, drone_y, obstacle_x, obstacle_y)
            if distance < self.min_distance + 20:  # 20 pixel buffer
                trajectory_changed = True
                obstacle_type = obstacle["type"]
                # Adjust drone trajectory to avoid the obstacle
                new_trajectory = self.calculate_new_trajectory(drone, obstacle_x, obstacle_y, distance, swarm_center)

                # self.log_collision(drone_id, "Obstacle", distance, new_trajectory)
                #self.log_collision(drone_id, obstacle_type, distance, new_trajectory)
                return new_trajectory, obstacle_type, obstacle_id
                #return new_trajectory, obstacle["type"], obstacle_id

        # Handle no-fly zones similarly to avoiding obstacles
        for no_fly_zone in no_fly_zones:
            zone_x, zone_y = no_fly_zone["position"]
            zone_size = no_fly_zone["size"]
            if zone_x <= drone_x <= zone_x + zone_size and zone_y <= drone_y <= zone_y + zone_size:
                trajectory_changed = True
                obstacle_type = "No Fly Zone"
                # Avoid no-fly zone
                new_trajectory = self.calculate_new_trajectory(drone, zone_x + zone_size / 2, zone_y + zone_size / 2,
                                                               self.min_distance, swarm_center)

                # self.log_collision(drone_id, "No Fly Zone", self.min_distance, new_trajectory)
                #self.log_collision(drone_id, obstacle_type, self.min_distance, new_trajectory)
                return new_trajectory, obstacle_type
                #return new_trajectory, "No Fly Zone", no_fly_zone["id"]

        # If no adjustment needed, return None
        return None, None, None
        # return None if not trajectory_changed else drone[2:]

    # ----------------------------------------------------------------------------------------------------------------------- #
    def calculate_new_trajectory(self, drone, obj_x, obj_y, distance, swarm_center):
        # Calculate a new trajectory based on obstacle and swarm center.
        overlap = (self.min_distance + 5) - distance
        dx = drone[0] - obj_x
        dy = drone[1] - obj_y
        angle = math.atan2(dy, dx)

        # Calculate a trajectory that moves away from the obstacle
        correction_x = math.cos(angle) * overlap
        correction_y = math.sin(angle) * overlap

        # Align slightly with the swarm center to maintain group behavior
        swarm_dx = swarm_center[0] - drone[0]
        swarm_dy = swarm_center[1] - drone[1]
        swarm_angle = math.atan2(swarm_dy, swarm_dx)

        # Combine both avoidance and swarm movement directions
        adjusted_correction_x = correction_x + 0.1 * math.cos(swarm_angle)
        adjusted_correction_y = correction_y + 0.1 * math.sin(swarm_angle)

        # Return a new 5D tuple for the drone's trajectory (x, y, and potentially other dimensions like velocity)
        new_trajectory = (adjusted_correction_x, adjusted_correction_y, drone[2], drone[3], drone[4])
        return new_trajectory

    def log_collision(self, drone_id, obj_type, distance, new_trajectory):
        collision_log = f"Drone {drone_id} avoided a collision with {obj_type}."

        # Log the collision avoidance event
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # writer.writerow([drone_id, obj_type, distance] + list(new_trajectory))
            writer.writerow([drone_id, obj_type, distance, new_trajectory[0], new_trajectory[1], collision_log])


# The DroneVisualizer class will manage the pygame window and display the drones' positions during each step.
class DroneVisualizer:
    def __init__(self, window_width=1280, window_height=960, drone_radius=5, num_drones=0,
                 num_no_fly_zones=0, num_humans=0, num_buildings=0, num_trees=0, num_animals=0):
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.window_width = window_width
        self.window_height = window_height
        self.drone_radius = drone_radius
        self.drone_positions = deque(maxlen=20)  # Store recent positions for path tracking

        # Initialize obstacles and no-fly zones
        self.num_no_fly_zones = num_no_fly_zones
        self.no_fly_zones = self.generate_no_fly_zones(num_no_fly_zones)
        self.humans, self.buildings, self.trees, self.animals = [], [], [], []
        self.generate_obstacles(num_humans, num_buildings, num_trees, num_animals)

        # Generate random target location
        self.target_location = self.generate_random_target()
        self.target_radius = 10

        # Setup Dash layout
        self.app.layout = html.Div([
            html.H1("Drone Swarm Visualization"),
            dcc.Graph(id='drone-visualizer'),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)  # Update every second
        ])

        # Define callback to update the graph with latest drone, obstacle, and no-fly zone data
        @self.app.callback(
            Output('drone-visualizer', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph(n_intervals):
            return self.create_figure()

    def generate_no_fly_zones(self, num_no_fly_zones):
        return [{"id": i, "position": (random.randint(0, 100), random.randint(0, 100)), "size": random.randint(5, 15)}
                for i in range(num_no_fly_zones)]

    def generate_random_target(self):
        return random.randint(0, 100), random.randint(0, 100)

    def generate_obstacles(self, num_humans, num_buildings, num_trees, num_animals):
        self.humans = [{"type": "Human", "position": (random.randint(0, 100), random.randint(0, 100))}
                       for _ in range(num_humans)]
        self.buildings = [{"type": "Building", "position": (random.randint(0, 100), random.randint(0, 100))}
                          for _ in range(num_buildings)]
        self.trees = [{"type": "Tree", "position": (random.randint(0, 100), random.randint(0, 100))}
                      for _ in range(num_trees)]
        self.animals = [{"type": "Animal", "position": (random.randint(0, 100), random.randint(0, 100))}
                        for _ in range(num_animals)]

    def create_figure(self):
        # Define Plotly traces for drones, obstacles, and no-fly zones
        drone_trace = go.Scatter(
            x=[pos[0] for pos in self.drone_positions],
            y=[pos[1] for pos in self.drone_positions],
            mode='markers+lines',
            name='Drones',
            marker=dict(size=self.drone_radius, color='blue')
        )

        no_fly_trace = go.Scatter(
            x=[zone['position'][0] for zone in self.no_fly_zones],
            y=[zone['position'][1] for zone in self.no_fly_zones],
            mode='markers',
            name='No-Fly Zones',
            marker=dict(size=15, color='yellow', opacity=0.5)
        )

        target_trace = go.Scatter(
            x=[self.target_location[0]],
            y=[self.target_location[1]],
            mode='markers',
            name='Target Location',
            marker=dict(size=self.target_radius, color='red')
        )

        obstacle_trace = go.Scatter(
            x=[obs['position'][0] for obs in self.get_all_obstacles()],
            y=[obs['position'][1] for obs in self.get_all_obstacles()],
            mode='markers',
            name='Obstacles',
            marker=dict(size=10, color='green')
        )

        layout = go.Layout(
            title="Drone Swarm Path with Obstacles and No-Fly Zones",
            xaxis=dict(range=[0, 100], title="X Coordinate"),
            yaxis=dict(range=[0, 100], title="Y Coordinate"),
            showlegend=True
        )

        return {"data": [drone_trace, no_fly_trace, target_trace, obstacle_trace], "layout": layout}

    def update(self, drone_positions):
        # Append latest drone positions to maintain a path
        self.drone_positions.extend(drone_positions)

    def get_all_obstacles(self):
        return self.humans + self.buildings + self.trees + self.animals

    def run(self):
        self.app.run_server(debug=True, use_reloader=False)


# Add custom logging and tracking of the environment's state, actions, and rewards.
# It logs these interactions at each step during training and writes them into a JSON file.
# In addition, it includes a pause function.
#

#max_episodes = 1030
class CustomMonitor(gym.Wrapper):
    # Initializes internal lists for rewards, episode length, and the log data
    def __init__(self, env, log_file='gym_log_data.json', initial_state_file='initial_state.json',
                 training_controller=None, collision_avoidance=None, num_drones=0,
                 num_no_fly_zones=0, num_humans=0, num_buildings=0, num_trees=0, num_animals=0, ):
        # super(CustomMonitor, self).__init__(env): Calls the parent class (gym.Wrapper) constructor to set up the environment.
        super(CustomMonitor, self).__init__(env)
        self.log_file = log_file
        self.initial_state_file = initial_state_file
        self.episode_rewards = []
        self.episode_length = 0
        self.log_data = []  # List to store all log entries
        self.initial_state_data = []  # List to store initial state information
        self.episode_id = 0  # Track episode ID

        # Initialize RunLogger for detailed step logging
        self.run_logger = RunLogger()
        self.training_controller = training_controller
        self.collision_avoidance = collision_avoidance

        # Initialize the visualizer
        self.visualizer = DroneVisualizer(num_drones=num_drones, num_no_fly_zones=num_no_fly_zones,
                                          num_humans=num_humans, num_buildings=num_buildings, num_trees=num_trees,
                                          num_animals=num_animals)
        self.visualizer.run()

        # Store the number of no-fly zones and their properties for logging
        self.no_fly_zones = self.visualizer.no_fly_zones
        # Generate obstacles after initializing the visualizer
        self.visualizer.generate_obstacles(num_humans, num_buildings, num_trees, num_animals)

    # reset(): Resets the environment to start a new episode. It also logs the initial state, agent states, and other matrices (like distance_matrix and angle_matrix).
    # If these attributes are not available, it samples from the observation space.
    def reset(self, **kwargs):
        # Resets the environment and initializes logging for a new episode.
        self.episode_rewards = []
        self.episode_length = 0
        self.episode_id += 1  # Increment episode_id for each new episode

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

    # Computes a reward based on how close drones are to the target and each other. Penalize drones that are far from the target by using the average distance.
    def calculate_swarm_reward(self, drone_positions, target_location):
        # Calculates reward based on how close drones are to the target and how close they are to each other.
        target_x, target_y = target_location

        # Calculate average distance of drones to the target
        total_distance_to_target = 0
        for pos in drone_positions:
            drone_x, drone_y = pos[:2]
            distance_to_target = np.sqrt((drone_x - target_x) ** 2 + (drone_y - target_y) ** 2)
            total_distance_to_target += distance_to_target

        average_distance_to_target = total_distance_to_target / len(drone_positions)

        # Reward is inversely proportional to the average distance to the target
        # Reward will be higher when drones are closer to the target
        reward = -average_distance_to_target

        return reward

    # step(action): Executes a single step in the environment using the provided action.
    # It logs the current state, next state, reward, and other environment-specific information like distance and angle matrices.
    # This information is appended to log_data, which tracks the whole episode.
    def step(self, action):
        # Check if the controller has paused
        while not self.training_controller.pause_event.is_set():
            time.sleep(0.1)  # Sleep for a bit until the visualizer is resumed

        # Logs state, action, reward, next state, and done.
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None

        # Get drone positions (this assumes your environment has a get_drone_positions method)
        drone_positions = self.env.get_drone_positions()

        # Log each drone's data
        for i, drone in enumerate(drone_positions):
            x, y = drone[0], drone[1]
            orientation_unscaled = drone[2]  # Assume this is the orientation
            linear_velocity = drone[3]
            angular_velocity = drone[4]

            # Log the data for each drone
            self.run_logger.log_step(self.episode_id, i, x, y, orientation_unscaled, linear_velocity, angular_velocity)

        # Get the reward based on distance to target and swarm behavior
        target_reward = self.calculate_swarm_reward(drone_positions, self.visualizer.target_location)

        # **Collision Avoidance Logic**:
        # all_obstacles = self.visualizer.trees + self.visualizer.buildings + self.visualizer.humans + self.visualizer.animals
        all_obstacles = self.visualizer.get_all_obstacles()
        no_fly_zones = self.visualizer.no_fly_zones

        # Pass swarm center (for swarm behavior)
        swarm_center = self.calculate_swarm_center(drone_positions)

        # Apply collision avoidance to adjust the drone positions
        # self.collision_avoidance.avoid_collisions(drone_positions, obstacles, no_fly_zones, swarm_center)
        # ------------------------------------------------------------------------------------------------------------------------ #
        self.collision_avoidance.avoid_collisions(drone_positions, all_obstacles, no_fly_zones, swarm_center, self.episode_id)

        # Update the visualizer with the current drone positions after collision avoidance
        self.visualizer.update(drone_positions)

        # Add the swarm reward to the normal environment reward
        total_reward = reward + target_reward  # Rewards are negative

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
            "drone_positions": drone_positions,
            "movement_reward": total_reward,
            "no_fly-zones": self.no_fly_zones,  # Logs the size and positions of the no_fly_zones
            "obstacle_positions": {
                "humans": self.visualizer.humans,
                "buildings": self.visualizer.buildings,
                "trees": self.visualizer.trees,
                "animals": self.visualizer.animals
            }
        }

        # Append log entry to the list
        self.log_data.append(log_entry)

        if done:
            self.episode_length += 1  # Increments the episode step count.
            self.episode_rewards.append(total_reward)
            self.episode_id += 1  # Increment episode counter
            #if self.episode_id >= max_episodes:
                #print(f"Reached {max_episodes} episodes, terminating.")
                #self.env.close()  # Ensures proper termination
                #return None, None, True, {}  # Signal done
        # self.episode_rewards.append(reward)  # Logs the reward obtained in this step.
        #self.episode_rewards.append(total_reward)

        # return next_state, reward, done, info
        return next_state, total_reward, done, info

    def calculate_swarm_center(self, drone_positions):
        # Calculate the centroid of the swarm (average of all drone positions)
        x_coords = [pos[0] for pos in drone_positions]
        y_coords = [pos[1] for pos in drone_positions]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)

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

    def toggle_pause(self):
        self.paused = not self.paused  # Flip the pause state

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
def train(num_timesteps, log_dir, training_controller, collision_avoidance, num_drones=0,
          num_no_fly_zones=0, num_humans=0, num_buildings=0, num_trees=0, num_animals=0, ):
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
    env = rendezvous.RendezvousEnv(  # nr_agents=20,
        nr_agents=num_drones,
        obs_mode='sum_obs_acc',
        comm_radius=100 * np.sqrt(2),
        world_size=100,
        distance_bins=8,
        bearing_bins=8,
        torus=False,
        dynamics='unicycle_acc')

    # Collision avoidance system
    collision_avoidance = CollisionAvoidance()

    # ------------------------------------------------------------------------------- #
    # ----------------------- MK code - adding custom logger ------------------------ #
    # ------------------------------------------------------------------------------- #

    # Initializes the CustomMonitor to wrap the environment for logging purposes.
    # The log files (gym_log_data.json and initial_state.json) are created in the logging directory
    custom_logger = CustomMonitor(env, log_file=logger.get_dir() + '/gym_log_data.json',
                                  initial_state_file=logger.get_dir() + '/initial_state.json',
                                  training_controller=training_controller, collision_avoidance=collision_avoidance,
                                  num_drones=num_drones, num_no_fly_zones=num_no_fly_zones, num_humans=num_humans,
                                  num_buildings=num_buildings, num_trees=num_trees, num_animals=num_animals)
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
    # Find a way to change the timesteps: default is 130 (self_termination)
    trpo_mpi.learn(custom_logger, policy_fn, timesteps_per_batch=10, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    # Closes the logger and the environment after training is complete, ensuring that all data is saved.
    custom_logger.close()
    env.close()


# This is the main entry point of the script. It generates a unique logging directory based on the current time and starts the training process by calling train().
def main():
    # Initialize training controller
    training_controller = TrainingController()

    # Function to listen for user input and control training state
    def listen_for_input():
        while True:
            command = input("Enter 'p' to pause, 'r' to resume, 's' to stop, or 're' to restart: ").strip().lower()
            if command == 'p':
                training_controller.pause()
            elif command == 'r':
                training_controller.resume()
            elif command == 're':
                training_controller.restart()
            elif command == 's':
                training_controller.stop()
                break

                # Get user input for how many drones, obstacles and no_fly_zones to include.

    try:
        num_drones = int(input("Enter the number of drones to run in the rendezvous mission: "))
        num_no_fly_zones = int(input("Enter the number of no-fly zones to generate: "))
        num_humans = int(input("Enter the number of humans to generate: "))
        num_buildings = int(input("Enter the number of buildings to generate: "))
        num_trees = int(input("Enter the number of trees to generate: "))
        num_animals = int(input("Enter the number of animals to generate: "))
    except ValueError:
        print("Invalid input. Please enter integers for the number of drones, obstacles and no fly zones.")
        return

    # Start listening for input in a separate thread
    input_thread = threading.Thread(target=listen_for_input)
    input_thread.start()

    num_timesteps = 10
    env_id = "Rendezvous-v0"
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = cpf.USER_OUTPUT_PATH + env_id + '_' + dstr

    #  Start the training process with the user-specified number of drones
    # train(num_timesteps=10, log_dir=log_dir, num_drones=num_drones)
    # train(num_timesteps=10, log_dir=log_dir, num_drones=num_drones, num_no_fly_zones=num_no_fly_zones, num_humans=num_humans, num_buildings=num_buildings, num_trees=num_trees, num_animals=num_animals)

    # Create a separate thread for the training process - this allows the visualizer to display
    # pygame requires a continuous event loop to render and update the window. If the event loop is blocked or not running, the window may not display or refresh correctly.
    # TRPO algorithm might block the pygame window from refreshing properly
    training_thread = threading.Thread(target=train, args=(
    num_timesteps, log_dir, training_controller, num_drones, num_no_fly_zones, num_humans, num_buildings, num_trees,
    num_animals))
    training_thread.start()

    # Wait for both threads to complete
    training_thread.join()
    input_thread.join()

    # Keep the Pygame visualizer running in the main thread to prevent it from freezing
    # while training_thread.is_alive():
    # pygame.time.wait(100)  # Adds a small delay to avoid busy-waiting
    # pygame.event.pump()    # This allows pygame to process its events like window close


if __name__ == '__main__':
    main()