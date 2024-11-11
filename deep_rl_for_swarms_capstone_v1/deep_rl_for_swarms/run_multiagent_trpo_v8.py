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
import math
import csv
import random
import threading   # Run training and visualizer in parallel threads to ensure both work simultaneously without blocking each other.
import time        # Additional imports for pausing and resuming functionality
import uuid        # Generate unique run_id


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
                             "avoidance_action", "obstacle_type", "timestamp"])

    # Log data for each drone at each step
    def log_step(self, episode_id, drone_id, x_coord, y_coord, orientation_unscaled,
                 linear_velocity, angular_velocity, avoidance_action=None, obstacle_type=None):
        timestamp = time.strftime("%M:%S")  # Current timestamp
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.run_id, episode_id, drone_id, x_coord, y_coord,
                             orientation_unscaled, linear_velocity, angular_velocity,
                             avoidance_action, obstacle_type, timestamp])


# This class contains the drone features for avoiding collisions.
class CollisionAvoidance:
    def __init__(self, min_distance=5, log_file="C:\\Users\\sawal\\Documents\\VS_Code\\Results\\DRL_collision_log.csv", run_logger=None):
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

    def avoid_collisions(self, drones, obstacles, no_fly_zones, fires, swarm_center):
        for i, drone in enumerate(drones):
            drone_x, drone_y = drone[:2]  # Extract drone's x, y coordinates

            # Adjust trajectory based on obstacle or no-fly zone proximity
            corrected_trajectory, obstacle_type = self.correct_trajectory(drone, obstacles, no_fly_zones, fires, swarm_center, i)

            # If collision avoidance changed trajectory, update drone's position
            if corrected_trajectory:
                #drone[2:] = corrected_trajectory  # Update the drone's 5D trajectory tuple
                drone[2:5] = corrected_trajectory[:3]   # Only update the relevant dimensions (e.g., 3D)

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
                        obstacle_type=obstacle_type
                    )

    # Adjust the drone's trajectory if it's near an obstacle or no-fly zone. Pass a 5D tuple representing the trajectory.
    def correct_trajectory(self, drone, obstacles, no_fly_zones, fires, swarm_center, drone_id):
        drone_x, drone_y = drone[:2]
        trajectory_changed = False
        obstacle_type = None

        for obstacle in obstacles:
            #obstacle_x, obstacle_y = obstacle[:2]
            obstacle_x, obstacle_y = obstacle["position"]
            distance = self.calculate_distance(drone_x, drone_y, obstacle_x, obstacle_y)
            if distance < self.min_distance + 20:  # 20 pixel buffer
                trajectory_changed = True
                obstacle_type = obstacle["type"]
                # Adjust drone trajectory to avoid the obstacle
                new_trajectory = self.calculate_new_trajectory(drone, obstacle_x, obstacle_y, distance, swarm_center)

                #self.log_collision(drone_id, "Obstacle", distance, new_trajectory)
                self.log_collision(drone_id, obstacle_type, distance, new_trajectory)
                return new_trajectory, obstacle_type

        # Handle no-fly zones similarly to avoiding obstacles
        for no_fly_zone in no_fly_zones:
            zone_x, zone_y = no_fly_zone["position"]
            zone_size = no_fly_zone["size"]
            if zone_x <= drone_x <= zone_x + zone_size and zone_y <= drone_y <= zone_y + zone_size:
                trajectory_changed = True
                obstacle_type = "No Fly Zone"
                # Avoid no-fly zone
                new_trajectory = self.calculate_new_trajectory(drone, zone_x + zone_size / 2, zone_y + zone_size / 2, self.min_distance, swarm_center)

                #self.log_collision(drone_id, "No Fly Zone", self.min_distance, new_trajectory)
                self.log_collision(drone_id, obstacle_type, self.min_distance, new_trajectory)
                return new_trajectory, obstacle_type

        # Handle fires zones similarly to avoiding obstacles and no_fly_zones
        for fire in fires:
            zone_x, zone_y = fire["position"]
            zone_size = fire["size"]
            if zone_x <= drone_x <= zone_x + zone_size and zone_y <= drone_y <= zone_y + zone_size:
                trajectory_changed = True
                obstacle_type = "Fire Zone"
                # Avoid Fire zone
                new_trajectory = self.calculate_new_trajectory(drone, zone_x + zone_size / 2,
                                                               zone_y + zone_size / 2, self.min_distance,
                                                               swarm_center)

                # self.log_collision(drone_id, "No Fly Zone", self.min_distance, new_trajectory)
                self.log_collision(drone_id, obstacle_type, self.min_distance, new_trajectory)
                return new_trajectory, obstacle_type

        # If no adjustment needed, return None
        return None, None
        #return None if not trajectory_changed else drone[2:]

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
            #writer.writerow([drone_id, obj_type, distance] + list(new_trajectory))
            writer.writerow([drone_id, obj_type, distance, new_trajectory[0], new_trajectory[1], collision_log])

# The DroneVisualizer class will manage the pygame window and display the drones' positions during each step.
class DroneVisualizer:
    def __init__(self, window_width=1280, window_height=960, drone_radius=5, num_drones=0,
                 num_no_fly_zones=0, num_humans=0, num_buildings=0, num_trees=0, num_animals=0, num_fires=0,
                 log_file="C:\\Users\\sawal\\Documents\\VS_Code\\Results\\DRL_visualizer_log.csv"):
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
        self.log_file = log_file
        self.init_log_file()

        # Initialize obstacle lists and colors for the obstacles/drones.
        self.drone_color = (0, 0, 255)  # Blue for drones
        self.human_color = (255, 165, 0)  # Orange color for humans
        self.building_color = (128, 128, 128)  # Gray color for buildings
        self.tree_color = (0, 255, 0)  # Green color for trees
        self.animal_color = (139, 69, 19)  # Brown color for animals
        self.humans = []
        self.buildings = []
        self.trees = []
        self.animals = []
        
        # Initialize no-fly zones and fires.
        self.no_fly_zones = []
        self.num_no_fly_zones = num_no_fly_zones
        self.generate_no_fly_zones(self.num_no_fly_zones)
        self.no_fly_zone_color = (255, 255, 0)  # Yellow

        self.fires = []
        self.num_fires = num_fires
        self.generate_fires(self.num_fires)
        self.fire_color = (255, 0, 0)  # Red
        
        # Generate a random target location
        self.target_location = self.generate_random_target()
        self.target_radius = 10  # Radius of the target

    # Initializes the log file for the visualizer
    def init_log_file(self):
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Drone ID", "Position X", "Position Y", "Target X", "Target Y"])

    # Combine obstacles into a single list with "type" keys for each entry.
    def get_all_obstacles(self):
        # Combines all obstacle types into a single list
        all_obstacles = self.humans + self.buildings + self.trees + self.animals
        return all_obstacles

    def generate_no_fly_zones(self, num_no_fly_zones):
        # Generate random positions and sizes for no-fly zones
        for i in range(num_no_fly_zones):
            size = random.randint(50, 150)  # Random size between 50 and 150 pixels
            x = random.randint(0, self.window_width - size)
            y = random.randint(0, self.window_height - size)
            self.no_fly_zones.append({"id": i, "position": (x, y), "size": size})

    def draw_no_fly_zones(self):
        # Draw the no-fly zones as yellow rectangles
        for zone in self.no_fly_zones:
            x, y = zone['position']
            size = zone['size']
            pygame.draw.rect(self.screen, self.no_fly_zone_color, (x, y, size, size), width=2)
            
            # Label no fly zone with its ID
            label = self.font.render(f"No Fly Zone {zone['id']}", True, (0, 0, 0))  # Black label
            label_rect = label.get_rect(center=(x + size // 2, y + size // 2))  # Center the label inside the rectangle
            self.screen.blit(label, label_rect)

    # Generate random positions and sizes for fires.
    def generate_fires(self, num_fires):
        for _ in range(num_fires):
            size = random.randint(10, 30)  # Random radius size for fires
            x = random.randint(0, self.window_width)
            y = random.randint(0, self.window_height)
            self.fires.append({"type": "Fire", "position": (x, y), "size": size})

    # Draw all fires as dashed circles with labels.
    def draw_fires(self, dash_length=10, gap_length=5):
        for fire in self.fires:
            x, y = fire['position']
            radius = fire['size']
            circumference = 2 * math.pi * radius
            num_dashes = int(circumference / (dash_length + gap_length))

            for i in range(num_dashes):
                start_angle = i * (dash_length + gap_length) / radius
                end_angle = start_angle + dash_length / radius
                start_pos = (x + radius * math.cos(start_angle), y + radius * math.sin(start_angle))
                end_pos = (x + radius * math.cos(end_angle), y + radius * math.sin(end_angle))
                pygame.draw.line(self.screen, self.fire_color, start_pos, end_pos, 2)  # Line thickness of 2

            # Add label for the fire
            label = self.font.render("Fire", True, self.fire_color)
            self.screen.blit(label, (x + radius + 5, y))  # Offset the label slightly to the right

    # Generate a random target location in the environment.
    def generate_random_target(self):
        target_x = random.randint(0, 100)  # Assuming the environment size is 100x100
        target_y = random.randint(0, 100)
        return target_x, target_y

    # Generates random positions for different types of obstacles.
    def generate_obstacles(self, num_humans, num_buildings, num_trees, num_animals):
        self.humans = [{"type": "Human", "position": (random.randint(0, 100), random.randint(0, 100))} for _ in range(num_humans)]
        self.buildings = [{"type": "Building", "position": (random.randint(0, 100), random.randint(0, 100))} for _ in range(num_buildings)]
        self.trees = [{"type": "Tree", "position": (random.randint(0, 100), random.randint(0, 100))} for _ in range(num_trees)]
        self.animals = [{"type": "Animal", "position": (random.randint(0, 100), random.randint(0, 100))} for _ in range(num_animals)]

    # Draws obstacles (humans, buildings, trees, and animals) on the screen.
    def draw_obstacles(self):
        # Draw humans (circles)
        for pos in self.humans:
            x, y = int((pos["position"][0] / 100) * self.window_width), int((pos["position"][1] / 100) * self.window_height)
            pygame.draw.circle(self.screen, self.human_color, (x, y), 10)
            label = self.font.render("Human", True, self.human_color)
            self.screen.blit(label, (x + 10, y))

        # Draw buildings (rectangles)
        for pos in self.buildings:
            x, y = int((pos["position"][0] / 100) * self.window_width), int((pos["position"][1] / 100) * self.window_height)
            pygame.draw.rect(self.screen, self.building_color, (x, y, 40, 40))  # Rectangle for building
            label = self.font.render("Building", True, self.building_color)
            self.screen.blit(label, (x + 5, y - 15))

        # Draw trees (triangles)
        for pos in self.trees:
            x, y = int((pos["position"][0] / 100) * self.window_width), int((pos["position"][1] / 100) * self.window_height)
            pygame.draw.polygon(self.screen, self.tree_color, [(x, y - 20), (x - 15, y + 10), (x + 15, y + 10)])
            label = self.font.render("Tree", True, self.tree_color)
            self.screen.blit(label, (x + 10, y))

        # Draw animals (small circles)
        for pos in self.animals:
            x, y = int((pos["position"][0] / 100) * self.window_width), int((pos["position"][1] / 100) * self.window_height)
            pygame.draw.circle(self.screen, self.animal_color, (x, y), 7)
            label = self.font.render("Animal", True, self.animal_color)
            self.screen.blit(label, (x + 10, y))

    def draw_drones_and_target(self, drone_positions):
        # White background screen
        self.screen.fill((255, 255, 255))

        # Draw the target location as a red circle
        target_x, target_y = self.target_location
        target_x = int((target_x / 100) * self.window_width)
        target_y = int((target_y / 100) * self.window_height)
        pygame.draw.circle(self.screen, (255, 0, 0), (target_x, target_y), self.target_radius)
        
        self.draw_obstacles()       # Draw the obstacles (humans, buildings, trees, and animals)
        self.draw_no_fly_zones()    # Draw no-fly zones
        self.draw_fires()           # Draw fires

        # Label the drone with its ID
        target_label = self.font.render(f"Target Location", True, (255, 0, 0))  # Red text for ID label
        self.screen.blit(target_label, (target_x + 15, target_y -10))  # Offset the label slightly to the right of the drone

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

    # Updates visualizer with the new drone positions.
    def update(self, drone_positions):
        self.check_for_quit()                            # Check for quit events
        self.draw_drones_and_target(drone_positions)     # Draw the drones in their current positions
        self.clock.tick(self.fps)                        # Limit the frame rate

    # Logs drone positions and target location each step
    def log_drones(self, step, drone_positions, target_location):
        target_x, target_y = target_location
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            for i, pos in enumerate(drone_positions):
                x, y = pos[:2]
                writer.writerow([step, i, x, y, target_x, target_y])

    def close(self):
        pygame.quit()

class DamageMeter:
    def __init__(self, num_drones, window_width=400, window_height=600):
        # Initialize drone health attributes
        self.drone_health = {i: 100 for i in range(num_drones)}  # Start each drone with full health (100)
        self.collision_info = {i: "None" for i in range(num_drones)}  # Track last obstacle that caused damage
        self.window_width = window_width
        self.window_height = window_height
        self.num_drones = num_drones
        self.drone_height = self.window_height // num_drones  # Height for each drone's health bar in the visualizer

        # Initialize pygame window for damage meter
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Drone Damage Meter")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()
        self.fps = 30  # Set frames per second
    # Decreases the health of a specific drone and updates the collision info.
    def take_damage(self, drone_id, obstacle_type, damage=10):
        if drone_id in self.drone_health:
            self.drone_health[drone_id] = max(0, self.drone_health[drone_id] - damage)  # Decrease health
            self.collision_info[drone_id] = obstacle_type  # Update last collision cause

    # Draws the health bars for each drone in a new pygame window.
    def draw_damage_meter(self):
        self.screen.fill((255, 255, 255))  # White background

        for drone_id in range(self.num_drones):
            # Calculate position and dimensions for each health bar
            bar_x = 50
            bar_y = drone_id * self.drone_height + 10
            bar_width = 200
            bar_height = self.drone_height - 20

            # Calculate health bar fill based on current health
            health_percentage = self.drone_health[drone_id] / 100
            health_color = (255 * (1 - health_percentage), 255 * health_percentage, 0)  # Gradient from red to green
            health_fill_width = int(bar_width * health_percentage)

            # Draw background for health bar
            pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
            # Draw health fill
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_fill_width, bar_height))

            # Display drone ID, health, and last collision cause
            drone_info = f"Drone {drone_id + 1} - Health: {self.drone_health[drone_id]}% - Last Hit: {self.collision_info[drone_id]}"
            label = self.font.render(drone_info, True, (0, 0, 0))  # Black text
            self.screen.blit(label, (bar_x + bar_width + 10, bar_y))

        pygame.display.update()
        self.clock.tick(self.fps)  # Maintain FPS

    # Check for quit event and continuously update damage meter display.
    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.draw_damage_meter()


# Add custom logging and tracking of the environment's state, actions, and rewards. 
# It logs these interactions at each step during training and writes them into a JSON file.
# In addition, it includes a pause function.
class CustomMonitor(gym.Wrapper):
    # Initializes internal lists for rewards, episode length, and the log data
    def __init__(self, env, log_file='gym_log_data.json', initial_state_file='initial_state.json', 
                 training_controller=None, collision_avoidance=None, num_drones=0,
                 num_no_fly_zones=0, num_fires=0, num_humans=0, num_buildings=0, num_trees=0,
                 num_animals=0,):
        super(CustomMonitor, self).__init__(env)             # Calls the parent class (gym.Wrapper) constructor to set up the environment.
        self.log_file = log_file
        self.initial_state_file = initial_state_file
        self.episode_rewards = []
        self.episode_length = 0
        self.log_data = []            # List to store all log entries
        self.initial_state_data = []  # List to store initial state information
        self.episode_id = 0  # Track episode ID

        # Initialize RunLogger for detailed step logging
        self.run_logger = RunLogger()
        self.training_controller = training_controller
        self.collision_avoidance = collision_avoidance
        
        # Initialize the visualizer
        self.visualizer = DroneVisualizer(num_drones=num_drones, num_no_fly_zones=num_no_fly_zones, num_fires=num_fires, num_humans=num_humans,
                                          num_buildings=num_buildings, num_trees=num_trees, num_animals=num_animals)
        self.no_fly_zones = self.visualizer.no_fly_zones                                        # Store the number of no-fly zones and their properties for logging
        self.fires = self.visualizer.fires                                                      # Store the number of fire zones and their properties for logging
        self.visualizer.generate_obstacles(num_humans, num_buildings, num_trees, num_animals)   # Generate obstacles after initializing the visualizer

    # reset(): Resets the environment to start a new episode. It also logs the initial state, agent states, and other matrices (like distance_matrix and angle_matrix).
    # If these attributes are not available, it samples from the observation space.
    def reset(self, **kwargs):
        self.episode_rewards = []       # Resets the environment and initializes logging for a new episode.
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
    """
    # Computes a reward based on how close drones are to the target and each other. Penalize drones that are far from the target by using the average distance.
    def calculate_swarm_reward(self, drone_positions, target_location):
        target_x, target_y = target_location
        # Calculate average distance of drones to the target
        total_distance_to_target = 0
        for pos in drone_positions:
            drone_x, drone_y = pos[:2]
            distance_to_target = np.sqrt((drone_x - target_x)**2 + (drone_y - target_y)**2)
            total_distance_to_target += distance_to_target
        average_distance_to_target = total_distance_to_target / len(drone_positions)

        # Reward is inversely proportional to the average distance to the target
        # Reward will be higher when drones are closer to the target
        reward = -average_distance_to_target
        return reward
    """

    # Calculate the centroid of the swarm (average of all drone positions)
    def calculate_swarm_center(self, drone_positions):
        x_cords = [pos[0] for pos in drone_positions]
        y_cords = [pos[1] for pos in drone_positions]
        center_x = sum(x_cords) / len(x_cords)
        center_y = sum(y_cords) / len(y_cords)
        return center_x, center_y

    # Target Proximity Reward: Encourages drones to approach the target location. Swarm Cohesion Penalty: Encourages drones to stay close to the swarm center.
    # Formation Penalty: Encourages drones to maintain a desired formation (e.g., circular radius). No-Fly Zone and Fire Penalty: Discourages drones from approaching hazardous areas.
    def calculate_swarm_reward(self, drone_positions, target_location):
        target_x, target_y = target_location

        # Distance to target reward (existing)
        target_proximity_reward = 0
        for pos in drone_positions:
            distance_to_target = np.sqrt((pos[0] - target_x) ** 2 + (pos[1] - target_y) ** 2)
            target_proximity_reward += max(1.0 / (1 + distance_to_target), 0)
        target_proximity_reward /= len(drone_positions)

        # Cohesion penalty based on distance from swarm center
        swarm_center_x = sum([pos[0] for pos in drone_positions]) / len(drone_positions)
        swarm_center_y = sum([pos[1] for pos in drone_positions]) / len(drone_positions)
        swarm_cohesion_penalty = sum([np.sqrt((pos[0] - swarm_center_x) ** 2 + (pos[1] - swarm_center_y) ** 2)
                                      for pos in drone_positions]) / len(drone_positions)

        # Formation penalty for maintaining ideal distance
        ideal_radius = 20
        formation_penalty = sum([abs(np.sqrt((pos[0] - target_x) ** 2 + (pos[1] - target_y) ** 2) - ideal_radius)
                                 for pos in drone_positions]) / len(drone_positions)

        # No-fly zone and fire zone penalty
        no_fly_zone_penalty = 0
        fire_penalty = 0
        safe_distance = 10  # Minimum safe distance from no-fly zones and fires
        for pos in drone_positions:
            for zone in self.no_fly_zones:
                if np.sqrt((pos[0] - zone['position'][0]) ** 2 + (pos[1] - zone['position'][1]) ** 2) < zone[
                    'size'] + safe_distance:
                    no_fly_zone_penalty += 10
            for fire in self.fires:
                if np.sqrt((pos[0] - fire['position'][0]) ** 2 + (pos[1] - fire['position'][1]) ** 2) < fire[
                    'size'] + safe_distance:
                    fire_penalty += 10

        # Combine all reward and penalty terms
        total_reward = target_proximity_reward - 0.1 * swarm_cohesion_penalty - 0.1 * formation_penalty - no_fly_zone_penalty - fire_penalty
        return total_reward

    # step(action): Executes a single step in the environment using the provided action. 
    # It logs the current state, next state, reward, and other environment-specific information like distance and angle matrices. 
    # This information is appended to log_data, which tracks the whole episode.
    def step(self, action):
        while not self.training_controller.pause_event.is_set():         # Check if the controller has paused
            time.sleep(0.1)                                              # Sleep for a bit until the visualizer is resumed
        
        # Logs state, action, reward, next state, and done.
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)
        distance_matrix = self.world.distance_matrix if hasattr(self.world, 'distance_matrix') else None
        angle_matrix = self.world.angle_matrix if hasattr(self.world, 'angle_matrix') else None

        drone_positions = self.env.get_drone_positions()    # Get drone positions (this assumes your environment has a get_drone_positions method)

        # Log each drone's data
        for i, drone in enumerate(drone_positions):
            x, y = drone[0], drone[1]
            orientation_unscaled = drone[2]  # Assume this is the orientation
            linear_velocity = drone[3]
            angular_velocity = drone[4]
            self.run_logger.log_step(self.episode_id, i, x, y, orientation_unscaled, linear_velocity, angular_velocity)      # Log the data for each drone

        swarm_center = self.calculate_swarm_center(drone_positions)  # Pass swarm center (for swarm behavior)
        target_reward = self.calculate_swarm_reward(drone_positions, self.visualizer.target_location)      # Get the reward based on distance to target and swarm behavior
        
        # **Collision Avoidance Logic**:
        #all_obstacles = self.visualizer.trees + self.visualizer.buildings + self.visualizer.humans + self.visualizer.animals
        all_obstacles = self.visualizer.get_all_obstacles()
        no_fly_zones = self.visualizer.no_fly_zones
        fires = self.visualizer.fires
        
        # Apply collision avoidance to adjust the drone positions
        #self.collision_avoidance.avoid_collisions(drone_positions, obstacles, no_fly_zones, swarm_center)
        self.collision_avoidance.avoid_collisions(drone_positions, all_obstacles, no_fly_zones, fires, swarm_center)
# ------------------------------------------------------------------------------------------------------------------------ #
        #self.collision_avoidance.avoid_collisions(drone_positions, all_obstacles, no_fly_zones, fires)
        self.visualizer.update(drone_positions)         # Update the visualizer with the current drone positions after collision avoidance
        total_reward = reward + target_reward           # Add the swarm reward to the normal environment reward# Rewards are negative
                               
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
            "drone_positions" : drone_positions,
            "movement_reward" : total_reward,
            "no_fly-zones" : self.no_fly_zones,   # Logs the size/positions of the no_fly_zones
            "num of fire zones" : self.fires,  # Logs size/positions of num_fires
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
            self.episode_length += 1             # Increments the episode step count.
        #self.episode_rewards.append(reward)  # Logs the reward obtained in this step.
        self.episode_rewards.append(total_reward)

        #return next_state, reward, done, info
        return next_state, total_reward, done, info

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
          num_no_fly_zones=0, num_fires=0, num_humans=0, num_buildings=0, num_trees=0, num_animals=0,):
    
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
                                  num_drones=num_drones, num_no_fly_zones=num_no_fly_zones, num_fires=num_fires, num_humans=num_humans,
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
    
    # Get user input for how many drones, obstacles no_fly_zones, and fires to include.
    try:
        num_drones = int(input("Enter the number of drones to run in the rendezvous mission: "))
        num_no_fly_zones = int(input("Enter the number of no-fly zones to generate: "))
        num_fires = int(input("Enter the number of fire zones to generate: "))
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
    #train(num_timesteps=10, log_dir=log_dir, num_drones=num_drones)
    #train(num_timesteps=10, log_dir=log_dir, num_drones=num_drones, num_no_fly_zones=num_no_fly_zones, num_humans=num_humans, num_buildings=num_buildings, num_trees=num_trees, num_animals=num_animals)
    
    # Create a separate thread for the training process - this allows the visualizer to display
    # pygame requires a continuous event loop to render and update the window. If the event loop is blocked or not running, the window may not display or refresh correctly.
    # TRPO algorithm might block the pygame window from refreshing properly
    training_thread = threading.Thread(target=train, args=(num_timesteps, log_dir, training_controller, num_drones,
                                                           num_no_fly_zones, num_fires, num_humans, num_buildings,
                                                           num_trees, num_animals))
    training_thread.start()

    # Wait for both threads to complete
    training_thread.join()
    input_thread.join()
    
    # Keep the Pygame visualizer running in the main thread to prevent it from freezing
    #while training_thread.is_alive():
        #pygame.time.wait(100)  # Adds a small delay to avoid busy-waiting
        #pygame.event.pump()    # This allows pygame to process its events like window close

if __name__ == '__main__':
    main()