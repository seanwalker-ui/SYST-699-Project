import math
import csv
import time
import random

# Function to calculate the distance between two objects
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Initialize CSV file and writer for logging collisions
def initialize_log(file_name="collision_log.csv"):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Drone ID", "Other Object", "Distance", "Correction X", "Correction Y", "Message"])  # Header

# Function to log a collision event
def log_collision(drone_id, other_id, distance, correction_x, correction_y, file_name="collision_log.csv"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Generate a message for collision event
    message = f"Drone {drone_id + 1} avoided obstacle: {other_id}. Adjusted its path."

    # Print the message to the terminal
    print(f"[{timestamp}] {message}")
    
    # Log the event in the CSV file
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, drone_id, other_id, round(distance, 2), round(correction_x, 2), round(correction_y, 2), message])

# Function to calculate the safe area (circular area) around a drone
def is_in_safe_zone(drone1, drone2, min_distance):
    distance = calculate_distance(drone1, drone2)
    return distance < min_distance

# Function to avoid collisions with obstacles and no-fly zones
def avoid_collisions(drones, obstacles, no_fly_zones, min_distance=30, log_file="collision_log.csv"):
    for i, drone in enumerate(drones):
        # Avoid collisions with obstacles
        for obstacle in obstacles:
            distance = calculate_distance(drone.x, drone.y, obstacle["x"], obstacle["y"])
            if distance < min_distance + 20:  # Extra buffer for obstacles
                # Adjust drone to avoid obstacle
                overlap = (min_distance + 20) - distance
                dx = drone.x - obstacle["x"]
                dy = drone.y - obstacle["y"]
                angle = math.atan2(dy, dx)

                correction_x = math.cos(angle) * overlap
                correction_y = math.sin(angle) * overlap
                drone.x += correction_x
                drone.y += correction_y

                # Log the collision event with obstacle
                log_collision(i, obstacle["type"], distance, correction_x, correction_y, log_file)

        # Avoid entering no-fly zones
        for no_fly_zone in no_fly_zones:
            if no_fly_zone["x"] <= drone.x <= no_fly_zone["x"] + no_fly_zone["size"] and \
               no_fly_zone["y"] <= drone.y <= no_fly_zone["y"] + no_fly_zone["size"]:
                # Move the drone outside the no-fly zone
                dx = drone.x - (no_fly_zone["x"] + no_fly_zone["size"] / 2)
                dy = drone.y - (no_fly_zone["y"] + no_fly_zone["size"] / 2)
                angle = math.atan2(dy, dx)

                correction_x = math.cos(angle) * (min_distance + 20)
                correction_y = math.sin(angle) * (min_distance + 20)
                drone.x += correction_x
                drone.y += correction_y

                # Log the collision event with the no-fly zone
                log_collision(i, "No Fly Zone", min_distance + 20, correction_x, correction_y, log_file)

"""
# Function to adjust drone positions to avoid collisions with other drones or obstacles
def avoid_collisions(drones, obstacles, no_fly_zones, min_distance=30, log_file="C:\\Users\\sawal\\Documents\\VS_Code\\Output_Test\\collision_log.csv"):
    for i, drone in enumerate(drones):
        # Avoid collisions with other drones
        for j, other_drone in enumerate(drones):
            if i != j:
                distance = calculate_distance(drone.x, drone.y, other_drone.x, other_drone.y)
                if distance < min_distance:
                    # Adjust drones to maintain min_distance
                    overlap = min_distance - distance
                    dx = drone.x - other_drone.x
                    dy = drone.y - other_drone.y
                    angle = math.atan2(dy, dx)

                    correction_x = math.cos(angle) * (overlap / 2)
                    correction_y = math.sin(angle) * (overlap / 2)
                    drone.x += correction_x
                    drone.y += correction_y
                    other_drone.x -= correction_x
                    other_drone.y -= correction_y

                    # Log the collision event between drones
                    log_collision(i, j, distance, correction_x, correction_y, log_file)

        # Avoid collisions with obstacles
        for j, obstacle in enumerate(obstacles):
            distance = calculate_distance(drone.x, drone.y, obstacle["x"], obstacle["y"])
            if distance < min_distance + 20:  # Extra buffer for obstacles
                # Adjust drone to avoid obstacle
                overlap = (min_distance + 20) - distance
                dx = drone.x - obstacle["x"]
                dy = drone.y - obstacle["y"]
                angle = math.atan2(dy, dx)

                correction_x = math.cos(angle) * overlap
                correction_y = math.sin(angle) * overlap
                drone.x += correction_x
                drone.y += correction_y

                # Log the collision event with obstacle
                log_collision(i, obstacle["type"], distance, correction_x, correction_y, log_file)
        
         # Avoid entering no-fly zones
        for no_fly_zone in no_fly_zones:
            if no_fly_zone["x"] <= drone.x <= no_fly_zone["x"] + no_fly_zone["size"] and \
               no_fly_zone["y"] <= drone.y <= no_fly_zone["y"] + no_fly_zone["size"]:
                # Move the drone outside the no-fly zone
                dx = drone.x - (no_fly_zone["x"] + no_fly_zone["size"] / 2)
                dy = drone.y - (no_fly_zone["y"] + no_fly_zone["size"] / 2)
                angle = math.atan2(dy, dx)

                correction_x = math.cos(angle) * (min_distance + 20)
                correction_y = math.sin(angle) * (min_distance + 20)
                drone.x += correction_x
                drone.y += correction_y

                # Log the collision event with the no-fly zone
                log_collision(i, "No Fly Zone", min_distance + 20, correction_x, correction_y, log_file)
"""

# Function to generate random no-fly zones
def generate_no_fly_zones(num_zones, width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, zone_size=100):
    no_fly_zones = []
    
    for _ in range(num_zones):
        while True:
            x = random.randint(100, width - 100 - zone_size)
            y = random.randint(100, height - 100 - zone_size)
            
            # Ensure the no-fly zone is not within the lift-off area
            if not (lift_off_area_x <= x <= lift_off_area_x + lift_off_area_width and
                    lift_off_area_y <= y <= lift_off_area_y + lift_off_area_height):
                no_fly_zones.append({"x": x, "y": y, "size": zone_size})
                break
    return no_fly_zones