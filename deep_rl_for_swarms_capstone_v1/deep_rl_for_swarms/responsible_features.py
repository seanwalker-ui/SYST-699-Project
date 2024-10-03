# ------------------------------------------------------------------------------- #
# --------------- SW code - get and define RAI features --------------------- #
# ------------------------------------------------------------------------------- #

# This code defines the responsibility features for responsible_features.py
import numpy as np

class ResponsibleAI:
    def __init__(self, min_safe_distance=5, alert_distance=2):
        # minimum safe distance from obstacles
        self.min_safe_distance = min_safe_distance  
        # distance for triggering alerts
        self.alert_distance = alert_distance        

    def avoid_collisions(self, drone_positions, obstacle_positions):
        """
        This function checks the distance between drones and obstacles.
        If any drone is too close to an obstacle, it adjusts its path.
        """
        for i, drone_pos in enumerate(drone_positions):
            for j, obs_pos in enumerate(obstacle_positions):
                #distance = np.linalg.norm(drone_pos - obs_pos)
                distance = np.linalg.norm(drone_pos[:2] - obs_pos[:2])  # Ensure both positions are 2D
                
                # If too close to an obstacle, take evasive action
                if distance < self.min_safe_distance:
                    print(f"Drone {i} is too close to obstacle {j}! Adjusting path...")
                    # Example: Move drone away from the obstacle
                    self.adjust_path(i, drone_positions, obs_pos)
                elif distance < self.alert_distance:
                    print(f"Drone {i} is within alert distance of obstacle {j}.")
                    self.raise_alert(i, j)

# ---------------------------------------- No Fly Zone Data ------------------------------------------------#
    def avoid_no_fly_zones(self, drone_positions):
        """Ensure drones avoid no-fly zones."""
        for i, drone_pos in enumerate(drone_positions):
            if self.is_in_no_fly_zone(drone_pos):
                print(f"Drone {i} is inside a no-fly zone! Adjusting path...")
                self.adjust_path_out_of_no_fly_zone(i, drone_positions)

    def adjust_path_out_of_no_fly_zone(self, drone_index, drone_positions):
        """Adjust the drone's path to exit a no-fly zone."""
        drone_pos = drone_positions[drone_index][:2]
        # Example: Move the drone directly away from the no-fly zone boundary
        for zone in self.no_fly_zones:
            if zone.contains(Point(drone_pos)):
                # Adjust by moving out in a straight line
                closest_point = zone.boundary.interpolate(zone.boundary.project(Point(drone_pos)))
                direction_away = drone_pos - closest_point.coords[0]
                adjusted_position = drone_pos + direction_away / np.linalg.norm(direction_away)
                drone_positions[drone_index][:2] = adjusted_position
                print(f"Adjusted position of Drone {drone_index} to avoid no-fly zone.")
# ---------------------------------------------------------------------------------------------------------#

    def adjust_path(self, drone_index, drone_positions, obstacle_position):
        """
        Adjust the path of the drone to avoid collision.
        """
        drone_pos = drone_positions[drone_index][:2]  # Only use the x, y position (first two elements)
        direction_away_from_obstacle = drone_pos - obstacle_position[:2]  # Ensure both are 2D

        # Normalize the direction vector and adjust the drone's position
        adjusted_position = drone_pos + 0.5 * direction_away_from_obstacle / np.linalg.norm(direction_away_from_obstacle)

        # Update the position back in the drone_positions list (keeping original structure)
        drone_positions[drone_index][:2] = adjusted_position  # Modify only the position part
        print(f"Adjusted position of Drone {drone_index} to avoid collision.")
        
        # Log the adjustment
        self.log_decision(drone_index, f"Adjusted path to avoid obstacle at {obstacle_position}.")
    
    def raise_alert(self, drone_index, obstacle_index):
        """
        Raise alert when a drone is too close to an obstacle.
        """
        print(f"Alert! Drone {drone_index} is too close to obstacle {obstacle_index}.")
        
        # Log the alert
        self.log_decision(drone_index, f"Raised alert for proximity to obstacle {obstacle_index}.")

    def log_decision(self, drone_id, decision):
        """
        Log the decision for accountability and review later.
        Append each decision to the log file for tracking.
        """
        
        # Note: Change this file path to your choosing
        log_file_path = ''
        with open(log_file_path, 'a') as f:
            f.write(f"Drone {drone_id}: {decision}\n")

    def ensure_ethics(self, drone_states):
        """
        Ensure that the drones are behaving ethically.
        Ex: No drone should come too close to others or obstacles.
        """
        # Placeholder for more complex ethical checks
        print("Ensuring ethical operation of the drones...")