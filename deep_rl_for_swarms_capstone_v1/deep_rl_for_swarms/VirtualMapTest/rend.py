import random
import math
import pygame # For the timer for Pygame's clock
import collision  # Import the collision avoidance logic

class Drone:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.lifted_off = False
        self.lift_off_velocity = 0  # Start with no initial velocity for slow lift-off
        self.max_velocity = 1.5  # Lower maximum speed for slower lift-off
        self.acceleration = 0.01  # Slow acceleration for gradual speed increase
        self.deceleration_start_y = 400  # Start decelerating when the drone reaches y = 400
        self.target_lift_off_height = 700  # Lift-off height limit

        # Swarm target position and movement
        self.swarm_center_x = None
        self.swarm_center_y = None
        self.swarm_radius = None
        self.swarm_angle = None
        self.rotation_speed = 0.01  # Rotation speed (smaller is slower)

        # Movement towards a random target
        self.target_x = None
        self.target_y = None
        self.swarm_speed = 2  # Speed of moving to the swarm position
        self.reached_swarm = False  # Flag to check if the drone has reached the swarm position
        self.target_reached = False  # Flag to check if the drone has reached the random target
        self.acceleration_rate = 0.02  # Acceleration rate to gradually increase speed
        self.current_speed = 0  # Introduce current speed variable for gradual acceleration
        
        # Timer to delay movement to the target
        self.swarm_timer_started = False
        self.swarm_timer = 0  # Timer to stay in the swarm formation
        self.global_swarm_timer = 5000  # Set 5 seconds for all drones to wait before moving
        
        #self.hovering_speed = random.uniform(0.05, 0.1)  # Randomized hovering speed for each drone
        #self.hovering_speed = 0.01
        #self.hover_angle = 0  # Angle for hovering effect
        #self.hover_amplitude = 5  # Max distance for hovering up and down

    def lift_off(self):
        if not self.lifted_off:
            # Accelerate or decelerate during the lift-off phase
            if self.y > self.deceleration_start_y:
                # Accelerating phase
                if self.lift_off_velocity < self.max_velocity:
                    self.lift_off_velocity += self.acceleration
            elif self.y <= self.deceleration_start_y:
                # Decelerating phase
                if self.lift_off_velocity > 0.5:  # Avoid stopping completely during deceleration
                    self.lift_off_velocity -= self.acceleration

            # Move upward by the current velocity
            self.y -= self.lift_off_velocity

            # Check if lift-off is complete
            if self.y < self.target_lift_off_height:
                self.lifted_off = True

    def join_swarm(self, center_x, center_y, radius, angle):
        # Assign a target position and radius for the swarm
        self.swarm_center_x = center_x
        self.swarm_center_y = center_y
        self.swarm_radius = radius
        self.swarm_angle = angle
 
    def accelerate_to_target(self, target_distance):
        if self.current_speed < self.max_velocity:
            self.current_speed += self.acceleration_rate
        if self.current_speed > target_distance:  # Prevent overshooting the target
            self.current_speed = target_distance
    
     ### -------------- This code is a working progress. Need to make sure that this is ready for official testing -------------------###
    def is_obstacle_near_position(self, x, y, obstacles, avoidance_distance=100):
        """Check if any obstacle is within `avoidance_distance` pixels of the given (x, y) position."""
        for obstacle in obstacles:
            distance = math.sqrt((x - obstacle["x"]) ** 2 + (y - obstacle["y"]) ** 2)
            if distance < avoidance_distance:
                return True
        return False

    def find_alternate_position(self, obstacles, avoidance_distance=100):
        """Find an alternate position for the drone to move around obstacles."""
        for angle_offset in range(0, 360, 15):  # Try different angles in increments of 15 degrees
            angle_radians = math.radians(angle_offset)
            new_x = self.swarm_center_x + math.cos(angle_radians) * (self.swarm_radius + avoidance_distance)
            new_y = self.swarm_center_y + math.sin(angle_radians) * (self.swarm_radius + avoidance_distance)
            if not self.is_obstacle_near_position(new_x, new_y, obstacles, avoidance_distance):
                return new_x, new_y  # Return a new safe position
        return self.swarm_center_x, self.swarm_center_y  # Fallback to the swarm center if no position is safe

    def follow_swarm(self, leader_drone):
        """Make the drone follow the position of another drone."""
        dx = leader_drone.x - self.x
        dy = leader_drone.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance > 1:  # Move towards the leader drone
            direction_x = dx / distance
            direction_y = dy / distance
            self.x += direction_x * self.current_speed
            self.y += direction_y * self.current_speed
     # ---------------------------------------------------------------------------------------------------------------- #
    
    def move_to_swarm_position(self, drones, obstacles, no_fly_zones):
        # Move gradually towards the swarm target position
        if self.lifted_off and not self.reached_swarm:
            # Gradually move to the swarm position and rotate
            dx = self.swarm_center_x + math.cos(self.swarm_angle) * self.swarm_radius - self.x
            dy = self.swarm_center_y + math.sin(self.swarm_angle) * self.swarm_radius - self.y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # -------------------------------------------------------------------------------------------------- #
            if self.is_obstacle_near_position(self.swarm_center_x, self.swarm_center_y, obstacles):
                # Find an alternate position if the swarm center is blocked by an obstacle
                new_x, new_y = self.find_alternate_position(obstacles)
                self.swarm_center_x, self.swarm_center_y = new_x, new_y
            # -------------------------------------------------------------------------------------------------- #
            if distance > 1:  # Move towards the swarm position first
                # Gradually accelerate towards the swarm position
                self.accelerate_to_target(distance)
                direction_x = dx / distance
                direction_y = dy / distance
                #self.x += direction_x * self.swarm_speed
                #self.y += direction_y * self.swarm_speed 
                self.x += direction_x * self.current_speed
                self.y += direction_y * self.current_speed
                    
            else:
                # Rotate the swarm once the drone is at its position
                self.swarm_angle += self.rotation_speed  # Increment angle for rotation
                self.x = self.swarm_center_x + math.cos(self.swarm_angle) * self.swarm_radius
                self.y = self.swarm_center_y + math.sin(self.swarm_angle) * self.swarm_radius
                self.reached_swarm = True  # Mark as reached the swarm position
                # self.target_reached = True # Mark as reached the target position
                        
                if not self.swarm_timer_started:
                    self.swarm_timer_started = True
                    self.swarm_timer = pygame.time.get_ticks()  # Start the timer
        
        # Rotate while in swarm position, whether the target is reached or not
        if self.reached_swarm:
            self.swarm_angle += self.rotation_speed  # Increment the rotation angle
            self.x = self.swarm_center_x + math.cos(self.swarm_angle) * self.swarm_radius
            self.y = self.swarm_center_y + math.sin(self.swarm_angle) * self.swarm_radius
        
        # Avoid collisions between drones while moving or rotating
        collision.avoid_collisions(drones, obstacles, no_fly_zones)
    
    """     
    def move_to_random_target(self):
        # Move the entire swarm to a new random target
        if self.reached_target and self.target_x is not None and self.target_y is not None:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance > 1:
                direction_x = dx / distance
                direction_y = dy / distance
                self.x += direction_x * self.swarm_speed
                self.y += direction_y * self.swarm_speed
            else:
                # Mark as reached the random target location
                self.reached_target = True
                # Once the drone reaches the random target, rotate around it
                self.swarm_angle += self.rotation_speed  # Rotate around the target
                self.x = self.target_x + math.cos(self.swarm_angle) * self.swarm_radius
                self.y = self.target_y + math.sin(self.swarm_angle) * self.swarm_radius
    """
    
    def move_to_random_target(self, target_x, target_y, drones, obstacles, no_fly_zones):
        # Ensure that all drones respect the same 5-second swarm wait time before moving
        current_time = pygame.time.get_ticks()
           
        # Check if any drone has already started moving to the target
        swarm_moving = any(drone.swarm_timer_started and current_time - drone.swarm_timer >= self.global_swarm_timer for drone in drones)
        """
        # Move the entire swarm to a new random target location
        if self.reached_swarm and not self.target_reached and current_time - self.swarm_timer >= self.global_swarm_timer:
            current_time = pygame.time.get_ticks()
            if current_time - self.swarm_timer >= 5000:  # 5 seconds delay
        """
        
        # Move the entire swarm to a new random target location
        if self.reached_swarm and not self.target_reached and (current_time - self.swarm_timer >= self.global_swarm_timer or swarm_moving):   
            # Gradually move the swarm center towards the target location
            dx = target_x - self.swarm_center_x
            dy = target_y - self.swarm_center_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance > 1:
                # Move the swarm center towards the target
                self.accelerate_to_target(distance)
                direction_x = dx / distance
                direction_y = dy / distance
                #self.swarm_center_x += direction_x * self.swarm_speed
                #self.swarm_center_y += direction_y * self.swarm_speed
                self.swarm_center_x += direction_x * self.current_speed
                self.swarm_center_y += direction_y * self.current_speed
                
                # Update each drone's position relative to the moving swarm center
                self.x = self.swarm_center_x + math.cos(self.swarm_angle) * self.swarm_radius
                self.y = self.swarm_center_y + math.sin(self.swarm_angle) * self.swarm_radius
            
            else:
                self.target_reached = True  # Mark as reached the target location
                #self.velocity = 0  # Reset velocity once the target is reached
                
        # Always rotate around the target once it is reached
        if self.target_reached:
            self.swarm_angle += self.rotation_speed
            self.x = self.swarm_center_x + math.cos(self.swarm_angle) * self.swarm_radius
            self.y = self.swarm_center_y + math.sin(self.swarm_angle) * self.swarm_radius
        
        # Avoid collisions between drones while moving to target
        collision.avoid_collisions(drones, obstacles, no_fly_zones)

    def set_random_target(self, screen_width, screen_height):
        self.target_x = random.randint(100, screen_width - 100)
        self.target_y = random.randint(100, screen_height - 100)
            
    """    
    def move_randomly(self, screen_width, screen_height):
        if self.lifted_off:
            self.x += self.speed_x
            self.y += self.speed_y

            # Bounce off the window edges
            if self.x <= 0 or self.x >= screen_width:
                self.speed_x *= -1
            if self.y <= 0 or self.y >= screen_height:
                self.speed_y *= -1
    """
# This function returns a list of Drone objects
def create_drones_in_lift_off_area(area_x, area_y, area_width, num_drones, spacing, colors):
    drones = []
    # Calculate the x position for the first drone
    start_x = area_x + (area_width - (num_drones - 1) * spacing) // 2

    # Create drones
    for i in range(num_drones):
        x = start_x + i * spacing
        y = area_y + 50  # Fixed y-position, middle of the lift-off area
        drone = Drone(x, y, colors[i])
        drones.append(drone)
    return drones