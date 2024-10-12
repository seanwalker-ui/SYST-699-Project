import pygame
import math
import random
import rend # import the rendezvous module for drone movement
import collision  # Import the collision module to avoid collisions

# Initialize Pygame
def initialize_pygame():
    pygame.init()
    
    # Set up display
    width, height = 1280, 960
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone Display")
    return window, width, height

def draw_lift_off_area(window, font, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height):
    # Draw the lift-off area as a dotted rectangle
    for i in range(lift_off_area_x, lift_off_area_x + lift_off_area_width, 10):
        pygame.draw.line(window, (255, 255, 255), (i, lift_off_area_y), (i + 5, lift_off_area_y))
        pygame.draw.line(window, (255, 255, 255), (i, lift_off_area_y + lift_off_area_height), (i + 5, lift_off_area_y + lift_off_area_height))

    for i in range(lift_off_area_y, lift_off_area_y + lift_off_area_height, 10):
        pygame.draw.line(window, (255, 255, 255), (lift_off_area_x, i), (lift_off_area_x, i + 5))
        pygame.draw.line(window, (255, 255, 255), (lift_off_area_x + lift_off_area_width, i), (lift_off_area_x + lift_off_area_width, i + 5))

    # Label the lift-off area
    label = font.render("Lift-Off Area", True, (255, 255, 255))
    window.blit(label, (lift_off_area_x + lift_off_area_width // 2 - label.get_width() // 2, lift_off_area_y - 30))

def draw_target(window, font, target_x, target_y):
    # Draw the target location with an 'X'
    pygame.draw.line(window, (255, 0, 0), (target_x - 10, target_y - 10), (target_x + 10, target_y + 10), 2)
    pygame.draw.line(window, (255, 0, 0), (target_x - 10, target_y + 10), (target_x + 10, target_y - 10), 2)
    
    # Label the target area
    label = font.render("Target Location", True, (255, 255, 255))
    window.blit(label, (target_x - label.get_width() // 2, target_y - 20))

def draw_obstacles(window, obstacles, no_fly_zones):
    # Draw obstacles with different shapes and colors
    for obstacle in obstacles:
        if obstacle["type"] == "human":
            pygame.draw.circle(window, obstacle["color"], (obstacle["x"], obstacle["y"]), 15)  # Red circle for humans
        elif obstacle["type"] == "tree":
            pygame.draw.polygon(window, obstacle["color"], [(obstacle["x"], obstacle["y"] - 20), 
                                                            (obstacle["x"] - 10, obstacle["y"] + 10), 
                                                            (obstacle["x"] + 10, obstacle["y"] + 10)])  # Green triangle for trees
        elif obstacle["type"] == "animal":
            pygame.draw.rect(window, obstacle["color"], pygame.Rect(obstacle["x"] - 10, obstacle["y"] - 5, 20, 10))  # Blue rectangle for animals
        elif obstacle["type"] == "building":
            pygame.draw.rect(window, obstacle["color"], pygame.Rect(obstacle["x"] - 20, obstacle["y"] - 40, 40, 80))  # Gray rectangle for buildings
    
    # Draw the no-fly zones
    for no_fly_zone in no_fly_zones:
        draw_no_fly_zone(window, no_fly_zone)

def draw_no_fly_zone(window, no_fly_zone):
    # Draw the no-fly zone as a yellow dashed square
    x, y, size = no_fly_zone["x"], no_fly_zone["y"], no_fly_zone["size"]
    for i in range(x, x + size, 10):
        pygame.draw.line(window, (255, 255, 0), (i, y), (i + 5, y))  # Top border
        pygame.draw.line(window, (255, 255, 0), (i, y + size), (i + 5, y + size))  # Bottom border
    for i in range(y, y + size, 10):
        pygame.draw.line(window, (255, 255, 0), (x, i), (x, i + 5))  # Left border
        pygame.draw.line(window, (255, 255, 0), (x + size, i), (x + size, i + 5))  # Right border
    
    # Label the no-fly zone
    font = pygame.font.SysFont(None, 16)
    label = font.render("No Fly Zone", True, (255, 255, 0))
    window.blit(label, (x + size // 2 - label.get_width() // 2, y - 20))
      
# Draw the drone
def draw_drone(window, drone, drone_radius):
    pygame.draw.circle(window, drone.color, (int(drone.x), int(drone.y)), drone_radius)

# Function to generate a target location at least 200 pixels away from the lift-off area
#def generate_safe_target(width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width,
                         #lift_off_area_height, min_distance=200, no_fly_zones, min_distance_from_no_fly_zone=100):
def generate_safe_target(width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height,
                         no_fly_zones, min_distance=200, min_distance_from_no_fly_zone=100):
    while True:
        target_x = random.randint(100, width - 100)
        target_y = random.randint(100, height - 100)

        # Calculate the nearest distance from the target to the edge of the lift-off area
        #lift_off_center_x = lift_off_area_x + lift_off_area_width / 2
        #lift_off_center_y = lift_off_area_y + lift_off_area_height / 2

        # Calculate the nearest distance from the target to the edge of the lift-off area
        dx_lift_off = max(lift_off_area_x - target_x, target_x - (lift_off_area_x + lift_off_area_width), 0)
        dy_lift_off = max(lift_off_area_y - target_y, target_y - (lift_off_area_y + lift_off_area_height), 0)
        distance_from_lift_off = math.sqrt(dx_lift_off**2 + dy_lift_off**2)
        
        # Check if the target is far enough from all no-fly zones
        is_safe_from_no_fly_zones = all(
            math.sqrt((target_x - no_fly_zone["x"]) ** 2 + (target_y - no_fly_zone["y"]) ** 2) >= min_distance_from_no_fly_zone
            for no_fly_zone in no_fly_zones
        )

        # If the distance is greater than the minimum safe distance from the lift-off area and the target is safe from no-fly zones, return the target
        if distance_from_lift_off >= min_distance and is_safe_from_no_fly_zones:
            return target_x, target_y
        
        """
        dx = max(lift_off_area_x - target_x, target_x - (lift_off_area_x + lift_off_area_width), 0)
        dy = max(lift_off_area_y - target_y, target_y - (lift_off_area_y + lift_off_area_height), 0)
        distance = math.sqrt(dx**2 + dy**2)

        # If the distance is greater than the minimum safe distance, return the target
        if distance >= min_distance:
            return target_x, target_y
        """

# Function to generate a random set of obstacles, ensuring they are not too close to the lift-off area or target
def generate_obstacles(num_obstacles, width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, target_x, target_y, min_distance_from_lift_off=200, min_distance_from_target=100):
    obstacle_types = ["human", "tree", "animal", "building"]
    obstacles = []
    
    for _ in range(num_obstacles):
        while True:
            obstacle_type = random.choice(obstacle_types)
            x = random.randint(100, width - 100)
            y = random.randint(100, height - 100)
            
            # Ensure obstacle is at least min_distance_from_lift_off away from the lift-off area
            dx_lift_off = max(lift_off_area_x - x, x - (lift_off_area_x + lift_off_area_width), 0)
            dy_lift_off = max(lift_off_area_y - y, y - (lift_off_area_y + lift_off_area_height), 0)
            distance_from_lift_off = math.sqrt(dx_lift_off**2 + dy_lift_off**2)
            
            # Ensure obstacle is at least min_distance_from_target away from the target location
            distance_from_target = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

            # If the obstacle is far enough from both the lift-off area and the target location, add it to the list
            if distance_from_lift_off >= min_distance_from_lift_off and distance_from_target >= min_distance_from_target:
                color = {"human": (255, 0, 0), "tree": (0, 255, 0), "animal": (0, 0, 255), "building": (128, 128, 128)}
                obstacles.append({"type": obstacle_type, "x": x, "y": y, "color": color[obstacle_type]})
                break
    return obstacles

# Render the label text ("Drone 1", "Drone 2", etc.)
def render_labels(window, font, drone, i, drone_radius):
    label = font.render(f"Drone {i + 1}", True, (255, 255, 255))  # White text
    window.blit(label, (int(drone.x) - label.get_width() // 2, int(drone.y) - drone_radius - 15))  # Label above the drone

def update_drones(drones, obstacles, no_fly_zones, swarm_center_x, swarm_center_y, swarm_radius, swarm_angles, target_x, target_y, log_file):
    # Update and draw each drone
    for i, drone in enumerate(drones):
        if not drone.lifted_off:
            # Lift-off phase
            drone.lift_off()
        else:
            # Assign the target swarm position if not done already
            if drone.swarm_angle is None:
                drone.join_swarm(swarm_center_x, swarm_center_y, swarm_radius, swarm_angles[i])
            
            # Move to the swarm position
            drone.move_to_swarm_position(drones, obstacles, no_fly_zones)

            if drone.reached_swarm:
                drone.move_to_random_target(target_x, target_y, drones, obstacles, no_fly_zones)
    
    # Check for and log collisions with drones and obstacles
    #collision.avoid_collisions(drones, obstacles, log_file=log_file)
    # Check for and log collisions with drones, obstacles, and no-fly zones
    collision.avoid_collisions(drones, obstacles, no_fly_zones, log_file=log_file)

# Game loop
def game_loop(window, width, height, drones, obstacles, no_fly_zones, lift_off_area_x, lift_off_area_y, lift_off_area_width, 
              lift_off_area_height, swarm_center_x, swarm_center_y, swarm_radius, swarm_angles, font, 
              drone_radius, target_x, target_y, log_file):
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill((0, 0, 0))  # Clear the screen with black

        # Draw lift-off area and target location
        draw_lift_off_area(window, font, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height)
        draw_target(window, font, target_x, target_y)
        
        # Draw obstacles and no-fly zones
        draw_obstacles(window, obstacles, no_fly_zones)

        # Update drones and check for collisions with obstacles and no fly zones
        update_drones(drones, obstacles, no_fly_zones, swarm_center_x, swarm_center_y, swarm_radius, swarm_angles, target_x, target_y, log_file)

        # Draw each drone and render its label
        for i, drone in enumerate(drones):
            draw_drone(window, drone, drone_radius)
            render_labels(window, font, drone, i, drone_radius)

        pygame.display.flip()  # Update the display
        clock.tick(60)  # Cap the frame rate

    pygame.quit()

def main():
    window, width, height = initialize_pygame()

    # Define drone properties
    drone_radius = 5
    drone_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Lift-off area properties
    lift_off_area_x = 340
    lift_off_area_y = height - 150
    lift_off_area_width = 400
    lift_off_area_height = 100
    num_drones = 5
    drone_spacing = 60

    # Create drones
    drones = rend.create_drones_in_lift_off_area(lift_off_area_x, lift_off_area_y, lift_off_area_width, 
                                                 num_drones, drone_spacing, drone_colors)
    
    # Random target location
    #target_x, target_y = generate_safe_target(width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, min_distance=200)

    # Generate no-fly zones (maximum 3)
    #no_fly_zones = generate_no_fly_zones(3, width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height)
    # Generate no-fly zones (maximum 3)
    no_fly_zones = collision.generate_no_fly_zones(3, width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, zone_size=75)
    
    # Generate a random target location, ensuring it is not inside a no-fly zon
    target_x, target_y = generate_safe_target(width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, no_fly_zones, min_distance=200, min_distance_from_no_fly_zone=100)
    
    # Generate obstacles (ensuring they are not within 50 pixels of the target)
    obstacles = generate_obstacles(20, width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height,
                                   target_x, target_y, min_distance_from_lift_off=200, min_distance_from_target=100)

    # Font for labels
    font = pygame.font.SysFont(None, 16)

    # Swarm properties
    swarm_center_x = width // 2
    swarm_center_y = height // 2
    swarm_radius = 40
    swarm_angles = [i * (2 * math.pi / num_drones) for i in range(num_drones)]

    # Initialize the collision log file
    log_file = "collision_log.csv"
    collision.initialize_log(log_file)

    # Start the game loop
    game_loop(window, width, height, drones, obstacles, no_fly_zones, lift_off_area_x, lift_off_area_y, 
              lift_off_area_width, lift_off_area_height, swarm_center_x, swarm_center_y, swarm_radius, 
              swarm_angles, font, drone_radius, target_x, target_y, log_file)

if __name__ == '__main__':
    main()



















## _______________________________ Keep as a reference _______________________________ ##

"""
import pygame
import math
import random
import rend
import collision  # Import the collision module to avoid collisions

# Initialize Pygame
def initialize_pygame():
    pygame.init()
    
    # Set up display
    width, height = 1280, 960
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone Display")
    return window, width, height

def draw_lift_off_area(window, font, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height):
    # Draw the lift-off area as a dotted rectangle
    for i in range(lift_off_area_x, lift_off_area_x + lift_off_area_width, 10):
        pygame.draw.line(window, (255, 255, 255), (i, lift_off_area_y), (i + 5, lift_off_area_y))
        pygame.draw.line(window, (255, 255, 255), (i, lift_off_area_y + lift_off_area_height), (i + 5, lift_off_area_y + lift_off_area_height))

    for i in range(lift_off_area_y, lift_off_area_y + lift_off_area_height, 10):
        pygame.draw.line(window, (255, 255, 255), (lift_off_area_x, i), (lift_off_area_x, i + 5))
        pygame.draw.line(window, (255, 255, 255), (lift_off_area_x + lift_off_area_width, i), (lift_off_area_x + lift_off_area_width, i + 5))

    # Label the lift-off area
    label = font.render("Lift-Off Area", True, (255, 255, 255))
    window.blit(label, (lift_off_area_x + lift_off_area_width // 2 - label.get_width() // 2, lift_off_area_y - 30))


def draw_target(window, font, target_x, target_y):
    # Draw the target location with an 'X'
    pygame.draw.line(window, (255, 0, 0), (target_x - 10, target_y - 10), (target_x + 10, target_y + 10), 2)
    pygame.draw.line(window, (255, 0, 0), (target_x - 10, target_y + 10), (target_x + 10, target_y - 10), 2)
    
    # Label the target area
    label = font.render("Target Location", True, (255, 255, 255))
    window.blit(label, (target_x - label.get_width() // 2, target_y - 20))

# Function to generate a target location at least 200 pixels away from the lift-off area
def generate_safe_target(width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, min_distance=200):
    while True:
        target_x = random.randint(100, width - 100)
        target_y = random.randint(100, height - 100)

        # Calculate the nearest distance from the target to the edge of the lift-off area
        lift_off_center_x = lift_off_area_x + lift_off_area_width / 2
        lift_off_center_y = lift_off_area_y + lift_off_area_height / 2

        dx = max(lift_off_area_x - target_x, target_x - (lift_off_area_x + lift_off_area_width), 0)
        dy = max(lift_off_area_y - target_y, target_y - (lift_off_area_y + lift_off_area_height), 0)
        distance = math.sqrt(dx**2 + dy**2)

        # If the distance is greater than the minimum safe distance, return the target
        if distance >= min_distance:
            return target_x, target_y

# Draw the drone
def draw_drone(window, drone, drone_radius):
    pygame.draw.circle(window, drone.color, (int(drone.x), int(drone.y)), drone_radius)

# Render the label text ("Drone 1", "Drone 2", etc.)
def render_labels(window, font, drone, i, drone_radius):
    label = font.render(f"Drone {i + 1}", True, (255, 255, 255))  # White text
    window.blit(label, (int(drone.x) - label.get_width() // 2, int(drone.y) - drone_radius - 15))  # Label above the drone


def update_drones(drones, swarm_center_x, swarm_center_y, swarm_radius, swarm_angles, target_x, target_y, log_file):
    # Update and draw each drone
    for i, drone in enumerate(drones):
        if not drone.lifted_off:
            # Lift-off phase
            drone.lift_off()
        else:
            # Assign the target swarm position if not done already
            if drone.swarm_angle is None:
                drone.join_swarm(swarm_center_x, swarm_center_y, swarm_radius, swarm_angles[i])
            
            # Move to the swarm position
            drone.move_to_swarm_position(drones)

            if drone.reached_swarm:
                drone.move_to_random_target(target_x, target_y, drones)
    
    # Check for and log collisions
    collision.avoid_collisions(drones, log_file=log_file)

# Game loop
def game_loop(window, width, height, drones, lift_off_area_x, lift_off_area_y, lift_off_area_width, 
              lift_off_area_height, swarm_center_x, swarm_center_y, swarm_radius, swarm_angles, font, 
              drone_radius, target_x, target_y, log_file):
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill((0, 0, 0))  # Clear the screen with black

        # Draw lift-off area and target location
        draw_lift_off_area(window, font, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height)
        draw_target(window, font, target_x, target_y)
        
        # Update drones and check for collisions
        update_drones(drones, swarm_center_x, swarm_center_y, swarm_radius, swarm_angles, target_x, target_y, log_file)

        # Draw each drone and render its label
        for i, drone in enumerate(drones):
            draw_drone(window, drone, drone_radius)
            render_labels(window, font, drone, i, drone_radius)

        pygame.display.flip()  # Update the display
        clock.tick(60)  # Cap the frame rate

    pygame.quit()

def main():
    window, width, height = initialize_pygame()

    # Define drone properties
    drone_radius = 5
    drone_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Lift-off area properties
    lift_off_area_x = 340
    lift_off_area_y = height - 150
    lift_off_area_width = 400
    lift_off_area_height = 100
    num_drones = 5
    drone_spacing = 60

    # Create drones
    drones = rend.create_drones_in_lift_off_area(lift_off_area_x, lift_off_area_y, lift_off_area_width, 
                                                 num_drones, drone_spacing, drone_colors)

    # Font for labels
    font = pygame.font.SysFont(None, 16)

    # Swarm properties
    swarm_center_x = width // 2
    swarm_center_y = height // 2
    swarm_radius = 40
    swarm_angles = [i * (2 * math.pi / num_drones) for i in range(num_drones)]
    
    
    # Random target location
    #target_x = random.randint(100, width - 100)
    #target_y = random.randint(100, height - 100)
    
    
    # Generate a random target location at least 200 pixels away from the lift-off area
    target_x, target_y = generate_safe_target(width, height, lift_off_area_x, lift_off_area_y, lift_off_area_width, lift_off_area_height, min_distance=200)
    
    # Initialize the collision log file
    log_file = "C:\\Users\\sawal\\Documents\\VS_Code\\Output_Test\\collision_log.csv"
    collision.initialize_log(log_file)

    # Start the game loop
    game_loop(window, width, height, drones, lift_off_area_x, lift_off_area_y, 
              lift_off_area_width, lift_off_area_height, swarm_center_x, swarm_center_y, swarm_radius, 
              swarm_angles, font, drone_radius, target_x, target_y, log_file)

if __name__ == '__main__':
    main()
"""