# Example function that returns positions for 5 drones.
# In a real use case, this might be reading from sensors, a network, or some external source.
import random
import math
import csv
import time
import signal

# Initialize random positions, velocities, and accelerations for each drone
positions = [(random.randint(0, 640), random.randint(0, 480)) for _ in range(5)]
velocities = [(random.choice([-2, -1, 1, 2]), random.choice([-2, -1, 1, 2])) for _ in range(5)]
accelerations = [(0, 0) for _ in range(5)]  # Initialize accelerations for each drone
DRONE_RADIUS = 5  # 10 pixels diameter, so radius is 5
SAFE_DISTANCE = 30  # 30 pixels. Safe distance to avoid collision (greater than the diameter)
MAX_SPEED = 4  # Maximum speed for the drones
ACCELERATION_RATE = 0.1  # Rate of acceleration
DECELERATION_RATE = 0.1  # Rate of deceleration when avoiding a collision
REPULSION_FORCE = 0.2  # Force to repel drones when they get too close

# Log list to store the data for each drone
drone_log = []

# Get the current time for logging purposes
start_time = time.time()

def distance(x1, y1, x2, y2):
    """Calculate the distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_orientation(vx, vy):
    """Calculate the orientation of the drone based on its velocity vector."""
    return math.degrees(math.atan2(vy, vx))

def avoid_collisions():
    """Adjust velocities if drones are too close to each other to avoid collision."""
    global accelerations
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):  # Compare each drone to each other
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            
            # Calculate the distance between the two drones
            dist = distance(x1, y1, x2, y2)
            
            # If the distance is less than the safe distance, adjust velocities
            if dist < SAFE_DISTANCE:
                # Calculate the direction to move away from each other
                dx = x2 - x1
                dy = y2 - y1
                if dist == 0:
                    dist = 0.1  # Prevent division by zero
                
                # Normalize the direction vector
                dx /= dist
                dy /= dist
                
                # Apply repulsion force based on proximity
                repulsion_x = dx * REPULSION_FORCE
                repulsion_y = dy * REPULSION_FORCE

                # Apply repulsion to both drones' velocities
                velocities[i] = (velocities[i][0] - repulsion_x, velocities[i][1] - repulsion_y)
                velocities[j] = (velocities[j][0] + repulsion_x, velocities[j][1] + repulsion_y)
                
                """
                # Apply deceleration for both drones
                accelerations[i] = (-dx * DECELERATION_RATE, -dy * DECELERATION_RATE)
                accelerations[j] = (dx * DECELERATION_RATE, dy * DECELERATION_RATE)
            else:
                # Gradually stop acceleration if no collision is near
                accelerations[i] = (0, 0)
                accelerations[j] = (0, 0)
                """

def limit_speed(vx, vy):
    """Limit the speed to the maximum speed."""
    speed = math.sqrt(vx**2 + vy**2)
    
    # Factor in the speed if the speed is greater than the set MAXIMUM speed.
    if speed > MAX_SPEED:
        factor = MAX_SPEED / speed
        vx *= factor
        vy *= factor
    return vx, vy

def log_drone_data():
    """Log the data for each drone at the current time step."""
    current_time = time.time() - start_time
    for i, (x, y) in enumerate(positions):
        vx, vy = velocities[i]
        ax, ay = accelerations[i]
        orientation = calculate_orientation(vx, vy)  # Calculate orientation based on velocity
        drone_log.append({
            "time": current_time,
            "drone_id": i + 1,
            "x": x,
            "y": y,
            "velocity_x": vx,
            "velocity_y": vy,
            "acceleration_x": ax,
            "acceleration_y": ay,
            "orientation": orientation
        })

def export_log_to_csv(filename="C:\\Users\\sawal\\Documents\\VS_Code\\Output_Test\\drone_log.csv"):
    """Export the collected drone data to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["time", "drone_id", "x", "y", "velocity_x", "velocity_y", "acceleration_x", "acceleration_y", "orientation"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in drone_log:
            writer.writerow(entry)

def get_drone_positions():
    global positions, velocities, acceleration
    new_positions = []
    
    for i, (x, y) in enumerate(positions):
        vx, vy = velocities[i]
        ax, ay = accelerations[i]
        
        # Apply acceleration to the velocity
        vx += ax
        vy += ay

        # Update the drone's position by its velocity
        x += vx
        y += vy

        # Boundary checks: if a drone hits the screen edge, reverse its direction and decelerate
        if x < 0 or x > 640:
            vx = -vx
            ax = -ax * DECELERATION_RATE
        if y < 0 or y > 480:
            vy = -vy
            ay = -ay * DECELERATION_RATE

        # Limit the speed of the drones
        vx, vy = limit_speed(vx, vy)

        # Store the new position and velocity. Also, store the acceleration to alter velocity.
        new_positions.append((x, y))
        velocities[i] = (vx, vy)
        accelerations[i] = (ax, ay)
    
    positions = new_positions
    
    # Check for collisions after updating positions
    avoid_collisions()
    
    # Log the current state of the drones
    log_drone_data()
    return positions

# This is an example of a simulation loop
def main_simulation():
    fps = 60  # Frames per second
    time_per_step = 1.0 / fps  # Time duration for each step

    try:
        for step in range(100):  # Simulate 100 steps
            get_drone_positions()
            time.sleep(time_per_step)  # Delay to simulate real-time movement at 60 FPS
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        # Export the CSV when the simulation is done or interrupted
        export_log_to_csv()
        print("Log data exported to CSV.")

if __name__ == "__main__":
    main_simulation()
 
    """
    # Hardcoded positions for example purposes.
    # Replace these with dynamic data as needed.
    return [
        (100, 100),  # Drone 1
        (200, 150),  # Drone 2
        (300, 250),  # Drone 3
        (400, 300),  # Drone 4
        (500, 350),  # Drone 5
    ]
    """
