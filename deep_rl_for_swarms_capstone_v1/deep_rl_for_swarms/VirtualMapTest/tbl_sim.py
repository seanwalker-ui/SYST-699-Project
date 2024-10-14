import pandas as pd
import pygame
import sys
import math

# Load the drone data from the Excel file
file_path = 'F:\\Fall 2024\\SYST-699\\Week 7\\Data Dictionary and Positions\\tbl_local_state.xlsx'  # Update this path if needed
#df = pd.read_excel(file_path)
df = pd.read_excel(file_path, engine='openpyxl')

# Initialize Pygame and set up the window
pygame.init()
screen = pygame.display.set_mode((1280, 980))
pygame.display.set_caption('Drone Movement Simulation')

# Define colors
WHITE = (255, 255, 255)
DRONE_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)

# Simulation settings
FPS = 60
drone_radius = 5  # 10 pixels diameter

# Convert coordinates to match Pygame window scale (example scaling factor)
def scale_coords(x, y):
    return int(x * 10), int(y * 10)  # Adjust this scaling factor to fit your simulation

# Main loop for simulation
def run_simulation():
    clock = pygame.time.Clock()

    # Set up font for labeling drones
    font = pygame.font.SysFont(None, 24)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the screen with white background
        screen.fill(WHITE)

        # Draw each drone based on the data
        for _, row in df.iterrows():
            x, y = scale_coords(row['x_coord'], row['y_coord'])
            orientation = row['orientation']
            drone_id = row['drone_id']
            
            # Draw drone as a circle with a 10-pixel diameter
            pygame.draw.circle(screen, DRONE_COLOR, (x, y), drone_radius)
                      
            # Draw drone orientation as a line
            end_x = x + int(math.cos(orientation) * drone_radius * 2)
            end_y = y + int(math.sin(orientation) * drone_radius * 2)
            pygame.draw.line(screen, (0, 0, 0), (x, y), (end_x, end_y), 2)

            # Label the drone with its ID
            label = font.render(f'Drone ID: {drone_id}', True, TEXT_COLOR)
            screen.blit(label, (x + 10, y))  # Offset label from the circle for clarity
            
        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    run_simulation()