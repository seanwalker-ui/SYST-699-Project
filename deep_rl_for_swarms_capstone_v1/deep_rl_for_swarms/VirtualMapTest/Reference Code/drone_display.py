import pygame
import sys
from drone_positions import get_drone_positions, export_log_to_csv

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Display")

# Define drone colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow, Cyan
DRONE_RADIUS = 5  # Diameter = 10 pixels, radius = 5

# Initialize font
font = pygame.font.SysFont(None, 24)  # Default system font with size 24

# Set the frame rate
clock = pygame.time.Clock()

# Main loop
running = True
try:
    while running:
        # Handle events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill((0, 0, 0))  # Fill with black

        # Get drone positions
        positions = get_drone_positions()

        # Draw each drone and the labels
        for i, (x, y) in enumerate(positions):
            pygame.draw.circle(screen, colors[i], (x, y), DRONE_RADIUS)
            
            # Render the label text ("Drone 1", "Drone 2", etc.)
            label = f"Drone {i + 1}"
            text_surface = font.render(label, True, (255, 255, 255))  # White text
            screen.blit(text_surface, (x + 10, y - 10))  # Offset the text a bit to avoid overlap

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(60)

except KeyboardInterrupt:
    print("Simulation interrupted.")

finally:
    # Ensure the CSV is exported when the simulation ends
    export_log_to_csv()  # Export the log data to a CSV file
    print("Log data exported to CSV.")

# Quit Pygame
pygame.quit()
sys.exit()