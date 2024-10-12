import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Path to your CSV file (replace with the actual path)
csv_file = "C:\\Users\\sawal\\Documents\\VS_Code\\Output_Test\\drone_log.csv"

# Read the CSV file into a pandas dataframe
df = pd.read_csv(csv_file)

# Plotting static trajectory for a specific drone
def plot_individual_drone_trajectory(drone_id):
    plt.figure(figsize=(10, 6))

    # Filter data for the selected drone
    drone_data = df[df['drone_id'] == drone_id]

    # Plot the x, y positions of the selected drone
    plt.plot(drone_data['x'], drone_data['y'], label=f'Drone {drone_id}')
    
    plt.title(f'Drone {drone_id} Movement Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to allow user to select individual drone trajectories
def plot_drone_trajectories():
    unique_drones = df['drone_id'].unique()
    
    while True:
        # Show available drones
        print("\nAvailable Drones:")
        for drone_id in unique_drones:
            print(f"Drone {drone_id}")
        
        # Prompt user to select a drone
        try:
            selected_drone = int(input("\nEnter the Drone ID to view its trajectory (or type -1 to exit): "))
            if selected_drone == -1:
                break  # Exit the loop
            elif selected_drone in unique_drones:
                plot_individual_drone_trajectory(selected_drone)
            else:
                print("Invalid Drone ID. Please select a valid one.")
        except ValueError:
            print("Please enter a valid integer Drone ID.")

# Function to create an animation of the drone movement over time
def animate_drone_movement():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 640)  # Set X-axis limits (same as your screen size)
    ax.set_ylim(0, 480)  # Set Y-axis limits (same as your screen size)
    
    # Initialize scatter points for each drone
    drones_scatter = [ax.scatter([], [], s=50, label=f'Drone {drone_id}') for drone_id in df['drone_id'].unique()]

    # Update function for animation
    def update(frame):
        for drone_id, scatter in enumerate(drones_scatter, start=1):
            drone_data = df[(df['drone_id'] == drone_id) & (df['time'] <= frame)]
            scatter.set_offsets(drone_data[['x', 'y']].values)

    ani = animation.FuncAnimation(fig, update, frames=df['time'].max(), interval=50, repeat=False)
    
    # Display the plot with animation
    plt.title('Drone Movement Animation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function to choose between static plot or animation
if __name__ == "__main__":
    print("1. Plot static drone trajectories")
    print("2. Animate drone movement over time")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        plot_drone_trajectories()
    elif choice == "2":
        animate_drone_movement()
    else:
        print("Invalid choice!")