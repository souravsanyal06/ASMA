import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def plot_trajectory(npy_file, instruction_id):
    """Load and visualize a saved 3D trajectory from a .npy file."""
    try:
        trajectory = np.load(npy_file)
        if trajectory.ndim != 2 or trajectory.shape[1] < 3:
            print(f"Error: Invalid trajectory shape {trajectory.shape}. Expected (N,3).")
            return

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory in 3D (X, Y, Z) without markers, and use a consistent color.
        ax.plot(
            trajectory[:, 0], 
            trajectory[:, 1], 
            trajectory[:, 2],
            linestyle='-', 
            color='blue', 
            label=f"Instruction {instruction_id}"
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")

        # Restrict Z-axis to [0, 6]
        ax.set_zlim(0, 10)
        # ax.set_ylim(-5, 5)

        ax.set_title(f"3D Drone Trajectory for Instruction {instruction_id}")
        ax.legend()
        ax.grid(True)

        plt.show()

    except FileNotFoundError:
        print(f"Error: File {npy_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 3D drone trajectory from a .npy file.")
    parser.add_argument("instruction_id", type=int, help="Instruction ID (1-4)")
    
    args = parser.parse_args()
    npy_file = f"/home/sourav/ASMA/dataset/trajectories/drone_trajectory_instr_{args.instruction_id}.npy"

    plot_trajectory(npy_file, args.instruction_id)
