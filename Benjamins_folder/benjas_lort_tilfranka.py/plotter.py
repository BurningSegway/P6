import pandas as pd
import matplotlib.pyplot as plt




def plot_ala():
    # In your visualization script:
    column_names = (
        [f"action_joint_{i} "for i in range(7)] +
        [f"action_gripper"] + 
        [f"joint_pos_{i}" for i in range(7)] +
        [f"joint_pos_gripper_{i}" for i in range(2)] +
        [f"joint_vel_{i}" for i in range(7)] +
        [f"joint_vel_gripper_{i}" for i in range(2)] +
        ["obj_x", "obj_y", "obj_z"] +
        ["target_obj_x", "target_obj_y", "target_obj_z"] +
        [f"euler_{i}" for i in range(4)]
    )

    df = pd.read_csv("DATA_alabenja_01_-01_002.csv", header=None, names=column_names)
    # Set up the plot figure
    plt.figure(figsize=(12, 8))

    # Plot joint positions (columns 8-16 in your original buffer)
    for i in range(7):
        plt.plot(df[f'joint_pos_{i}'], label=f'Joint {i} Position')

    plt.xlabel('Time steps')
    plt.ylabel('Position (radians or meters)')
    plt.title('Joint Positions Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_basic():
    column_names = (
        [f"progress_buff"] + 
        [f"joint_pos_{i}" for i in range(7)] +
        [f"joint_vel_{i}" for i in range(7)] +
        ["obj_x", "obj_y", "obj_z"]
    )

    df = pd.read_csv("DATA_01_-01_002.csv", header=None, names=column_names)
    # Set up the plot figure
    plt.figure(figsize=(12, 8))

    # Plot joint positions (columns 8-16 in your original buffer)
    for i in range(7):
        plt.plot(df[f'joint_pos_{i}'], label=f'Joint {i} Position')

    plt.xlabel('Time steps')
    plt.ylabel('Position (radians or meters)')
    plt.title('Joint Positions Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#plot_ala()
plot_basic()