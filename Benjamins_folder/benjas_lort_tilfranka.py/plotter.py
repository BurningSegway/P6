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
        [f"joint_pose_{i}" for i in range(7)] +
        [f"joint_vel_{i}" for i in range(7)] +
        ["obj_x", "obj_y", "obj_z"] + 
        [f"action_{i}" for i in range(7)] +
        [f"dof_pose{i}" for i in range(7)]    
    )

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'b']
    df = pd.read_csv("test.csv", header=None, names=column_names)
    
    # Set up the plot figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot joint velocities on the right y-axis (ax2)
    for i in range(0,6):
        #ax1.plot(df[f'joint_vel_{i}'], label=f'Joint {i} velocity', color=colors[i], linewidth=2)
        ax1.plot(df[f'joint_pose_{i}'], label=f'Joint {i} pose', color=colors[i], linewidth=2)
        #ax1.plot(df[f'dof_pose{i}'], label=f'dof pose {i} ', color=colors[i], linewidth=2)
        ax2.plot(df[f'action_{i}'], label=f'action {i}', color=colors[i], linestyle='--', linewidth=2)
        
    

    # Set labels and titles
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Position (radians or meters)')
    ax2.set_ylabel('Velocity (rad/s or m/s)')
    plt.title('Joint Positions and Velocities Over Time')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True)
    plt.tight_layout()
    plt.show()



def plot_simple():
    column_names = (
        [f"joint_pose_{i}" for i in range(7)] +
        [f"gripper_pose_1", "gripper_pose_2"] +
        [f"joint_vel_{i}" for i in range(7)] +
        ["obj_x", "obj_y", "obj_z"] +
        [f"dof_pose{i}" for i in range(7)] + 
        [f"action_{i}" for i in range(8)] +
        ["distance"]
    )


    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'b']
    #df = pd.read_csv("test.csv", header=None, names=column_names)
    df = pd.read_csv("P6/test.csv", header=None, names=column_names)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'b']
    df = pd.read_csv("test.csv", header=None, names=column_names)
    
    # Set up the plot figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot joint velocities on the right y-axis (ax2)
    for i in range(0,4):
        #ax2.plot(df[f'joint_vel_{i}'], label=f'Joint {i} velocity', color=colors[i], linewidth=2)
        ax2.plot(df[f'joint_pose_{i}'], label=f'Joint {i} pose', color=colors[i], linewidth=2)
        #ax2.plot(df[f'dof_pose{i}'], label=f'dof pose {i} ', color=colors[i], linewidth=2)
    
    # Plot actions and distance on the left y-axis (ax1)
    for i in range(0,4):
        ax1.plot(df[f'action_{i}'], label=f'action {i}', color=colors[i], linestyle='--', linewidth=2)
        #ax1.plot(df[f'joint_pose_{i}'], label=f'Joint {i} position', color=colors[i],linestyle='--', linewidth=2)
    ax1.plot(df[f'distance'], label=f'distance', linewidth=2, linestyle=':', color='blue')

    #ax2.set_ylim([-3.0, 3.0])

    # Set labels and titles
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Position (radians or meters)')
    ax2.set_ylabel('Velocity (rad/s or m/s)')
    plt.title('Joint Positions and Velocities Over Time')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True)
    plt.tight_layout()
    plt.show()


        # Set up the plot figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot joint velocities on the right y-axis (ax2)
    for i in range(4,7):
        #ax2.plot(df[f'joint_vel_{i}'], label=f'Joint {i} velocity', color=colors[i], linewidth=2)
        #ax2.plot(df[f'joint_pose_{i}'], label=f'Joint {i} pose', color=colors[i], linewidth=2)
        ax2.plot(df[f'dof_pose{i}'], label=f'dof pose {i} ', color=colors[i], linewidth=2)
    
    # Plot actions and distance on the left y-axis (ax1)
    for i in range(4,8):
        ax1.plot(df[f'action_{i}'], label=f'action {i}', color=colors[i], linestyle='--', linewidth=2)
        #ax1.plot(df[f'joint_pose_{i}'], label=f'Joint {i} position', color=colors[i], linestyle='--', linewidth=2)
    ax1.plot(df[f'distance'], label=f'distance', linewidth=2, linestyle=':', color='blue')


    #ax2.set_ylim([-3.0, 3.0])
    # Set labels and titles
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Position (radians or meters)')
    ax2.set_ylabel('Velocity (rad/s or m/s)')
    plt.title('Joint Positions and Velocities Over Time')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True)
    plt.tight_layout()
    plt.show()




def plot_simple_2():
    column_names = (
        [f"joint_pose_{i}" for i in range(7)] +
        [f"gripper_pose_1", "gripper_pose_2"] +
        [f"joint_vel_{i}" for i in range(7)] +
        ["obj_x", "obj_y", "obj_z"] +
        [f"dof_pose{i}" for i in range(7)] + 
        [f"action_{i}" for i in range(8)] +
        ["distance"]
    )


    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'b']
    df = pd.read_csv("test2.csv", header=None, names=column_names)
    
    # Set up the plot figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot joint velocities on the right y-axis (ax2)
    for i in range(0,6):
        ax1.plot(df[f'joint_vel_{i}'], label=f'Joint {i} velocity', color=colors[i], linewidth=2)
        #ax1.plot(df[f'joint_pose_{i}'], label=f'Joint {i} pose', color=colors[i], linewidth=2)
        #ax1.plot(df[f'dof_pose{i}'], label=f'dof pose {i} ', color=colors[i], linewidth=2)
        ax2.plot(df[f'action_{i}'], label=f'action {i}', color=colors[i], linestyle='--', linewidth=2)
        
    

    # Set labels and titles
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Position (radians or meters)')
    ax2.set_ylabel('Velocity (rad/s or m/s)')
    plt.title('Joint Positions and Velocities Over Time')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True)
    plt.tight_layout()
    plt.show()


        # Set up the plot figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot joint velocities on the right y-axis (ax2)
    for i in range(0,7):
        #ax1.plot(df[f'joint_vel_{i}'], label=f'Joint {i} velocity', color=colors[i], linewidth=2)
        #ax1.plot(df[f'joint_pose_{i}'], label=f'Joint {i} pose', color=colors[i], linewidth=2)
        ax1.plot(df[f'dof_pose{i}'], label=f'dof pose {i} ', color=colors[i], linewidth=2)
        ax2.plot(df[f'action_{i}'], label=f'action {i}', color=colors[i], linestyle='--', linewidth=2)
    
    # Plot actions and distance on the left y-axis (ax1)
    #for i in range(4,7):
        #ax1.plot(df[f'action_{i}'], label=f'action {i}', color=colors[i], linestyle='--', linewidth=2)
        #ax1.plot(df[f'joint_pose_{i}'], label=f'Joint {i} position', color=colors[i], linestyle='--', linewidth=2)
   #ax1.plot(df[f'distance'], label=f'distance', linewidth=2, linestyle=':', color='blue')


    #ax2.set_ylim([-3.0, 3.0])
    # Set labels and titles
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Position (radians or meters)')
    ax2.set_ylabel('Velocity (rad/s or m/s)')
    plt.title('Joint Positions and Velocities Over Time')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True)
    plt.tight_layout()
    plt.show()















#plot_ala()
#plot_basic()
plot_simple()
"""

#Range
#Række 1
self.target_pos = np.array([0.65, 0.34, 0.0])
self.target_pos = np.array([0.65, 0.17, 0.0])
self.target_pos = np.array([0.65, 0.0, 0.0])
self.target_pos = np.array([0.65, -0.17, 0.0])
self.target_pos = np.array([0.65, -0.34, 0.0])
#Række 2
self.target_pos = np.array([0.51, 0.34, 0.0])
self.target_pos = np.array([0.51, 0.17, 0.0])
self.target_pos = np.array([0.51, 0.0, 0.0])
self.target_pos = np.array([0.51, -0.17, 0.0])
self.target_pos = np.array([0.51, -0.34, 0.0])
#Række 3
self.target_pos = np.array([0.35, 0.34, 0.0])
self.target_pos = np.array([0.35, 0.17, 0.0])
self.target_pos = np.array([0.35, 0.0, 0.0])
self.target_pos = np.array([0.35, -0.17, 0.0])
self.target_pos = np.array([0.35, -0.34, 0.0])
#Række 4
self.target_pos = np.array([0.24, 0.34, 0.0])
self.target_pos = np.array([0.24, 0.17, 0.0])
self.target_pos = np.array([0.24, -0.17, 0.0])
self.target_pos = np.array([0.24, -0.34, 0.0])
#Række 5
self.target_pos = np.array([0.1, 0.34, 0.0])
self.target_pos = np.array([0.1, -0.34, 0.0])






#Within area
#Række 1
self.target_pos = np.array([0.6, 0.25, 0.0])
self.target_pos = np.array([0.6, 0.08, 0.0])
self.target_pos = np.array([0.6, -0.08, 0.0])
self.target_pos = np.array([0.6, -0.25, 0.0])
#Række 2
self.target_pos = np.array([0.53, 0.25, 0.0])
self.target_pos = np.array([0.53, 0.08, 0.0])
self.target_pos = np.array([0.53, -0.08, 0.0])
self.target_pos = np.array([0.53, -0.25, 0.0])
#Række 3
self.target_pos = np.array([0.46, 0.25, 0.0])
self.target_pos = np.array([0.46, 0.08, 0.0])
self.target_pos = np.array([0.46, -0.08, 0.0])
self.target_pos = np.array([0.46, -0.25, 0.0])
#Række 4
self.target_pos = np.array([0.4, 0.25, 0.0])
self.target_pos = np.array([0.4, 0.08, 0.0])
self.target_pos = np.array([0.4, -0.08, 0.0])
self.target_pos = np.array([0.4, -0.25, 0.0])

#Tilfældige
"""