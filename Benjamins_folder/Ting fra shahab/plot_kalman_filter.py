from random import random
import numpy as np
import matplotlib.pyplot as plt
import kalman_filter as kf


sampling_time_Ts = 0.1

system_matrix_F = np.array([[1., sampling_time_Ts],
                            [0., 1.              ]])
input_matrix_G = np.array([[0.],
                           [sampling_time_Ts]])
output_matrix_H = np.array([[0., 1.]])

kalman_filter = kf.KalmanFilter822(system_matrix_F, input_matrix_G, output_matrix_H)

# Generate the input for the filter
acceleration_commands = [4 for _ in range(10)]
acceleration_commands.extend([0 for _ in range(10)])
acceleration_commands.extend([-4 for _ in range(10)])

# Integrate the acceleration into velocity and position
actual_velocity = [0.0]
for acc_command in acceleration_commands:
    last_velocity = actual_velocity[-1]
    actual_velocity.append(last_velocity + acc_command*sampling_time_Ts)
actual_position = [0.0]
for velocity in actual_velocity:
    last_position = actual_position[-1]
    actual_position.append(last_position + velocity*sampling_time_Ts)

# Run the commands through the filter with a zero value sensor reading
measured_velocities = []
predicted_positions = []
predicted_velocities = []
estimated_positions = []
estimated_velocities = []
for acc_command, act_vel in zip(acceleration_commands, actual_velocity):
    prediction = kalman_filter.predict( np.array([[acc_command]]) )
    predicted_positions.append(prediction[0])
    predicted_velocities.append(prediction[1])

    measured_vel = act_vel + random() *0.1
    measured_velocities.append(measured_vel)
    estimate = kalman_filter.update( np.array([[measured_vel]]) )
    estimated_positions.append(estimate[0])
    estimated_velocities.append(estimate[1])

#Plot Positions, velocities and accelerations
fig, (position_ax, velocity_ax, acceleration_ax) = plt.subplots(3,1)

position_ax.plot(actual_position, label="actual")
position_ax.plot(predicted_positions, label="predicted")
position_ax.plot(estimated_positions, label="estimated")
position_ax.set_title("Position")
position_ax.legend()

velocity_ax.plot(actual_velocity, label="actual")
velocity_ax.plot(measured_velocities, "-o", label="measured")
velocity_ax.plot(predicted_velocities, label="predicted")
velocity_ax.plot(estimated_velocities, label="estimated")
velocity_ax.set_title("Velocity")
velocity_ax.legend()

acceleration_ax.plot(acceleration_commands)
acceleration_ax.set_title("Acceleration Commands")
plt.show()
