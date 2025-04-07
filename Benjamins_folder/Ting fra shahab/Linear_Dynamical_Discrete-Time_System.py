

#x_k is the robot’s position
#y_k is the robot’s position
#theta_k is the orientation,
#v_k is the linear velocity,
#omega_k is the angular velocity.


x = [x_k, y_k, theta_k, v_k, omega_k]



F





#a_k is the robots linear acceleration
#alpha_k is the robots angular acceleration

u = [a_k, alpha_k]


#represents how the control inputs affect the state change
G = [0, 0;
     0, 0;
     0, 0;
     0, 0;
     delta_t, 0;
     0, delta_t;]

#updates the robots positional output
y = [x_k, y_k]


# something something Defines how the state relates to measurements
H = [1, 0, 0, 0, 0;
     0, 1, 0, 0, 0;]