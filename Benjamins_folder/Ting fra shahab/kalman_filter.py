import numpy as np
from numpy.linalg import inv

class KalmanFilter822:
    def __init__(self, system_matrix: np.ndarray,
                 input_matrix: np.ndarray, output_matrix: np.ndarray):
        self.F = system_matrix
        self.G = input_matrix
        self.H = output_matrix

        size_of_x = self.F.shape[0]
        self.x_estimated = np.zeros((size_of_x, 1))
        self.x_predicted = self.x_estimated

    def predict(self, control_input: np.ndarray):
        self.x_predicted = self.F @ self.x_estimated + self.G @ control_input
        return self.x_predicted
    
    def update(self, sensor_input: np.ndarray):
        K = self.H.T @ inv(self.H @ self.H.T)
        self.x_estimated = self.x_predicted + \
                           K @ (sensor_input - self.H @ self.x_predicted)
        return self.x_estimated




###############
#    def update(self, sensor_input: np.ndarray):
#        K = self.H.T @ inv(self.H @ self.H.T)
#        self.x_estimated = self.x_predicted + \
#                           K @ (sensor_input - self.H @ self.x_predicted)
#        return self.x_estimated
#
#Ligning fra slides
#S = self.H @ 
#
#
#

###############