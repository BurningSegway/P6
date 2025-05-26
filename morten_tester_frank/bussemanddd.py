import frankx
import numpy as np

robot = frankx.Robot("192.168.2.30")
robot.set_dynamic_rel(0.05)

robot_default_dof_pos = np.array([-0.217, 0.698, 0.050, 0.239, 0.435, 0.767, -1.175])


robot.move(frankx.JointMotion(robot_default_dof_pos.tolist()))
dof_pos = robot_default_dof_pos # + 0.25 * (np.random.rand(7) - 0.5) #start position offset
robot.move(frankx.JointMotion(dof_pos.tolist()))