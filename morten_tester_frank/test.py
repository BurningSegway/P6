from frankx import Affine, LinearRelativeMotion, Robot

robot = Robot("192.168.2.30")
robot.set_dynamic_rel(0.05)

motion = LinearRelativeMotion(Affine(0.2, 0.0, 0.0)) #Meters
robot.move(motion)
