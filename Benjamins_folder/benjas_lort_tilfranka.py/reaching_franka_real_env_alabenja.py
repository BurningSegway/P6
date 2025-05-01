import gym
import time
import threading
import numpy as np
from packaging import version
from frankx import Robot, Gripper

import frankx


class ReachingFranka(gym.Env):
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0", control_space="blind_agent", motion_type="waypoint", camera_tracking=False):
        # gym API
        self._drepecated_api = version.parse(gym.__version__) < version.parse(" 0.25.0")

        self.device = device
        self.control_space = control_space  # joint or cartesian
        self.motion_type = motion_type  # waypoint or impedance

        if self.control_space == "cartesian" and self.motion_type == "impedance":
            # The operation of this mode (Cartesian-impedance) was adjusted later without being able to test it on the real robot.
            # Dangerous movements may occur for the operator and the robot.
            # Comment the following line of code if you want to proceed with this mode.
            raise ValueError("See comment in the code to proceed with this mode")
            pass

        # camera tracking (disabled by default)
        self.camera_tracking = camera_tracking
        if self.camera_tracking:
            threading.Thread(target=self._update_target_from_camera).start()

        # spaces
        if self.control_space == "blind_agent": #setting the control space as the blind agents
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32) #setting the action space (7 for joint and 1 for gripper)

        else:
            raise ValueError("Invalid control space:", self.control_space)
        

        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(36,), dtype=np.float32)
        

        # init real franka
        print("Connecting to robot at {}...".format(robot_ip))
        self.robot = frankx.Robot(robot_ip)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()

        # the robot's response can be better managed by independently setting the following properties, for example:
        self.robot.velocity_rel = 0.15
        self.robot.acceleration_rel = 0.05
        self.robot.jerk_rel = 0.005
        #self.robot.set_dynamic_rel(0.2)

        #print("FIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEEEE")
        #print(gripper_width)
        #print(gripper_speed)
        #print("Robot connected")

        state = self.robot.read_once()
        #print('\nPose: ', self.robot.current_pose())
        #print('O_TT_E: ', state.O_T_EE)

        self.motion = None
        self.motion_thread = None

        self.dt = 1 / 120.0
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.65, 0.2, 0.2])
        self.robot_default_dof_pos = np.array([0, -0.569, 0, -2.810, 0, 3.037, 0.741])
        self.robot_default_gripper_pos = np.array([0.04])
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
        #self.rock_target = np.array([0.5, 0, 0.3, 1, 0, 0, 0])
        self.rock_target = np.array([-0.7009, 0.9867, -1.6942, 0, 0, 0, 0])

        self.progress_buf = 1
        self.obs_buf = np.zeros((36,), dtype=np.float32)

    def _update_target_from_camera(self):
        pixel_to_meter = 1.11 / 375  # m/px: adjust for custom cases

        import cv2
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # convert to HSV and remove noise
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 15)

            # color matching in HSV
            mask = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))
            M = cv2.moments(mask)
            if M["m00"]:
                x = M["m10"] / M["m00"]
                y = M["m01"] / M["m00"]

                # real-world position (fixed z to 0.2 meters)
                pos = np.array([pixel_to_meter * (y - 185), pixel_to_meter * (x - 320), 0.2])
                if self is not None:
                    self.target_pos = pos

                # draw target
                frame = cv2.circle(frame, (int(x), int(y)), 30, (0,0,255), 2)
                frame = cv2.putText(frame, str(np.round(pos, 4).tolist()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # show images
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cap.release()

    def _get_observation_reward_done(self):
        # get robot state
        try:
            robot_state = self.robot.get_state(read_once=True)
        except frankx.InvalidOperationException:
            robot_state = self.robot.get_state(read_once=False)

        # observations

        #actions in observation
        self.last_action = np.zeros(8, dtype=np.float32)

        #gripper instantiation
        self.gripper = self.robot.get_gripper()
        gripper_width = self.gripper.width()
        gripper_speed = self.gripper.__getattribute__("gripper_speed")
        self.gripper.open()

        #Joint positions in observation
        self.joint_pos = np.zeros((9,))
        self.joint_pos[0:7] = np.array(robot_state.q)
        self.joint_pos[7:8] = gripper_width/2
        self.joint_pos[8:9] = gripper_width/2

        #Joint velocities in observation
        self.joint_vel = np.zeros((9,))
        self.joint_vel[0:7] = np.array(robot_state.dq)
        self.joint_vel[7:8] = gripper_speed
        self.joint_vel[8:9] = gripper_speed

        #Rock position
        self.object_position = self.target_pos

        #Target position
        self.target_object_position = self.rock_target

        #Observation creation
        self.obs_buf[0:8] =   self.last_action
        self.obs_buf[8:17] =  self.joint_pos
        self.obs_buf[17:26] = self.joint_vel
        self.obs_buf[26:29] = self.object_position
        self.obs_buf[29:36] = self.target_object_position 


        # reward
        self.end_effector_pos = np.array(robot_state.O_T_EE[-4:-1])
        distance = np.linalg.norm(self.end_effector_pos - self.target_pos)
        reward = -distance

        # done
        done = self.progress_buf >= self.max_episode_length - 1
        done = done or distance <= 0.075

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    def reset(self):
        print("Resetting...")

        # end current motion
        if self.motion is not None:
            self.motion.finish()
            self.motion_thread.join()
        self.motion = None
        self.motion_thread = None

        # open/close gripper
        #self.gripper.open()
        # self.gripper.clamp()

        # go to 1) safe position, 2) random position
        self.robot.move(frankx.JointMotion(self.robot_default_dof_pos.tolist()))
        dof_pos = self.robot_default_dof_pos # + 0.25 * (np.random.rand(7) - 0.5) #start position offset
        self.robot.move(frankx.JointMotion(dof_pos.tolist()))
        #self.gripper.move(0.04)
        # get target position from prompt
        if not self.camera_tracking:
            while True:
                try:
                    print("Enter target position (X, Y, Z) in meters")
                    raw = input("or press [Enter] key for a random target position: ")
                    if raw:
                        self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                    else:
                        #self.target_pos = np.array([0.1, -0.1, 0.02])
                        self.target_pos = np.array([0.09, -1.33, -1.22])
                    print("Target position:", self.target_pos)
                    break
                except ValueError:
                    print("Invalid input. Try something like: 0.65, 0.0, 0.2")

        # initial pose with gripper
        affine = frankx.Affine(frankx.Kinematics.forward(dof_pos.tolist()))
        affine = affine * frankx.Affine(x=0, y=0, z=-0.10335, a=np.pi/2)
        
        # motion type
        if self.motion_type == "waypoint":
            self.motion = frankx.WaypointMotion([frankx.Waypoint(affine)], return_when_finished=False)
        elif self.motion_type == "impedance":
            self.motion = frankx.ImpedanceMotion(500, 50)
        else:
            raise ValueError("Invalid motion type:", self.motion_type)

        self.motion_thread = self.robot.move_async(self.motion)
        if self.motion_type == "impedance":
            self.motion.target = affine

        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        if self._drepecated_api:
            return observation
        else:
            return observation, {}

    def step(self, action):
        self.last_action = action
        self.progress_buf += 1
        # control space
        # joint
        if self.control_space == "blind_agent":
            #print("går jeg herind")
            # get robot state
            try:
                robot_state = self.robot.get_state(read_once=True)
            except frankx.InvalidOperationException:
                robot_state = self.robot.get_state(read_once=False)
            # forward kinematics

            dof_pos = np.array(robot_state.q) + (self.robot_dof_speed_scales * self.dt * action[0:7] * self.action_scale) # fikser at der ikke er gripper med i øjeblikket... måske
            #dof_pos = np.array(robot_state.q) + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
            affine = frankx.Affine(self.robot.forward_kinematics(dof_pos.flatten().tolist()))
            affine = affine * frankx.Affine(x=0, y=0, z=-0.10335, a=np.pi/2)
            #print("går jeg herind")
        # cartesian
        elif self.control_space == "cartesian":
            #print("cartesian")
            action /= 100.0
            if self.motion_type == "waypoint":
                #print("Cartesian: waypoint")
                affine = frankx.Affine(x=action[0], y=action[1], z=action[2])
            elif self.motion_type == "impedance":
                #print("Cartesian: impedance")
                # get robot pose
                try:
                    robot_pose = self.robot.current_pose(read_once=True)
                except frankx.InvalidOperationException:
                    robot_pose = self.robot.current_pose(read_once=False)
                affine = robot_pose * frankx.Affine(x=action[0], y=action[1], z=action[2])
        
        # motion type
        # waypoint motion
        if self.motion_type == "waypoint":
            #print("går jeg herind")
            if self.control_space == "blind_agent":
                #print("går jeg herind")
                self.motion.set_next_waypoint(frankx.Waypoint(affine))
            elif self.control_space == "cartesian":
                #print("waypoint: cartesian")
                self.motion.set_next_waypoint(frankx.Waypoint(affine, frankx.Waypoint.Relative))
        # impedance motion
        elif self.motion_type == "impedance":
            #print("impedance")
            self.motion.target = affine
        else:
            raise ValueError("Invalid motion type:", self.motion_type)

        # the use of time.sleep is for simplicity. This does not guarantee control at a specific frequency
        time.sleep(0.1)  # lower frequency, at 30Hz there are discontinuities

        observation, reward, done = self._get_observation_reward_done()

        print("DEBUGGING")
        print("action is: ", action)
        print("Affine is: ", affine)
        print("EE pose is: ", self.end_effector_pos)
        obs_array_type = ["joint action", "gripper close width", "joint position", "gripper position", "joint velocity", "gripper velocity", "stone x", "stone y", "stone z", "Target x", "Target y", "Target z", "Target quat"]
        print("oberservations is: ")
        for i in range(len(observation)):
            if i<=6: print(obs_array_type[0], i+1, "is: ", observation[i])
            if 6<i<8: print(obs_array_type[1], "is: ", observation[i])
            if 7<i<=14: print(obs_array_type[2], i-7, "is: ", observation[i])
            if 14<i<=16: print(obs_array_type[3], i-14, "is: ", observation[i])
            if 16<i<=23: print(obs_array_type[4], i-16, "is: ", observation[i])
            if 23<i<=25: print(obs_array_type[5], i-23, "is: ", observation[i])
            if 25<i<=28: print(obs_array_type[5+i-25], "is: ", observation[i])
            if 28<i<=31: print(obs_array_type[8+i-28], "is: ", observation[i])
            if 31<i<=35: print(obs_array_type[12], i-31, "is: ", observation[i])


        if self._drepecated_api:
            return observation, reward, done, {}
        else:
            return observation, reward, done, done, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass
