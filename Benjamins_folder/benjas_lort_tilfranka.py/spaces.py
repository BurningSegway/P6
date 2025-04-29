from gym.spaces import Box, Dict
import numpy as np

observation_space =  Dict({"actions": Box(low=-1000, high=1000, shape=(8,), dtype=np.float32), 
                               "joint_pos": Box(low=-1000, high=1000, shape=(9,), dtype=np.float32),
                                "joint_vel": Box(low=-1000, high=1000, shape=(9,), dtype=np.float32),
                                "object_position": Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
                                "target_object_position": Box(low=-1000, high=1000, shape=(7,), dtype=np.float32)})


print(observation_space)
#self.obs_buf = np.zeros((18,), dtype=np.float32)

#obs_buf[0] = self.progress_buf / float(self.max_episode_length)
#obs_buf[1:8] = dof_pos_scaled
#obs_buf[8:15] = dof_vel_scaled
#obs_buf[15:18] = self.target_pos


obs_buf = np.zeros((36,), dtype=np.float32)

obs_buf[0:8] =   actions
obs_buf[8:17] =  joint_pos
obs_buf[17:26] = joint_vel
obs_buf[26:29] = object_position
obs_buf[29:36] = target_object_position

