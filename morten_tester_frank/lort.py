import time
import torch
import torch.nn as nn
import numpy as np
import gym

# SKRL components
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env

# frankx environment
from reaching_franka_real_env import ReachingFranka

# 1) define policy network (same as during training in Isaac Sim)
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # inputs["states"] is the flattened observation
        return self.net(inputs["states"]), self.log_std_parameter, {}

# 2) create and wrap environment
control_space = "joint"  # or "joint"
motion_type = "waypoint"
camera_tracking = False

env = ReachingFranka(robot_ip="192.168.2.30",
                     device="cpu",
                     control_space=control_space,
                     motion_type=motion_type,
                     camera_tracking=camera_tracking)
env = wrap_env(env)
device = env.device

# 3) build SKRL agent with our Policy
models_ppo = {"policy": Policy(env.observation_space, env.action_space, device)}

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 0

agent = PPO(models=models_ppo,
            memory=None,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


agent.load("./best_agent.pt")

# 5) evaluate with SequentialTrainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.eval()  # runs one episode and prints rewards
