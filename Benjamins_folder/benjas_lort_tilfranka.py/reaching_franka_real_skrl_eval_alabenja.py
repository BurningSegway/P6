import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed
from frankx import Robot, Gripper

# seed for reproducibility
set_seed()  # e.g. set_seed(42) for fixed seed

# Define the shared policy/value model
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared = self._shared_output if self._shared_output is not None else self.net(inputs["states"])
            self._shared_output = None
            return self.value_layer(shared), {}

# Load your real-world Franka environment
from reaching_franka_real_env_alabenja import ReachingFranka
control_space = "blind_agent"
motion_type = "waypoint"
camera_tracking = False

env = ReachingFranka(robot_ip="192.168.2.30",
                     device="cpu",
                     control_space=control_space,
                     motion_type=motion_type,
                     camera_tracking=camera_tracking)
# wrap the environment
env = wrap_env(env)
device = env.device

# memory for rollouts
memory = RandomMemory(memory_size=96, num_envs=env.num_envs, device=device)

# build models and agent
models = {"policy": Shared(env.observation_space, env.action_space, device)}
models["value"] = models["policy"]  # shared model
g
cfg = PPO_DEFAULT_CONFIG.copy()
# disable preprocessors to retain raw obs
cfg["state_preprocessor"] = None
cfg["value_preprocessor"] = None
# (other PPO hyperparameters unchanged)

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# load the trained checkpoint
if control_space == "blind_agent":
    agent.load("Benjamins_folder/benjas_lort_tilfranka.py/best_agent.pt")
else:
    raise ValueError("Wrong control space")

# debug: confirm loaded weights
print("Loaded policy mean weights:", agent.model.mean_layer.weight.data)
print("Loaded log_std_parameter:", agent.model.log_std_parameter.data)

# Synthetic test of policy responsiveness
obs = torch.zeros(env.observation_space.shape, device=device)
print("Synthetic-obs policy test:")
for i in range(5):
    obs = obs + 0.1
    action, *_ = agent.model.act({"states": obs.unsqueeze(0)}, role="policy")
    print(f" step {i}, action = {action.cpu().numpy()}")

# Configure and run StepTrainer for real evaluation
step_trainer = StepTrainer(
    env=env,
    agents=[agent],  # must be a list
    cfg={
        "timesteps": 10,
        "headless": True,
        "stochastic_evaluation": True
    }
)
# Run eval (uses cfg["timesteps"] internally)
obs_batch, reward_batch, term_batch, trunc_batch, info = step_trainer.eval()

# print collected data
print("Batched observations shape:", obs_batch.shape)
print("Batched rewards shape:   ", reward_batch.shape)
print("Terminated flags:        ", term_batch)
print("Truncated flags:         ", trunc_batch)
print("Batched actions shape:   ", step_trainer.last_actions.shape)
print(step_trainer.last_actions)
