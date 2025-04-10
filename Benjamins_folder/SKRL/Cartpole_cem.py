import gymnasium as gym

import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define model (categorical model) using mixin
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))   
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x), {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("CartPole-v1")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("CartPole-v")][0]
    print("CartPole-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's model (function approximator).
# CEM requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/cem.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/cem.html#configuration-and-hyperparameters
cfg = CEM_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1000
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "P6\Benjamins_folder\SKRL/runs/torch/CartPole"
cfg["experiment"]["wandb"] = True
#herunder er den komando der gemmer vores skidt på W&B, og det kan Pierre godt lide...
cfg["experiment"]["wandb_kwargs"] ={
    "entity": "urkanin-aalborg-universitet",
    "project": "Legeland"
}

agent = CEM(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
