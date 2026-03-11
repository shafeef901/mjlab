"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Config for a single neural network model (Actor or Critic)."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the network."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize the observations. Default is False."""
  cnn_cfg: dict[str, Any] | None = None
  """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
  distribution_cfg: dict[str, Any] | None = None
  """Distribution config dict passed to rsl_rl. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  """
  class_name: str = "MLPModel"
  """Model class name resolved by RSL-RL (MLPModel or CNNModel)."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  class_name: str = "PPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"policy": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under
  ``logs/rsl_rl/{experiment_name}/``."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  upload_model: bool = True
  """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """The actor configuration."""
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The critic configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""


@dataclass
class RslRlReppoActorQCfg:
  """Config for the REPPO ActorQ network."""

  class_name: str = "ActorQ"
  """Ignore, required by RSL-RL."""
  init_noise_std: float = 1.0
  """The initial noise standard deviation of the policy."""
  init_alpha_kl: float = 0.01
  """The initial KL divergence coefficient (adaptive)."""
  init_alpha_temp: float = 0.01
  """The initial temperature coefficient (adaptive)."""
  actor_obs_normalization: bool = False
  """Whether to normalize the observation for the actor network."""
  critic_obs_normalization: bool = False
  """Whether to normalize the observation for the critic network."""
  actor_hidden_dims: Tuple[int, ...] = (512, 512, 512)
  """The hidden dimensions of the actor network."""
  critic_hidden_dims: Tuple[int, ...] = (512, 512, 512)
  """The hidden dimensions of the critic network."""
  activation: str = "elu"
  """The activation function for the actor and critic networks."""
  noise_std_type: Literal["scalar", "log", "sigmoid"] = "scalar"
  """The type of noise standard deviation for the policy."""
  vmin: float = -20.0
  """The minimum value for the value distribution."""
  vmax: float = 20.0
  """The maximum value for the value distribution."""


@dataclass
class RslRlReppoAlgorithmCfg:
  """Config for the REPPO algorithm."""

  class_name: str = "REPPO"
  """Ignore, required by RSL-RL."""
  num_learning_epochs: int = 4
  """The number of learning epochs per update."""
  num_mini_batches: int = 8
  """The number of mini-batches per update."""
  learning_rate: float = 3e-4
  """The learning rate."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  desired_kl: float = 0.1
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  target_entropy: float = -0.5
  """The target entropy for the policy."""
  scale_actions: bool = False
  """Whether to scale actions to the environment's action bounds."""
  action_upper_bound: float = 1.0
  """The upper bound for action scaling."""
  action_lower_bound: float = -1.0
  """The lower bound for action scaling."""


@dataclass
class RslRlReppoRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration for the REPPO on-policy runner."""

  class_name: str = "OnPolicyRunner"
  """The runner class name. Same OnPolicyRunner as PPO; REPPO is selected via policy/algorithm class_name."""
  policy: RslRlReppoActorQCfg = field(default_factory=RslRlReppoActorQCfg)
  """The ActorQ policy configuration."""
  algorithm: RslRlReppoAlgorithmCfg = field(default_factory=RslRlReppoAlgorithmCfg)
  """The REPPO algorithm configuration."""
