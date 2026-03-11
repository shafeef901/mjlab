"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlReppoRunnerCfg,
  RslRlReppoActorQCfg,
  RslRlReppoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_velocity_reppo_runner_cfg() -> RslRlReppoRunnerCfg:
  """Create REPPO runner configuration for Unitree G1 velocity task."""
  return RslRlReppoRunnerCfg(
    experiment_name="g1_velocity_reppo",
    num_steps_per_env=32,
    max_iterations=100_000,
    policy=RslRlReppoActorQCfg(
      init_noise_std=0.1,
      init_alpha_kl=0.001,
      init_alpha_temp=0.0001,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 512, 512),
      critic_hidden_dims=(1024, 1024, 1024),
      activation="elu",
      noise_std_type="sigmoid",
      vmin=-20.0,
      vmax=20.0,
    ),
    algorithm=RslRlReppoAlgorithmCfg(
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      gamma=0.99,
      lam=0.85,
      desired_kl=0.1,
      max_grad_norm=1.0,
      target_entropy=-0.5,
      scale_actions=True,
      action_upper_bound=1.0,
      action_lower_bound=-1.0,
    ),
  )
