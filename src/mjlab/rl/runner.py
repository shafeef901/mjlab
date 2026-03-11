import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  # Map RslRlModelCfg.class_name -> rsl_rl ActorCritic class name.
  _MODEL_CLASS_MAP = {
    "MLPModel": "ActorCritic",
    "CNNModel": "ActorCriticCNN",
  }

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Convert legacy actor/critic config to the flat policy dict expected
    # by rsl_rl's OnPolicyRunner.
    if "actor" in train_cfg and "critic" in train_cfg and "policy" not in train_cfg:
      actor = train_cfg.pop("actor")
      critic = train_cfg.pop("critic")

      policy: dict = {
        "actor_hidden_dims": actor["hidden_dims"],
        "critic_hidden_dims": critic["hidden_dims"],
        "actor_obs_normalization": actor.get("obs_normalization", False),
        "critic_obs_normalization": critic.get("obs_normalization", False),
        "activation": actor.get("activation", "elu"),
      }

      # Map model class name (e.g. MLPModel -> ActorCritic).
      model_cls = actor.get("class_name", "MLPModel")
      policy["class_name"] = self._MODEL_CLASS_MAP.get(model_cls, model_cls)

      # Extract noise parameters from actor distribution_cfg.
      dist_cfg = actor.get("distribution_cfg")
      if dist_cfg:
        if "init_std" in dist_cfg:
          policy["init_noise_std"] = dist_cfg["init_std"]
        if "std_type" in dist_cfg:
          policy["noise_std_type"] = dist_cfg["std_type"]

      # Forward CNN config if present.
      cnn_cfg = actor.get("cnn_cfg")
      if cnn_cfg is not None:
        policy["cnn_cfg"] = cnn_cfg

      train_cfg["policy"] = policy

    # Rename obs_groups "actor" -> "policy" to match rsl_rl convention.
    obs_groups = train_cfg.get("obs_groups", {})
    if "actor" in obs_groups and "policy" not in obs_groups:
      obs_groups["policy"] = obs_groups.pop("actor")

    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = {
      "model_state_dict": self.alg.policy.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos,
    }
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_optimizer: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to restore common_step_counter to
    preserve curricula state.
    """
    infos = super().load(path, load_optimizer, map_location)

    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
