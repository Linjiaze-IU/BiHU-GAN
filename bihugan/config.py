import os
import yaml
from pathlib import Path

def load_config(yaml_path: str) -> dict:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    required = [
        "train_list", "val_list", "pair_from", "pair_to",
        "hu_min", "hu_max", "model_input_resolution", "epochs",
        "lr", "batchSize", "body_hu_thresh", "bone_hu_thresh",
        "global_scaling_factor", "hu_loss_weight",
        "mc_dropout_p_train", "amp", "grad_accum_steps",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    return cfg