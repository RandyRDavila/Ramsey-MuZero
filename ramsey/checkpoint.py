# ramsey/checkpoint.py
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import torch


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Return the newest checkpoint file under ckpt_dir, or None.
    Accepts common extensions (.pt, .pth) and any file in the dir.
    """
    if not ckpt_dir.exists():
        return None
    candidates = []
    for p in ckpt_dir.glob("*"):
        if p.is_file() and (p.suffix in {".pt", ".pth", ""} or True):
            candidates.append(p)
    if not candidates:
        return None
    # newest by mtime
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model_state_safely(model: torch.nn.Module, ckpt_path: Path, map_location: str = "cpu") -> Tuple[bool, str]:
    """
    Try a few common checkpoint formats:
      - torch.save(model.state_dict())
      - torch.save({"model": state_dict})
      - torch.save({"state_dict": state_dict})
    """
    try:
        obj = torch.load(str(ckpt_path), map_location=map_location)
        if isinstance(obj, dict):
            if "model" in obj and isinstance(obj["model"], dict):
                model.load_state_dict(obj["model"], strict=False)
                return True, "Loaded 'model' key"
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                model.load_state_dict(obj["state_dict"], strict=False)
                return True, "Loaded 'state_dict' key"
            # Might actually be a raw state_dict stored in a weird dict
            try:
                model.load_state_dict(obj, strict=False)
                return True, "Loaded dict as state_dict"
            except Exception as e:
                return False, f"Unsupported dict format: {e}"
        elif isinstance(obj, (list, tuple)):
            return False, "Unsupported list/tuple checkpoint"
        else:
            # Raw state_dict
            model.load_state_dict(obj, strict=False)
            return True, "Loaded raw state_dict"
    except Exception as e:
        return False, f"Exception while loading: {e}"
