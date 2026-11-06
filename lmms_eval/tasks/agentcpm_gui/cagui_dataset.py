"""
CAGUI Dataset Loading Script

This script provides a custom HuggingFace datasets compatible loader
for the CAGUI (Chinese Android GUI) dataset.

Usage:
    1. Set LOCAL_CAGUI_PATH environment variable to your CAGUI directory
    2. Or modify the DATA_DIR constant below
    
Expected directory structure:
    CAGUI/
    └── CAGUI_agent/
        └── domestic/
            └── {episode_id}/
                ├── {episode_id}.json
                ├── {episode_id}_0.jpeg
                └── ...
"""

import json
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional

import datasets
from PIL import Image

# Default data directory - can be overridden by environment variable
DATA_DIR = os.environ.get("LOCAL_CAGUI_PATH", "/path/to/CAGUI")


class CAGUIConfig(datasets.BuilderConfig):
    """BuilderConfig for CAGUI dataset."""
    
    def __init__(
        self,
        subset: str = "domestic",
        split_name: str = "test",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.split_name = split_name


class CAGUIDataset(datasets.GeneratorBasedBuilder):
    """CAGUI: Chinese Android GUI Benchmark Dataset."""
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        CAGUIConfig(
            name="chinese_app_test",
            subset="domestic",
            split_name="test",
            description="Chinese App Test (Domestic) subset"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "chinese_app_test"
    
    def _info(self):
        return datasets.DatasetInfo(
            description="CAGUI: Chinese Android GUI Benchmark",
            features=datasets.Features({
                "episode_id": datasets.Value("string"),
                "step_id": datasets.Value("int32"),
                "instruction": datasets.Value("string"),
                "image": datasets.Image(),
                "image_path": datasets.Value("string"),
                "image_width": datasets.Value("int32"),
                "image_height": datasets.Value("int32"),
                "subset": datasets.Value("string"),
                "result_action_type": datasets.Value("int32"),
                "result_action_text": datasets.Value("string"),
                "result_touch_yx": datasets.Value("string"),
                "result_lift_yx": datasets.Value("string"),
                "ui_positions": datasets.Value("string"),
                "duration": datasets.Value("float32"),
                "episode_length": datasets.Value("int32"),
            }),
            homepage="https://huggingface.co/datasets/openbmb/CAGUI",
            license="CC-BY-NC-4.0",
        )
    
    def _split_generators(self, dl_manager):
        data_dir = Path(DATA_DIR)
        subset = self.config.subset
        
        # Check for CAGUI_agent subdirectory
        agent_dir = data_dir / "CAGUI_agent" / subset
        if not agent_dir.exists():
            # Try without CAGUI_agent prefix
            agent_dir = data_dir / subset
        
        if not agent_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {agent_dir}\n"
                f"Please set LOCAL_CAGUI_PATH environment variable to your CAGUI directory"
            )
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": str(agent_dir)},
            ),
        ]
    
    def _generate_examples(self, data_dir: str) -> Generator:
        """Generate examples from CAGUI directory structure."""
        data_path = Path(data_dir)
        
        # Get all episode directories
        episode_dirs = sorted([
            d for d in data_path.iterdir()
            if d.is_dir()
        ])
        
        idx = 0
        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            json_path = episode_dir / f"{episode_id}.json"
            
            if not json_path.exists():
                continue
            
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    episode_data = json.load(f)
                
                for step in episode_data:
                    # Get image path
                    image_rel_path = step.get("image_path", "")
                    
                    # Try to find the image
                    image_path = None
                    if image_rel_path:
                        # Try relative path from episode dir
                        img_name = os.path.basename(image_rel_path)
                        possible_paths = [
                            episode_dir / img_name,
                            data_path.parent / image_rel_path,
                            data_path / image_rel_path,
                        ]
                        for p in possible_paths:
                            if p.exists():
                                image_path = str(p)
                                break
                    
                    yield idx, {
                        "episode_id": step.get("episode_id", episode_id),
                        "step_id": step.get("step_id", 0),
                        "instruction": step.get("instruction", ""),
                        "image": image_path,
                        "image_path": image_rel_path,
                        "image_width": step.get("image_width", 0),
                        "image_height": step.get("image_height", 0),
                        "subset": step.get("subset", self.config.subset),
                        "result_action_type": step.get("result_action_type", -1),
                        "result_action_text": step.get("result_action_text", ""),
                        "result_touch_yx": step.get("result_touch_yx", "[-1, -1]"),
                        "result_lift_yx": step.get("result_lift_yx", "[-1, -1]"),
                        "ui_positions": step.get("ui_positions", "[]"),
                        "duration": step.get("duration") if step.get("duration") is not None else 0.0,
                        "episode_length": step.get("episode_length", 0),
                    }
                    idx += 1
                    
            except Exception as e:
                print(f"Error loading episode {episode_id}: {e}")
                continue


def load_cagui_dataset(
    data_dir: Optional[str] = None,
    subset: str = "domestic",
) -> datasets.Dataset:
    """
    Convenience function to load CAGUI dataset.
    
    Args:
        data_dir: Path to CAGUI directory. If None, uses LOCAL_CAGUI_PATH env var.
        subset: Dataset subset (e.g., "domestic")
    
    Returns:
        HuggingFace Dataset object
    """
    if data_dir:
        os.environ["LOCAL_CAGUI_PATH"] = data_dir
    
    # Load using the custom builder
    builder = CAGUIDataset(name="chinese_app_test")
    builder.download_and_prepare()
    return builder.as_dataset()["test"]
