import logging
from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils

from omegaconf import MISSING, DictConfig

from src.tasks.video.enhance.video_enhance import VideoEnhanceTask


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.structured_configs import register_structured_configs
from src.utils import print_config_tree

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def enhance(cfg: DictConfig):
    video_folder = cfg.video_folder
    output_folder = cfg.output_folder
    if video_folder in [MISSING, None]:
        raise ValueError(f"video_folder is not set, got: {video_folder}")
    if output_folder in [MISSING, None]:
        raise ValueError(f"output_folder is not set, got: {output_folder}")
    processors = []
    for processor_name, processor_cfg in cfg.pipeline.processor.items():
        log.info(
            f"Instantiating processor <{processor_name}> <{processor_cfg._target_}>"
        )
        processor = hydra.utils.instantiate(processor_cfg)
        processors.append(processor)
    task = VideoEnhanceTask(
        processors=processors,
        video_folder=video_folder,
        output_folder=output_folder,
        writer_kwargs=cfg.get("writer_kwargs", {}),
        reader_kwargs=cfg.get("reader_kwargs", {}),
    )
    task.enhance()
    log.info("Enhance task")


@hydra.main(version_base="1.3", config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig):
    print_config_tree(cfg)

    task_name = cfg.task_name
    if task_name == "enhance":
        log.info("Enhance task")
        enhance(cfg)


if __name__ == "__main__":
    register_structured_configs()
    main()
