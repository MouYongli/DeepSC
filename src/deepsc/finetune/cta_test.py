import hydra
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from deepsc.finetune.cell_type_annotation import CellTypeAnnotation
from deepsc.models.deepsc_new.model import DeepSCClassifier
from deepsc.utils.utils import setup_logging


@hydra.main(
    version_base=None, config_path="../../../configs/finetune", config_name="finetune"
)
def cta_test(cfg: DictConfig):
    """
    Cell Type Annotation Test Script

    This script loads a trained checkpoint and evaluates it on the test dataset.
    Results are saved to a timestamped directory including:
    - Test metrics (accuracy, precision, recall, F1)
    - Confusion matrix plot
    - Classification report CSV

    Configuration:
    - Uses the same config structure as finetune.py (configs/finetune/finetune.yaml)
    - Test-specific parameters are defined in configs/finetune/tasks/cell_type_annotation.yaml
    - Requires: checkpoint_path, test_save_dir in the task config
    """
    # Merge task-specific configs to top level (same as finetune.py)
    if "tasks" in cfg:
        # Disable struct mode to allow merging new keys
        OmegaConf.set_struct(cfg, False)
        task_cfg = OmegaConf.to_container(cfg.tasks, resolve=True)
        cfg = OmegaConf.merge(cfg, task_cfg)
        # Remove the tasks key to avoid confusion
        if "tasks" in cfg:
            del cfg["tasks"]
        # Re-enable struct mode to prevent accidental key additions
        OmegaConf.set_struct(cfg, True)

    # Initialize fabric
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        num_nodes=cfg.num_nodes,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=False,
        ),
        precision="bf16-mixed",
    )
    fabric.launch()

    # Initialize log
    setup_logging(rank=fabric.global_rank, log_path="./logs")

    # Get task type from config
    task_type = getattr(cfg, "task_type", "cell_type_annotation")
    print(f"Running test for task: {task_type}")

    if task_type == "cell_type_annotation":
        # IMPORTANT: Load cell_type_count from checkpoint first
        # to ensure model structure matches the trained model
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided in config for testing")

        if fabric.global_rank == 0:
            print(
                f"\n[INFO] Pre-loading checkpoint to get cell_type_count: {checkpoint_path}"
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            actual_cell_type_count = checkpoint["cell_type_count"]
            print(f"[INFO] Cell type count from checkpoint: {actual_cell_type_count}")
        else:
            checkpoint = None
            actual_cell_type_count = None

        # Broadcast to all processes
        checkpoint = fabric.broadcast(checkpoint, src=0)
        if checkpoint is not None:
            actual_cell_type_count = checkpoint["cell_type_count"]

        # Now instantiate model with correct cell_type_count from checkpoint
        model: nn.Module = hydra.utils.instantiate(cfg.model)
        encoder = model.float()
        model = DeepSCClassifier(
            deepsc_encoder=encoder,
            n_cls=actual_cell_type_count,  # Use count from checkpoint!
            num_layers_cls=3,
            cell_emb_style="avg-pool",
        )
        model = model.float()

        # Create tester instance
        # Note: We don't call build_dataset_sampler_from_h5ad in __init__ for testing
        # Instead, we'll load the checkpoint first, then build test dataset
        tester = CellTypeAnnotation.__new__(CellTypeAnnotation)
        tester.args = cfg
        tester.fabric = fabric
        tester.model = model
        tester.world_size = fabric.world_size
        tester.is_master = fabric.global_rank == 0
        tester.epoch = 0

        # Set random seed
        from deepsc.utils import seed_all

        seed_all(cfg.seed + fabric.global_rank)

        # Set cell_type_count from pre-loaded checkpoint
        # (needed for init_loss_fn if using per_bin loss)
        tester.cell_type_count = actual_cell_type_count

        # Setup model with fabric before loading checkpoint
        tester.model = fabric.setup(tester.model)

        # Initialize loss function (now safe because cell_type_count is set)
        tester.init_loss_fn()

        # Run test (will call load_checkpoint internally, which will update cell_type_count again)
        print("\nStarting test...")
        test_loss, test_error = tester.test()

        if fabric.global_rank == 0:
            print("\nTest completed successfully!")
            print(f"Final test loss: {test_loss:.4f}")
            print(f"Final test error rate: {test_error:.4f}")
    else:
        raise ValueError(
            f"Unknown task type: {task_type}. Supported types: ['cell_type_annotation']"
        )


if __name__ == "__main__":
    cta_test()
