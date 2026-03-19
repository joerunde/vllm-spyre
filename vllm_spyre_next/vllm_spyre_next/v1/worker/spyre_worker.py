"""A Torch Spyre worker class."""

from vllm.config import VllmConfig
from vllm.v1.worker.cpu_worker import CPUWorker

from vllm_spyre_next.custom_ops import register_all

import os


class TorchSpyreWorker(CPUWorker):
    """A worker class that executes the model on a group of Spyre cores."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)
        self.init_distributed()

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the layers will be swapped out
        # with the custom implementations for spyre.
        register_all()

    def init_distributed(self) -> None:
        """Setup torch_spyre to access the correct card in parallel setups"""
        # 🌶️🌶️🌶️ torch_spyre currently only supports access to the rank=0 card.
        # This method overwrites rank=0 to a different device address to work around this.
        rank = self.parallel_config.rank
        world_size = int(os.getenv("AIU_WORLD_SIZE", "0"))
        assert world_size >= rank, f"Cannot run rank {rank} with world size {world_size}"

        if rank > 0:        
            device_addr = os.getenv(f"AIU_WORLD_RANK_{rank}")
            assert device_addr is not None, f"No device address found for rank {rank}"

            # Overwrite the rank=0 device address to point to the rank={rank} device instead
            os.environ["AIU_WORLD_RANK_0"] = device_addr
