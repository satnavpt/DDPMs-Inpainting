import io
import os

import blobfile as bf
import torch as torch
import torch.distributed as dist


class DistributedHelper:
    def __init__(self):
        try:
            assert dist.is_available() and torch.cuda.is_available()

            world_size = int(
                os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
            )
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
            dist.init_process_group(
                "nccl", init_method="env://", world_size=world_size, rank=rank
            )

            local_world_size = (
                int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
                or int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))
                or torch.cuda.device_count()
            )

            local_rank = (
                int(os.environ.get("LOCAL_RANK", "0")) or rank % local_world_size
            )
            num_devices = world_size or local_world_size

            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))

            distributed = True

            self.world_size = world_size
            self.local_world_size = local_world_size

        except Exception as e:
            print(f"Not distributed!: {e}")
            distributed = False
            rank = local_rank = 0
            num_devices = 0
            self.world_size = 1
            self.local_world_size = 1

        self.is_leader = rank == 0
        self.distributed = distributed
        self.rank = rank
        self.local_rank = local_rank
        self.num_devices = num_devices

    def dev(self):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def num_dev(self):
        return self.num_devices

    def sync_params(self, params):
        if self.distributed:
            for p in params:
                with torch.no_grad():
                    dist.broadcast(p, 0)

    def load_state_dict(self, path, **kwargs):
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data), **kwargs)
