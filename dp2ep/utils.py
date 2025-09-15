from torch.distributed import get_rank

def print0(*args, **kwargs):
    if not get_rank() == 0:
        return
    print(*args, **kwargs)
