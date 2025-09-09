# fsdp
# OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 dtensor_parallelisms/main.py fsdp

# tp
# OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 dtensor_parallelisms/main.py tp

# dp2ep
OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 dtensor_parallelisms/dp2ep.py
