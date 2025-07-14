# #!/usr/bin/env bash

# set -x

# FILE=$1
# CONFIG=$2
# GPUS=$3
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-$((28500 + $RANDOM % 2000))}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}


# if command -v torchrun &> /dev/null
# then
#   echo "Using torchrun mode."
#   PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     torchrun --nnodes=${NNODES} \
#     --nnodes=${NNODES} \
#     --node_rank=${NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${PORT} \
#     --nproc_per_node=${GPUS} \
#     tools/${FILE}.py ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
# else
#   echo "Using launch mode."
#   PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     python -m torch.distributed.launch \
#     --nnodes=${NNODES} \
#     --node_rank=${NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${PORT} \
#     --nproc_per_node=${GPUS} \
#     tools/${FILE}.py ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
# fi


#######################################################################################

#!/usr/bin/env bash
# =================================================================================
# 单进程／单 GPU 脚本：不再使用 torchrun / torch.distributed.launch / deepspeed
# 调用方式举例：
#   CUDA_VISIBLE_DEVICES=7 bash tools/dist.sh train \
#       projects/llava_sam2/configs/sa2va_4b_relation.py \
#       --work-dir debug4
# =================================================================================

set -e
set -x

FILE=$1           # train / test / infer 等脚本名
CONFIG=$2         # config 文件路径
shift 2           # 去掉前两个参数，剩余全当成 python 脚本参数传递

# 由用户在外部通过 CUDA_VISIBLE_DEVICES 指定用哪个卡
# 例如：CUDA_VISIBLE_DEVICES=7

# 确保能 import 到项目根目录下的模块
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 执行单进程脚本，不启用任何分布式、deepspeed、launcher
python3.10 -u tools/${FILE}.py ${CONFIG} "$@"