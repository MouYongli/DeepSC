#!/bin/bash

# 数据预处理脚本运行器 - 支持GPU加速
# 使用方法: ./run_preprocessing.sh [options]

set -e  # 遇到错误时立即退出

# 默认参数
INPUT_DIR="/home/angli/baseline/DeepSC/data/cellxgene/original_pth/pancreas"
OUTPUT_DIR="/home/angli/baseline/DeepSC/data/cellxgene/0717_normalized/pancreas"
GENE_MAP_PATH="/home/angli/baseline/DeepSC/data/gene_map.csv"
NUM_PROCESSES=4
USE_GPU=true
MIN_GENES=200
NORMALIZE=true

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印帮助信息
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input_dir PATH        输入目录路径 (必需)"
    echo "  -o, --output_dir PATH       输出目录路径 (必需)"
    echo "  -g, --gene_map_path PATH    基因映射文件路径 (默认: $GENE_MAP_PATH)"
    echo "  -p, --num_processes NUM     并行进程数 (默认: $NUM_PROCESSES)"
    echo "  --no_gpu                    不使用GPU加速"
    echo "  --no_normalize              跳过标准化步骤"
    echo "  --min_genes NUM             每个细胞最少基因数量 (默认: $MIN_GENES)"
    echo "  -h, --help                  显示帮助信息"
    echo ""
    echo "Example:"
    echo "  $0 -i /path/to/input -o /path/to/output --use_gpu"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gene_map_path)
            GENE_MAP_PATH="$2"
            shift 2
            ;;
        -p|--num_processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --no_gpu)
            USE_GPU=false
            shift
            ;;
        --no_normalize)
            NORMALIZE=false
            shift
            ;;
        --min_genes)
            MIN_GENES="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo -e "${RED}错误: 输入目录和输出目录是必需的${NC}"
    print_help
    exit 1
fi

# 检查输入目录是否存在
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/integrated_preprocess.py"

# 检查Python脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo -e "${RED}错误: Python脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 检查GPU可用性
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}检测到NVIDIA GPU:${NC}"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        return 0
    else
        echo -e "${YELLOW}警告: 未检测到NVIDIA GPU${NC}"
        return 1
    fi
}

# 激活conda环境
activate_conda() {
    echo -e "${GREEN}正在激活conda环境 'deepsc'...${NC}"

    # 初始化conda
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo -e "${YELLOW}警告: 未找到conda安装，尝试使用系统Python${NC}"
        return 1
    fi

    # 激活环境
    if conda activate deepsc 2>/dev/null; then
        echo -e "${GREEN}成功激活conda环境 'deepsc'${NC}"
        return 0
    else
        echo -e "${YELLOW}警告: 无法激活conda环境 'deepsc'，使用当前环境${NC}"
        return 1
    fi
}

# 设置GPU环境变量
setup_gpu_env() {
    if [[ "$USE_GPU" == "true" ]]; then
        echo -e "${GREEN}配置GPU环境...${NC}"

        # 设置CUDA可见设备 (默认使用第一个GPU)
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

        # 设置CuPy内存池
        export CUPY_CACHE_DIR="/tmp/cupy_cache_$$"
        mkdir -p "$CUPY_CACHE_DIR"

        # 设置GPU内存增长策略
        export TF_FORCE_GPU_ALLOW_GROWTH=true

        echo "使用GPU设备: $CUDA_VISIBLE_DEVICES"

        # 检查GPU状态
        if ! check_gpu; then
            echo -e "${YELLOW}警告: GPU检查失败，将使用CPU模式${NC}"
            USE_GPU=false
        fi
    else
        echo -e "${YELLOW}使用CPU模式${NC}"
    fi
}

# 清理函数
cleanup() {
    echo -e "${GREEN}正在清理...${NC}"
    if [[ -n "$CUPY_CACHE_DIR" && -d "$CUPY_CACHE_DIR" ]]; then
        rm -rf "$CUPY_CACHE_DIR"
    fi
}

# 设置退出时清理
trap cleanup EXIT

# 主函数
main() {
    echo -e "${GREEN}=== 数据预处理脚本启动 ===${NC}"
    echo "输入目录: $INPUT_DIR"
    echo "输出目录: $OUTPUT_DIR"
    echo "基因映射文件: $GENE_MAP_PATH"
    echo "并行进程数: $NUM_PROCESSES"
    echo "使用GPU: $USE_GPU"
    echo "标准化: $NORMALIZE"
    echo "最少基因数: $MIN_GENES"
    echo ""

    # 激活conda环境
    activate_conda

    # 设置GPU环境
    setup_gpu_env

    # 构建Python命令
    PYTHON_CMD="python \"$PYTHON_SCRIPT\""
    PYTHON_CMD="$PYTHON_CMD --input_dir \"$INPUT_DIR\""
    PYTHON_CMD="$PYTHON_CMD --output_dir \"$OUTPUT_DIR\""
    PYTHON_CMD="$PYTHON_CMD --gene_map_path \"$GENE_MAP_PATH\""
    PYTHON_CMD="$PYTHON_CMD --num_processes $NUM_PROCESSES"
    PYTHON_CMD="$PYTHON_CMD --min_genes $MIN_GENES"

    if [[ "$USE_GPU" == "true" ]]; then
        PYTHON_CMD="$PYTHON_CMD --use_gpu"
    fi

    if [[ "$NORMALIZE" == "false" ]]; then
        PYTHON_CMD="$PYTHON_CMD --no_normalize"
    fi

    echo -e "${GREEN}执行命令:${NC}"
    echo "$PYTHON_CMD"
    echo ""

    # 记录开始时间
    START_TIME=$(date +%s)

    # 运行Python脚本
    echo -e "${GREEN}开始处理...${NC}"
    if eval "$PYTHON_CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo -e "${GREEN}=== 处理完成 ===${NC}"
        echo -e "${GREEN}总耗时: ${DURATION}秒${NC}"
        echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
    else
        echo -e "${RED}处理失败！${NC}"
        exit 1
    fi
}

# 运行主函数
main "$@"
