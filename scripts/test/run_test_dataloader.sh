#!/usr/bin/zsh

# 测试 Perturbation Prediction 数据加载器脚本
# 用于探索 batch_data 的数据结构

echo "========================================="
echo "测试 Perturbation Prediction DataLoader"
echo "========================================="
echo ""

# 设置环境变量
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""  # 不使用GPU，只测试数据结构

# 方案1: 使用简化版测试脚本（推荐）
echo "运行测试脚本: test_dataloader.py"
echo "-----------------------------------------"
python test_dataloader.py

echo ""
echo ""

# 方案2: 使用完整版测试脚本（更详细）
echo "运行测试脚本: test_pp_dataloader.py"
echo "-----------------------------------------"
PYTHONPATH=src python test_pp_dataloader.py

echo ""
echo "========================================="
echo "测试完成!"
echo "========================================="
