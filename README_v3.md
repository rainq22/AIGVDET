# Qwen2.5-VL 视频真伪检测 - 修复版 (v3)

## 📌 版本概述
本版本 (`train_v3.py`) 是 `train_v2.py` 的**工程修复与改进版**。虽然 v2 引入了 Label Mask 和多任务学习的理念，但在实现上存在两个严重缺陷（Token对齐误差、静态Dropout）。v3 修复了这些问题，提供了数学上更严谨的实现。

## 🛠️ v3 核心修复

### 1. 精确 Token 定位 (Precise Token Masking)
- **v2 问题**: 使用字符位置比例 (`char_pos / total_chars`) 估算 Token 位置，误差可达 ±10%，导致 `<think>` 标签未完全 Mask 或误伤 `<answer>`。
- **v3 修复**: 采用 **分段 Tokenize** 策略。
  - 将文本切分为 `Pre-Think`, `Think`, `Answer`, `Source` 等片段独立编码。
  - 精确计算每个片段的 Token 索引范围，确保 0 误差 Mask。

### 2. 动态 Rationale Dropout (Dynamic Dropout)
- **v2 问题**: Dropout 在 `preprocess` 阶段随机决定。一旦生成 `.pt` 文件，Mask 状态即固定。这就不是 Dropout（随机正则化），而是固定部分样本的 Hard Mask。
- **v3 修复**: 
  - 预处理仅保存原始 `response_text`。
  - 移至 **`DataCollator`** 中实现。
  - **每次训练迭代 (Step)** 都会重新随机采样 mask 状态，实现真正的正则化效果。

---

## 🚀 快速开始

### 1. 生成数据 (沿用 v2)
```bash
python gen_cot_data_v2.py
```

### 2. 预处理 (v3 专用)
> ⚠️ 注意：v3 的预处理与 v2 不兼容（v3 保存了文本用于动态 mask），请重新运行。
```bash
python train_v3.py --preprocess
```

### 3. 训练命令
所有 v2 的参数均适用，推荐使用 `rationale_dropout` 模式。

```bash
# 推荐配置: 动态 Dropout (p=0.5)
torchrun --nproc_per_node=4 train_v3.py \
    --loss_mode rationale_dropout \
    --dropout_prob 0.5 \
    --output_dir ./output/v3_dynamic_dropout

# 对照组: 只训练答案 (Answer-Only)
torchrun --nproc_per_node=4 train_v3.py --loss_mode answer_only

# 对照组: 全量 Loss
torchrun --nproc_per_node=4 train_v3.py --loss_mode full
```

---

## 📊 v2 vs v3 对比

| 特性 | v2 (Draft) | v3 (Final) | 影响 |
|------|------------|------------|------|
| **Mask 精度** | 字符估算 (粗糙) | **分段编码 (精确)** | 避免 loss 泄露或错误监督 |
| **Mask 时机** | 预处理时 (静态) | **DataLoader时 (动态)** | 真正的正则化，防止过拟合 |
| **代码结构** | 冗余 (多处重复) | **模块化** (`DynamicMaskCollator`) | 易于维护和扩展 |

## ⚠️ 兼容性提示
- **数据**: `v3` 训练脚本 **不能** 读取 `v2` 预处理的 `.pt` 文件（缺少原始文本字段）。
- **模型**: `v3` 训练出的 LoRA 权重与 `v2` 结构一致，`eval_v2.py` 可直接用来评测 v3 模型。
