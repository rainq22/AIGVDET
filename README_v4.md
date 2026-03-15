# Qwen2.5-VL 视频真伪检测 - 改进版 (v4)

## 版本概述
v4 在 v3 的基础上继续改进训练稳定性与泛化能力，重点解决“硬 mask 带来的监督信息损失”与“重复 tokenize 的效率问题”，同时加入更丰富的提示词和模板扰动策略。

## 核心改进

### 1. 统一的权重式 Loss (Weighted Loss)
- v2/v3 通过把 `<think>` 直接 mask 掉来实现 `answer_only` 或 `rationale_dropout`。
- v4 使用 **权重式 loss**：
  - `answer_only` 等价于把 `<think>` 权重设为 0。
  - `rationale_dropout` 等价于 **每个 batch 动态**把 `<think>` 权重置 0。
  - 新增 `rationale_weighted`：把 `<think>` 设为较低权重，避免硬切。

### 2. 缓存分段 Token 范围
- v3 在训练时动态计算 token 范围，仍需重复 tokenize。
- v4 在 **预处理阶段**就把 `<think>/<answer>/<source>` 的 token 范围写入缓存，训练时直接使用，减少开销。

### 3. Prompt 多样化
- v4 预处理时随机选择多条 prompt 模板，降低模型对固定 prompt 记忆风险。

### 4. 数据生成改进
- 新增 `gen_cot_data_v4.py`：
  - 模板数量增加
  - 轻量扰动 `<think>` 内部顺序
  - generator 名称规范化，降低 source 噪声

---

## 快速开始

### 1. 生成数据
```bash
python gen_cot_data_v4.py
```

### 2. 预处理
```bash
python train_v4.py --preprocess
```

### 3. 训练
```bash
# Full Loss
torchrun --nproc_per_node=4 train_v4.py --loss_mode full

# Answer Only
torchrun --nproc_per_node=4 train_v4.py --loss_mode answer_only

# Rationale Dropout (动态)
torchrun --nproc_per_node=4 train_v4.py --loss_mode rationale_dropout --dropout_prob 0.5

# Rationale Weighted (新模式)
torchrun --nproc_per_node=4 train_v4.py --loss_mode rationale_weighted --think_weight 0.3
```

---

## v3 vs v4 对比

| 特性 | v3 | v4 | 影响 |
|------|----|----|------|
| Mask 方式 | Hard mask | Weighted mask | 更平滑监督信号 |
| Dropout 位置 | Collator 动态 | Collator 动态 | 保留 | 
| Token 边界 | 动态计算 | 预处理缓存 | 更高效率 |
| Prompt 多样化 | 固定 | 随机模板 | 降低 prompt 记忆 |
| 数据模板 | v2 模板 | v4 扩展模板 | 更强泛化 |

---

## 文件结构

```
Qwen2.5-VL/
├── train_v4.py
├── gen_cot_data_v4.py
├── README_v4.md
├── train_v4.json          # 运行 gen_cot_data_v4.py 生成
├── test_v4.json
├── cache/
│   ├── train_v4_pt/        # v4 预处理缓存
│   └── eval_v4_pt/
└── output/
    └── Qwen2.5-VL-Video-SFT-v4/
        ├── final-full/
        ├── final-answer_only/
        ├── final-rationale_dropout/
        └── final-rationale_weighted/
```

---

## 版本选择建议

| 需求 | 推荐版本 |
|------|----------|
| 复现旧实验 | v3 |
| 论文新增消融 (加权策略) | v4 |
| 更平滑监督 + 更强泛化 | v4 |

---

## 常见问题

**Q: rationale_weighted 的 think_weight 建议多少？**
A: 推荐 0.1~0.3，越小越接近 answer_only。

**Q: v4 能否直接读取 v3 缓存？**
A: 不能。v4 缓存中包含分段 token 范围，需要重新预处理。

**Q: v4 评测脚本可以用 eval_v2.py 吗？**
A: 可以，模型输出格式保持兼容。
