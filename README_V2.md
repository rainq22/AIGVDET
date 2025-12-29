## 📌 项目概述

基于 Qwen2.5-VL-7B 的 AI 生成视频检测系统，相比原版本增加了两大创新点，适合发表论文。

---

## 🔬 创新点对比

### 原版本 vs 创新版本

| 维度 | 原版本 (train.py) | 创新版本 (train_v2.py) |
|------|------------------|----------------------|
| **任务** | 二分类 (Real/Generated) | 多任务 (检测 + 生成器归因) |
| **输出格式** | `<answer>Real/Generated</answer>` | `<answer>...</answer><source>cogvideox/...</source>` |
| **CoT 模板** | 每类 1 个固定模板 | 每类 5+ 个多样化模板 |
| **Loss 策略** | 全部 token 计算 loss | 支持 3 种模式 |
| **LoRA 配置** | r=64, alpha=16, dropout=0.05 | r=32, alpha=64, dropout=0.1 |
| **训练轮数** | 20 epochs | 5 epochs (防止过拟合) |
| **学习率** | 2e-4 | 1e-4 |
| **评测指标** | Acc/P/R/F1 | + 按生成器细分 + 归因准确率 |

---

## 🎯 创新点 A: Label Mask 策略

### 动机
原版本训练时，模型很快就"记住"了固定的 CoT 模板（50 step 左右 loss 降到 0.05），导致：
- 模型学的是"背模板"而非"判别特征"
- 泛化能力受限

### 实现
在 `train_v2.py` 中实现了 3 种 loss 计算模式：

\`\`\`python
# 配置选项
loss_mode: str = "full"        # full / answer_only / rationale_dropout
dropout_prob: float = 0.5      # rationale_dropout 模式下的丢弃概率
\`\`\`

| 模式 | 说明 | 监督信号 |
|------|------|----------|
| `full` | 对全部输出计算 loss | `<think>` + `<answer>` + `<source>` |
| `answer_only` | 只对答案部分计算 loss | `<answer>` + `<source>` |
| `rationale_dropout` | 以概率 p 丢弃思维链的 loss | 随机选择 |

### 核心代码
\`\`\`python
def apply_label_mask(response_text, response_token_ids, tokenizer, loss_mode, dropout_prob=0.5):
    if loss_mode == "full":
        return response_token_ids.copy()
    
    # 找到 <think>...</think> 的位置
    think_start = response_text.find("<think>")
    think_end = response_text.find("</think>")
    
    if loss_mode == "answer_only":
        # mask 掉 <think> 部分
        for i in range(think_start_tok, think_end_tok):
            masked_labels[i] = -100
            
    elif loss_mode == "rationale_dropout":
        if random.random() < dropout_prob:
            for i in range(think_start_tok, think_end_tok):
                masked_labels[i] = -100
    
    return masked_labels
\`\`\`

---

## 🎯 创新点 B: 多任务学习 - 生成器归因

### 动机
- 原版本只做二分类，任务太简单
- 增加"生成器归因"任务，让模型学习更细粒度的判别特征
- 可以回答"这个视频像是哪个生成器生成的"

### 实现

**数据格式变化** (`gen_cot_data_v2.py`):
\`\`\`python
# 原版本输出
"<think>...</think>\n<answer>Generated</answer>"

# 创新版本输出
"<think>...</think>\n<answer>Generated</answer>\n<source>cogvideox</source>"
\`\`\`

**支持的生成器类别**:
- `real` - 真实视频
- `cogvideox` - CogVideoX
- `easyanimate` - EasyAnimate  
- `hunyuanvideo` - HunyuanVideo
- `ltxvideo` - LTX-Video
- 更多可扩展...

### 模板多样化
每个类别准备了 5+ 个不同表述的 CoT 模板，防止模板记忆：

\`\`\`python
REAL_COT_TEMPLATES = [
    # 模板 1: 强调运动一致性
    """<think>
    1. Motion Analysis: Object movements follow natural physics...
    </think><answer>Real</answer><source>real</source>""",
    
    # 模板 2: 强调纹理细节
    """<think>
    1. Surface Details: Microscopic irregularities in textures...
    </think><answer>Real</answer><source>real</source>""",
    
    # ... 更多模板
]
\`\`\`

---

## 📊 评测指标 (eval_v2.py)

### 1. 基础指标
- **Accuracy**: 整体准确率
- **Precision/Recall/F1**: 以 Generated 为正类

### 2. 创新指标
- **Per-Category Accuracy**: 按生成器类别分别统计准确率
- **Source Attribution Accuracy**: 生成器归因准确率

### 示例输出
\`\`\`
📌 Binary Classification (Real vs Generated)
   ✅ Accuracy:  0.9500 (95.00%)
   🎯 Precision: 0.9600
   🔍 Recall:    0.9400
   📈 F1 Score:  0.9500

📌 Per-Category Accuracy (创新指标)
   cogvideox      : 0.9200 (92/100)
   easyanimate    : 0.9400 (94/100)
   hunyuanvideo   : 0.9100 (91/100)
   ltxvideo       : 0.9300 (93/100)
   real           : 0.9800 (98/100)

📌 Source Attribution (创新任务)
   准确率: 0.7500 (375/500)
\`\`\`

---

## 🚀 运行命令

### Step 1: 生成 v2 数据
\`\`\`bash
cd /data/srq/Qwen/Qwen2.5-VL
python gen_cot_data_v2.py
\`\`\`

### Step 2: 预处理数据
\`\`\`bash
# 默认 full 模式
python train_v2.py --preprocess

# 或指定 loss 模式 (影响 rationale_dropout 的随机 mask)
python train_v2.py --preprocess --loss_mode answer_only
\`\`\`

### Step 3: 训练 (选择 loss 模式)
\`\`\`bash
# Baseline: Full Loss
torchrun --nproc_per_node=4 train_v2.py --loss_mode full

# 创新 A1: Answer-Only Loss
torchrun --nproc_per_node=4 train_v2.py --loss_mode answer_only

# 创新 A2: Rationale Dropout (p=0.5)
torchrun --nproc_per_node=4 train_v2.py --loss_mode rationale_dropout --dropout_prob 0.5

# 创新 A2: Rationale Dropout (p=0.3)
torchrun --nproc_per_node=4 train_v2.py --loss_mode rationale_dropout --dropout_prob 0.3
\`\`\`

### Step 4: 评测
\`\`\`bash
# 修改 eval_v2.py 中的 LORA_PATH 指向训练好的模型
python eval_v2.py
\`\`\`

---

## 📝 论文 Ablation Table

### Table 1: Loss 策略对比 (创新点 A)

| Loss Mode | Binary Acc | F1 | Source Attr Acc |
|-----------|------------|-----|-----------------|
| Full (Baseline) | - | - | - |
| Answer-Only | - | - | - |
| Rationale Dropout (p=0.3) | - | - | - |
| Rationale Dropout (p=0.5) | - | - | - |

### Table 2: 按生成器类别准确率 (创新点 B)

| Generator | Full | Answer-Only | Dropout (p=0.5) |
|-----------|------|-------------|-----------------|
| CogVideoX | - | - | - |
| EasyAnimate | - | - | - |
| HunyuanVideo | - | - | - |
| LTX-Video | - | - | - |
| Real | - | - | - |
| **Average** | - | - | - |

### Table 3: 多任务对比

| Method | Detection Acc | Source Attribution Acc |
|--------|---------------|------------------------|
| Binary Only (原版本) | - | N/A |
| + Source Attribution | - | - |

### Table 4: 与其他方法对比

| Method | Accuracy | F1 | 备注 |
|--------|----------|-----|------|
| Baseline (Qwen2.5-VL zero-shot) | - | - | 无微调 |
| 原版本 (train.py) | - | - | 二分类 |
| **Ours (v2)** | - | - | 多任务 + Label Mask |

---

## 📁 文件结构

\`\`\`
Qwen2.5-VL/
├── train.py              # 原版训练脚本
├── eval.py               # 原版评测脚本
├── gen_cot_data.py       # 原版数据生成
│
├── train_v2.py           # 🆕 创新版训练脚本
├── eval_v2.py            # 🆕 创新版评测脚本
├── gen_cot_data_v2.py    # 🆕 创新版数据生成
├── README_v2.md          # 🆕 本文档
│
├── train.json            # 原版训练数据
├── test.json             # 原版测试数据
├── train_v2.json         # 🆕 创新版训练数据 (运行 gen_cot_data_v2.py 生成)
├── test_v2.json          # 🆕 创新版测试数据
│
├── cache/
│   ├── train_pt/         # 原版预处理缓存
│   ├── eval_pt/
│   ├── train_v2_pt/      # 🆕 创新版预处理缓存
│   └── eval_v2_pt/
│
└── output/
    ├── Qwen2.5-VL-Video-SFT/           # 原版输出
    └── Qwen2.5-VL-Video-SFT-v2/        # 🆕 创新版输出
        ├── final-full/
        ├── final-answer_only/
        └── final-rationale_dropout/
\`\`\`

---

## 🔧 配置对比

| 参数 | 原版本 | 创新版本 | 说明 |
|------|--------|----------|------|
| `lora_r` | 64 | 32 | 减小防止过拟合 |
| `lora_alpha` | 16 | 64 | 2x r |
| `lora_dropout` | 0.05 | 0.1 | 增加正则化 |
| `num_epochs` | 20 | 5 | 减少防止过拟合 |
| `learning_rate` | 2e-4 | 1e-4 | 降低学习率 |
| `warmup_steps` | 100 | 50 | 相应减少 |
| `eval_samples` | 50 | 100 | 增加评估样本 |

---

## 📚 引用建议

如果使用本代码，建议在论文中这样描述创新点：

> **Method:**
> We propose two key innovations for AI-generated video detection:
> 
> 1. **Rationale-aware Loss (RAL)**: Instead of computing loss over the entire chain-of-thought output, we selectively mask the reasoning tokens, forcing the model to encode discriminative features in its representations rather than memorizing template patterns.
>
> 2. **Multi-task Source Attribution**: Beyond binary real/fake classification, we jointly train the model to identify the specific generator (e.g., CogVideoX, Sora), enabling fine-grained analysis and improved generalization to unseen generators.

---

## ❓ FAQ

**Q: 为什么 answer_only 模式可能更好？**
A: 因为 CoT 模板是人工构造的，模型容易记住固定表述。只对答案计算 loss，逼迫模型把判别信息压缩到视觉特征中。

**Q: rationale_dropout 的 p 值如何选择？**
A: 建议从 0.5 开始，如果 loss 下降太慢可以降到 0.3。本质是在"学习推理过程"和"防止记忆"之间权衡。

**Q: 生成器归因准确率低怎么办？**
A: 这是预期的，因为不同生成器的差异可能很微妙。可以考虑增加生成器特定的模板，或者收集更多该生成器的数据。

## ⚠️ 已知问题与改进方案

### 问题 1: Label Mask 位置估算不精确

**问题描述**: 
当前 `apply_label_mask()` 使用字符位置比例来估算 token 位置，这是不精确的，因为不同字符的 token 数量不同。

**当前代码**:
\`\`\`python
def char_to_token_pos(char_pos):
    return int((char_pos / total_chars) * total_tokens)
\`\`\`

**影响**: 可能会错误地 mask 掉部分 answer token，或漏 mask 部分 think token。

**改进方案**: 分段 tokenize，精确定位每个部分的 token 边界。

---

### 问题 2: rationale_dropout 在预处理时固定

**问题描述**:
当前 `rationale_dropout` 的随机性发生在**预处理阶段**，而不是训练时动态决定。这意味着：
- 每个样本的 mask 状态在整个训练过程中是固定的
- 无法实现"每个 epoch 不同 mask"的效果

**改进方案**: 将 dropout 逻辑移到 DataLoader 或 Collator 中，训练时动态决定。

---

### 问题 3: 模板仍然不够多样

**问题描述**:
虽然每类有 5 个模板，但对于大规模训练，这仍然不够。模型可能还是会记住这 5 个模板。

**改进方案**:
1. 使用 LLM 生成更多模板（100+）
2. 在模板中加入随机扰动（同义词替换、语序调整）
3. 考虑移除 CoT，直接预测 answer

---

### 问题 4: 评测时 source 提取可能失败

**问题描述**:
如果模型没有按格式输出 `<source>...</source>`，当前的正则匹配会返回 "unknown"。

**改进方案**: 添加 fallback 逻辑，根据 CoT 内容推断 source。

---


---

## 📦 Version 3: 修复版本 (train_v3.py)

### ✅ 修复内容

| 问题 | v2 实现 | v3 修复 |
|------|---------|---------|
| Token 位置估算 | `len(text)*2.5` 字符估算 | **分段 tokenize，精确计算边界** |
| Rationale Dropout | 预处理时固定决定 | **Collator 中动态决定（每次 forward 随机）** |
| 代码复杂度 | 两个处理函数 | **统一处理，逻辑更清晰** |

### 🔬 精确 Token 定位原理

\`\`\`
原始 Response:
"<think>分析：观察视频帧...</think>\n<answer>Generated</answer>\n<source>cogvideox</source>"

v2 (估算):  字符位置 * 2.5 → 可能有 ±10% 误差
v3 (精确):  分段 tokenize → 精确到每个 token
\`\`\`

**v3 算法**:
\`\`\`python
# 分段提取
think_match = re.search(r'<think>.*?</think>', response)
answer_match = re.search(r'<answer>.*?</answer>', response)

# 按顺序 tokenize
segments = [
    ("pre_think", text[:think_match.start()]),
    ("think", text[think_match.start():think_match.end()]),  # ← 要 mask 的部分
    ("between", ...),
    ("answer", ...),  # ← 保留 loss
    ("source", ...),  # ← 保留 loss
]

# 累计计算每段的 token 范围
for name, seg_text in segments:
    tokens = tokenizer.encode(seg_text)
    ranges[name] = (current_pos, current_pos + len(tokens))
    current_pos += len(tokens)
\`\`\`

### 🔄 动态 Dropout 原理

**v2 问题**: 预处理时就决定 mask，导致：
- 相同样本在多个 epoch 都是同一种 mask 状态
- 失去了 Dropout 的随机正则化效果

**v3 解决**: 在 `DynamicMaskDataCollator` 中：
\`\`\`python
def __call__(self, instances):
    for inst in instances:
        if self.loss_mode == "rationale_dropout":
            # 每次 batch 都重新随机决定
            if random.random() < self.dropout_prob:
                # mask think 部分
            else:
                # 保留 think 部分
\`\`\`

### 📊 运行命令

\`\`\`bash
cd /data/srq/Qwen/Qwen2.5-VL

# 使用同一份 v2 数据
python gen_cot_data_v2.py

# v3 预处理（保存 response_text 供动态 mask 使用）
python train_v3.py --preprocess

# 训练三组实验
torchrun --nproc_per_node=4 train_v3.py --loss_mode full
torchrun --nproc_per_node=4 train_v3.py --loss_mode answer_only
torchrun --nproc_per_node=4 train_v3.py --loss_mode rationale_dropout --dropout_prob 0.5
\`\`\`

### 📁 文件结构

\`\`\`
Qwen2.5-VL/
├── train.py          # 原始版本
├── train_v2.py       # v2: Label Mask (存在问题)
├── train_v3.py       # v3: 修复版 ✅ 推荐使用
├── gen_cot_data.py   # 原始数据生成
├── gen_cot_data_v2.py# v2: 多任务 + 多模板
├── eval.py           # 原始评估
├── eval_v2.py        # v2: 细粒度评估
└── cache/
    ├── train_v2_pt/  # v2 缓存
    ├── train_v3_pt/  # v3 缓存 (含 response_text)
    └── ...
\`\`\`

---

## 🎯 版本选择指南

| 需求 | 推荐版本 |
|------|----------|
| 快速复现，不求创新 | train.py (原版) |
| 论文需要 Label Mask 消融实验 | **train_v3.py** ✅ |
| 只需要 answer_only 模式 | train_v2.py 足够 |
| 需要动态 Dropout 正则化 | **train_v3.py** ✅ |

---

## 📝 论文写作建议

### Method 部分

1. **Label Mask Strategy for Efficient VLM Fine-tuning**
   - 问题：标准 SFT 在所有 token 上计算 loss，包括冗长的 rationale
   - 方案：提出三种策略 (Full Loss / Answer-Only / Rationale Dropout)
   - 优势：减少训练时间 + 提升泛化能力

2. **Multi-task Learning for Deepfake Detection**
   - 任务 1：Real/Generated 二分类
   - 任务 2：生成器来源识别（4-way 分类）
   - 优势：共享特征表示，提升检测性能

### Experiments 部分

建议的消融实验表格：

| Method | Label Mask | Source Task | Accuracy | FPR | Gen-Acc | Source-Acc |
|--------|------------|-------------|----------|-----|---------|------------|
| Baseline | ✗ | ✗ | - | - | - | - |
| +LM-Full | Full | ✗ | - | - | - | - |
| +LM-AO | Answer-Only | ✗ | - | - | - | - |
| +LM-RD | Rationale-Dropout | ✗ | - | - | - | - |
| +MT | Full | ✓ | - | - | - | - |
| **Ours** | RD | ✓ | **-** | **-** | **-** | **-** |

cd /data/srq/Qwen/Qwen2.5-VL

# 1. 生成多任务数据
python gen_cot_data_v2.py

# 2. 预处理
python train_v3.py --preprocess

# 3. 训练消融实验
torchrun --nproc_per_node=4 train_v3.py --loss_mode full            # Baseline
torchrun --nproc_per_node=4 train_v3.py --loss_mode answer_only     # 创新 A1
torchrun --nproc_per_node=4 train_v3.py --loss_mode rationale_dropout --dropout_prob 0.5  # 创新 A2

# 4. 评估
python eval_v2.py