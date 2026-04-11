## 项目概述

基于 Qwen2.5-VL 的视频真伪检测工程，包含 v2 基线训练、评测与双流 motion 扩展（光流二阶残差）。本工程默认将 v2 数据、缓存与输出写入 `/data1/srq/Qwen/Qwen2.5-VL`。

---

## 版本说明

| 版本 | 脚本 | 说明 |
|------|------|------|
| v2 | train_old.py | 多任务 + Label Mask（full / answer_only / rationale_dropout） |
| v3-motion | train_motion.py | 双流注入 + 光流二阶残差（基于 v2 数据与流程） |

---

## 核心创新点

### 1. Label Mask 策略 (v2)
- full: 全量 loss
- answer_only: 仅监督 `<answer>/<source>`
- rationale_dropout: 随机丢弃 `<think>` 监督

### 2. 多任务学习 (v2)
- 输出格式包含 `<answer>` 与 `<source>`
- 支持生成器归因评测

### 3. 双流 Motion 注入 (v3-motion)
- 预先计算光流二阶残差
- Motion Adapter 将残差映射为 motion tokens
- 与文本 token 拼接送入 Qwen 主干

---

## 目录结构

```
Qwen2.5-VL/
├── train_old.py                  # v2 训练
├── eval_old.py                   # v2 评测
├── gen_data.py               # v2 数据生成（默认）
├── train_motion.py           # v3-motion 训练
├── data_prep/
│   └── extract_flow_residual.py # 光流二阶残差提取
├── models/
│   ├── motion_adapter.py
│   └── dual_stream_qwen.py
├── README_old.md                 # v2 详细文档
├── guide.md                     # motion 方案设计
└── (outputs on /data1/srq/Qwen/Qwen2.5-VL/output)
```

---

## 快速开始 (v2 基线)

以下步骤仅针对 v2 基线训练与评测，不包含 motion 双流。

### Step 1: 生成 v2 数据
```bash
cd /data/srq/Qwen/Qwen2.5-VL
python gen_data.py
# 数据输出: /data1/srq/Qwen/Qwen2.5-VL/datasets/YYYYMMDD (auto-latest)
# 默认自动读取最新日期目录，如需指定请设置 DATASET_DIR
```

### Step 2: v2 预处理
```bash
python train_old.py --preprocess
# 缓存输出: /data1/srq/Qwen/Qwen2.5-VL/cache
```

### Step 3: v2 训练
```bash
torchrun --nproc_per_node=4 train_old.py --loss_mode full
# 输出路径: /data1/srq/Qwen/Qwen2.5-VL/output/v2-old/YYYYMMDD-HHMM-<loss_mode>
```

### Step 4: v2 评测
```bash
python eval_old.py
# 默认自动读取最新日期目录，如需指定请设置 DATASET_DIR
# 详细结果: /data1/srq/Qwen/Qwen2.5-VL/eval/v2-old/YYYYMMDD-HHMM/eval_results.json
# 核心结果: /data1/srq/Qwen/Qwen2.5-VL/eval/v2-old/YYYYMMDD-HHMM/eval_results_summary.json
```

---

## v3-motion (双流版本)

v3-motion 在 v2 数据上额外加入光流二阶残差与 motion 注入。
先完成 v2 数据生成（使用 gen_data.py）；推荐用 --dataset_dir 指定本次数据批次。

### Step 1: 生成光流二阶残差
```bash
python data_prep/extract_flow_residual.py --skip_existing \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410
# 默认同时处理 train_v2.json + test_v2.json
# 输出: /data1/srq/Qwen/Qwen2.5-VL/cache/flow_features
```

### Step 2: v3-motion 预处理
```bash
python train_motion.py --preprocess \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410 \
  --preprocess_workers 6 --preprocess_save_every 200 \
  --video_reader decord --max_pixels 150528 --video_max_frames 8 \
  --cpu_threads 24
# 缓存输出: /data1/srq/Qwen/Qwen2.5-VL/cache/train_v3_motion_pt
# 说明: 可调低/调高 preprocess_workers 与 cpu_threads 控制速度与服务器负载
```

### Step 3: v3-motion 训练
```bash
torchrun --nproc_per_node=4 train_motion.py --loss_mode full
# 输出路径: /data1/srq/Qwen/Qwen2.5-VL/output/motion/YYYYMMDD-HHMM-<loss_mode>
```

### Step 4: v3-motion 评测
```bash
python eval.py \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410 \
  --lora_path /data1/srq/Qwen/Qwen2.5-VL/output/your_run/final-full
# 详细结果: /data1/srq/Qwen/Qwen2.5-VL/eval/motion/YYYYMMDD-HHMM/eval_results.json
# 核心结果: /data1/srq/Qwen/Qwen2.5-VL/eval/motion/YYYYMMDD-HHMM/eval_results_summary.json
```

---

## 评测指标

- Binary Accuracy / Precision / Recall / F1
- Per-Category Accuracy
- Source Attribution Accuracy

---


## 新视频数据测试流程（从0到1）

下面以“模型已训练好、你有一批新视频要测试”为前提，给出完整流程。

### Step 0: 准备数据目录
把视频统一放到一个目录，例如：
- /data1/srq/Qwen/Qwen2.5-VL/new_videos/

### Step 1: 生成 test_v2.json
在任意位置执行以下脚本，生成测试 JSON：
```bash
python - <<'PY'
import json, os
video_dir = '/data1/srq/Qwen/Qwen2.5-VL/new_videos'
out_json = '/data1/srq/Qwen/Qwen2.5-VL/datasets/20260410/test_v2.json'
videos = [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir))
          if f.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))]
data = []
for i, vp in enumerate(videos):
    data.append({
        'id': f'sample_{i:06d}',
        'conversations': [
            {'from': 'user', 'value': vp},
            {'from': 'assistant', 'value': '<answer>Unknown</answer>\n<source>Unknown</source>'}
        ]
    })
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Saved:', out_json, 'samples:', len(data))
PY
```

说明：
- 如果你有真实标签，把 `<answer>` 和 `<source>` 替换成真实值。
- 没有标签也能跑流程，但评测指标没有意义。

### Step 2: 指定数据目录（推荐用参数传入）
假设本次数据目录为：
- /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410

### Step 3: 生成光流特征（v3-motion 评测需要）
```bash
python data_prep/extract_flow_residual.py --skip_existing \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410
```

### Step 4: 预处理（生成缓存）
```bash
python train_motion.py --preprocess \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410 \
  --preprocess_workers 2 --preprocess_save_every 200 \
  --video_reader decord --max_pixels 150528 --video_max_frames 8 \
  --cpu_threads 16
```

### Step 5: 评测
```bash
python eval.py \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410 \
  --lora_path /data1/srq/Qwen/Qwen2.5-VL/output/your_run/final-full
```

## 常见问题

**Q: v2 与 v3-motion 的数据是否兼容？**
A: 兼容。v3-motion 在 v2 数据基础上额外读取 flow_residuals。

**Q: eval.py 能评测 v3-motion 吗？**
A: 可以，只需将 `LORA_PATH` 指向 v3-motion 的输出目录。

---

## 推荐阅读
- v2 训练与消融细节见 README_old.md
- motion 方案设计见 guide.md
