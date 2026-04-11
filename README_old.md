## 项目概述
基于 Qwen2.5-VL-7B 的 AI 生成视频检测系统（v2 多任务版）。

---

## 主要变化
- 多任务：Real/Generated + 生成器归因
- 多模板 CoT，减少模板记忆
- 3 种 loss 策略：full / answer_only / rationale_dropout
- LoRA 与训练超参做了去过拟合配置

---

## Loss 策略
| 模式 | 监督信号 |
|------|----------|
| `full` | `<think>` + `<answer>` + `<source>` |
| `answer_only` | `<answer>` + `<source>` |
| `rationale_dropout` | 以概率 p 丢弃 `<think>` loss |

---

## 数据格式
- 原版输出：`<answer>Real/Generated</answer>`
- v2 输出：`<answer>...</answer><source>cogvideox/.../real</source>`

支持来源：`real / cogvideox / easyanimate / hunyuanvideo / ltxvideo / ...`

---

## 运行
```bash
cd /data/srq/Qwen/Qwen2.5-VL

# 1) 生成 v2 数据
python gen_data_old.py
# 输出: /data1/srq/Qwen/Qwen2.5-VL/datasets/YYYYMMDD (auto-latest)
# 默认自动读取最新日期目录，如需指定请设置 DATASET_DIR

# 2) 预处理
python train_old.py --preprocess
# 缓存: /data1/srq/Qwen/Qwen2.5-VL/cache

# 3) 训练（三种策略不会互相覆盖）
torchrun --nproc_per_node=4 train_old.py --loss_mode full
torchrun --nproc_per_node=4 train_old.py --loss_mode answer_only
torchrun --nproc_per_node=4 train_old.py --loss_mode rationale_dropout --dropout_prob 0.5

# 4) 评测（先修改 eval_old.py 的 LORA_PATH）
python eval_old.py
```

---

## 输出目录
- 训练输出：`/data1/srq/Qwen/Qwen2.5-VL/output/v2-old/YYYYMMDD-HHMM-<loss_mode>`
- 最终模型：`final-<loss_mode>` 目录下

---

## 文件结构
```
Qwen2.5-VL/
├── train_old.py
├── eval_old.py
├── gen_data_old.py
├── README_V2.md
├── README.md
├── guide.md
├── data_prep/
│   └── extract_flow_residual.py
├── models/
│   ├── motion_adapter.py
│   └── dual_stream_qwen.py
└── (outputs on /data1/srq/Qwen/Qwen2.5-VL)
    ├── datasets/YYYYMMDD (auto-latest)
    ├── cache/
    │   ├── train_v2_pt/
    │   └── eval_v2_pt/
    └── output/
        └── Qwen2.5-VL-Video-SFT-v2-<loss_mode>/
```

---

## 已知问题与优化
- v2 token 边界用字符估算，可能误 mask。优化：v3 分段 tokenize 精确定位。
- v2 rationale_dropout 在预处理时固定。优化：v3 训练时动态随机。
- 模板数量仍有限。优化：可扩充模板或做随机扰动。

---

## motion 说明（双流）
- 修复 token 边界定位
- rationale_dropout 变为训练时动态

运行：
```bash
python gen_data_old.py
python train_motion.py --preprocess
torchrun --nproc_per_node=4 train_motion.py --loss_mode full
torchrun --nproc_per_node=4 train_motion.py --loss_mode answer_only
torchrun --nproc_per_node=4 train_motion.py --loss_mode rationale_dropout --dropout_prob 0.5
```
