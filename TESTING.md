## 新视频数据测试文档（与训练格式一致）

本文用于“已有训练好的模型，新增一批视频进行测试”的完整流程。数据格式与训练时一致：
- 二分类：Real / Generated
- 生成来源：Fake 下按生成模型分目录

---

## 1. 前提
- 已有训练好的 LoRA 模型权重目录（用于 `eval.py` 的 `LORA_PATH`）
- 新视频数据可以访问（本地路径可读）

---

## 2. 数据组织格式（推荐）
把新视频按如下目录组织：

```
/new_dataset/
  Real/
    vid_0001.mp4
    vid_0002.mp4
  Fake/
    Gen2/
      vid_1001.mp4
    Sora/
      vid_2001.mp4
    Pika/
      vid_3001.mp4
```

规则：
- `Real/` 下所有视频标注为 `Real`，`source=Real`
- `Fake/<Generator>/` 下所有视频标注为 `Generated`，`source=<Generator>`

---

## 3. 生成 test_v2.json
执行以下脚本，把目录结构转成 `test_v2.json`：

```bash
python - <<'PY'
import os, json

DATA_ROOT = '/data1/srq/Qwen/Qwen2.5-VL/new_dataset'
OUT_JSON = '/data1/srq/Qwen/Qwen2.5-VL/datasets/20260410/test_v2.json'

exts = ('.mp4', '.mov', '.mkv', '.webm')

def iter_videos(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(exts):
                yield os.path.join(dirpath, fn)

samples = []
idx = 0

# Real
real_dir = os.path.join(DATA_ROOT, 'Real')
if os.path.isdir(real_dir):
    for vp in iter_videos(real_dir):
        samples.append({
            'id': f'sample_{idx:06d}',
            'conversations': [
                {'from': 'user', 'value': vp},
                {'from': 'assistant', 'value': '<answer>Real</answer>\n<source>Real</source>'}
            ]
        })
        idx += 1

# Fake
fake_dir = os.path.join(DATA_ROOT, 'Fake')
if os.path.isdir(fake_dir):
    for gen in sorted(os.listdir(fake_dir)):
        gen_dir = os.path.join(fake_dir, gen)
        if not os.path.isdir(gen_dir):
            continue
        for vp in iter_videos(gen_dir):
            samples.append({
                'id': f'sample_{idx:06d}',
                'conversations': [
                    {'from': 'user', 'value': vp},
                    {'from': 'assistant', 'value': f'<answer>Generated</answer>\n<source>{gen}</source>'}
                ]
            })
            idx += 1

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print('Saved:', OUT_JSON, 'samples:', len(samples))
PY
```

说明：
- 目录名就是生成器类别名称（如 `Gen2`、`Sora`）
- 如果需要自定义 label/source，可直接修改脚本

---

## 4. 指定数据目录（推荐用参数传入）
假设本次数据目录为：
- /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410

---

## 5. 生成光流特征（v3-motion 评测需要）
```bash
python data_prep/extract_flow_residual.py --skip_existing \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410
# 输出: /data1/srq/Qwen/Qwen2.5-VL/cache/flow_features
```

---

## 6. 预处理（生成缓存）
```bash
python train_motion.py --preprocess \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410 \
  --preprocess_workers 6 --preprocess_save_every 200 \
  --video_reader decord --max_pixels 150528 --video_max_frames 8 \
  --cpu_threads 24
# 输出: /data1/srq/Qwen/Qwen2.5-VL/cache/train_v3_motion_pt
```

---

## 7. 评测
1) 修改 `eval.py` 中的 `LORA_PATH` 为你的模型输出目录，或用 `--lora_path` 覆盖。
2) 运行评测：

```bash
python eval.py \
  --dataset_dir /data1/srq/Qwen/Qwen2.5-VL/datasets/20260410 \
  --lora_path /data1/srq/Qwen/Qwen2.5-VL/output/your_run/final-full
# 输出: /data1/srq/Qwen/Qwen2.5-VL/eval/motion/YYYYMMDD-HHMM/
```

---

## 8. 结果文件
- 详细结果: `/data1/srq/Qwen/Qwen2.5-VL/eval/motion/YYYYMMDD-HHMM/eval_results_v2.json`
- 汇总指标: `/data1/srq/Qwen/Qwen2.5-VL/eval/motion/YYYYMMDD-HHMM/eval_results_v2_summary.json`
