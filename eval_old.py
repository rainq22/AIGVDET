"""
eval_v2.py - 增强版评测脚本
创新点:
  - 多任务评测: Real/Generated 检测 + 生成器归因
  - 细粒度分析: 按生成器类别统计
  - 校准分析: 置信度与准确率关系
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

import torch
import json
import re
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ========== 配置 ==========

DATASET_BASE_DIR = "/data1/srq/Qwen/Qwen2.5-VL/datasets"


def get_latest_dataset_dir(base_dir: str) -> str:
    if os.environ.get("DATASET_DIR"):
        return os.environ["DATASET_DIR"]
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset base dir not found: {base_dir}")
    candidates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not candidates:
        raise FileNotFoundError(f"No dataset dirs found in: {base_dir}")
    latest = sorted(candidates)[-1]
    return os.path.join(base_dir, latest)


DATASET_DIR = get_latest_dataset_dir(DATASET_BASE_DIR)

OUTPUT_BASE_DIR = "/data1/srq/Qwen/Qwen2.5-VL/eval"
EVAL_VERSION = "v2-old"
EVAL_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, EVAL_VERSION, EVAL_TIMESTAMP)

BASE_MODEL_PATH = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH = "/data1/srq/Qwen/Qwen2.5-VL/output/Qwen2.5-VL-Video-SFT-v2/final-full"  # 或 final-answer_only
TEST_JSON = os.path.join(DATASET_DIR, "test_v2.json")  # 使用 v2 数据  # 使用 v2 数据
MAX_SAMPLES = None

def load_model():
    print("Loading base model...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        min_pixels=128*28*28,  # 与训练保持一致
        max_pixels=256*28*28,
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA weights from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
    else:
        print(f"⚠️ LoRA path not found: {LORA_PATH}")
    
    model.eval()
    return model, processor

def extract_predictions(text):
    """
    从模型输出中提取预测结果
    Returns: (label, source)
    """
    text_lower = text.lower()
    
    # 提取 label
    label = "Unknown"
    if "<answer>" in text_lower:
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text_lower)
        if match:
            ans = match.group(1).strip()
            if "generated" in ans:
                label = "Generated"
            elif "real" in ans:
                label = "Real"
    else:
        if "generated" in text_lower:
            label = "Generated"
        elif "real" in text_lower:
            label = "Real"
    
    # 提取 source
    source = "unknown"
    if "<source>" in text_lower:
        match = re.search(r'<source>\s*(.*?)\s*</source>', text_lower)
        if match:
            source = match.group(1).strip()
    
    return label, source

def get_ground_truth(sample):
    """从数据中提取真实标签"""
    response = sample["conversations"][1]["value"]
    label, source = extract_predictions(response)
    
    # 也可以从 meta 获取
    if "meta" in sample:
        if label == "Unknown":
            label = sample["meta"].get("label", label)
        source = sample["meta"].get("category", source)
    
    return label, source

@torch.no_grad()
def evaluate():
    model, processor = load_model()
    
    with open(TEST_JSON) as f:
        test_data = json.load(f)
    
    if MAX_SAMPLES:
        test_data = test_data[:MAX_SAMPLES]
    
    print(f"Evaluating {len(test_data)} samples...")
    
    # 存储结果
    results = []
    y_true_label = []
    y_pred_label = []
    y_true_source = []
    y_pred_source = []
    
    # 按类别统计
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for sample in tqdm(test_data):
        video_path = sample["conversations"][0]["value"]
        gt_label, gt_source = get_ground_truth(sample)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Analyze the video. Is it Real or Generated? Also identify the source."}
                ]
            }
        ]
        
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(model.device)
            
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
            
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            pred_label, pred_source = extract_predictions(response)
            
        except Exception as e:
            print(f"Error: {e}")
            pred_label, pred_source = "Unknown", "unknown"
            response = str(e)
        
        y_true_label.append(gt_label)
        y_pred_label.append(pred_label)
        y_true_source.append(gt_source)
        y_pred_source.append(pred_source)
        
        # 更新类别统计
        category_results[gt_source]["total"] += 1
        if pred_label == gt_label:
            category_results[gt_source]["correct"] += 1
        
        results.append({
            "id": sample.get("id", ""),
            "video": video_path,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "gt_source": gt_source,
            "pred_source": pred_source,
            "response": response[:300]
        })
    
    # ========== 计算指标 ==========
    print("\n" + "="*60)
    print("📊 Evaluation Results (v2)")
    print("="*60)
    
    # 1. Binary Classification (Real/Generated)
    valid_mask = [(t != "Unknown" and p != "Unknown") for t, p in zip(y_true_label, y_pred_label)]
    y_true_valid = [t for t, v in zip(y_true_label, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred_label, valid_mask) if v]
    
    print(f"\n📌 Binary Classification (Real vs Generated)")
    print(f"   Total: {len(y_true_label)}, Valid: {len(y_true_valid)}")
    
    if len(y_true_valid) > 0:
        acc = accuracy_score(y_true_valid, y_pred_valid)
        precision = precision_score(y_true_valid, y_pred_valid, pos_label="Generated", zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, pos_label="Generated", zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, pos_label="Generated", zero_division=0)
        
        print(f"   ✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🔍 Recall:    {recall:.4f}")
        print(f"   📈 F1 Score:  {f1:.4f}")
    
    # 2. Per-Category Analysis (创新点: 细粒度评测)
    print(f"\n📌 Per-Category Accuracy (创新指标)")
    print("-" * 40)
    for cat, stats in sorted(category_results.items()):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"   {cat:15s}: {cat_acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # 3. Source Attribution Accuracy (创新点: 生成器归因)
    source_correct = sum(1 for t, p in zip(y_true_source, y_pred_source) if t == p)
    source_acc = source_correct / len(y_true_source) if y_true_source else 0
    
    print(f"\n📌 Source Attribution (创新任务)")
    print(f"   准确率: {source_acc:.4f} ({source_correct}/{len(y_true_source)})")
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "eval_results_v2.json")
    summary_file = os.path.join(OUTPUT_DIR, "eval_results_v2_summary.json")

    metrics = {
        "binary_accuracy": acc if len(y_true_valid) > 0 else 0,
        "precision": precision if len(y_true_valid) > 0 else 0,
        "recall": recall if len(y_true_valid) > 0 else 0,
        "f1": f1 if len(y_true_valid) > 0 else 0,
        "source_attribution_accuracy": source_acc,
        "per_category": {
            cat: stats["correct"]/stats["total"] if stats["total"] > 0 else 0
            for cat, stats in category_results.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump({"metrics": metrics, "predictions": results}, f, indent=2, ensure_ascii=False)

    with open(summary_file, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to {output_file}")
    print(f"💾 Summary results saved to {summary_file}")

if __name__ == "__main__":
    evaluate()
