# eval.py - 评估训练后的模型
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 单卡评估
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

import torch
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ========== 配置 ==========
BASE_MODEL_PATH = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH = "./output/Qwen2.5-VL-Video-SFT/final"  # 训练后的 LoRA 权重
TEST_JSON = "test.json"
MAX_SAMPLES = None  # 设为 None 评估全部，或设为 100 快速测试

def load_model():
    """加载基座模型 + LoRA 权重"""
    print("Loading base model...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 加载 LoRA 权重
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA weights from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()  # 合并权重加速推理
    else:
        print(f"⚠️ LoRA path not found: {LORA_PATH}, using base model")
    
    model.eval()
    return model, processor

def extract_answer(text):
    """从模型输出中提取 Real/Generated 标签"""
    text = text.lower()
    if "<answer>" in text:
        # 提取 <answer>...</answer> 之间的内容
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        if end > start:
            answer = text[start:end].strip()
            if "generated" in answer:
                return "Generated"
            elif "real" in answer:
                return "Real"
    # fallback: 直接搜索关键词
    if "generated" in text:
        return "Generated"
    elif "real" in text:
        return "Real"
    return "Unknown"

def get_ground_truth(sample):
    """从数据中提取真实标签"""
    response = sample["conversations"][1]["value"]
    return extract_answer(response)

@torch.no_grad()
def evaluate():
    model, processor = load_model()
    
    # 加载测试数据
    with open(TEST_JSON) as f:
        test_data = json.load(f)
    
    if MAX_SAMPLES:
        test_data = test_data[:MAX_SAMPLES]
    
    print(f"Evaluating {len(test_data)} samples...")
    
    y_true = []
    y_pred = []
    results = []
    
    for sample in tqdm(test_data):
        video_path = sample["conversations"][0]["value"]
        gt_label = get_ground_truth(sample)
        
        # 构建输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Analyze the video. Is it Real or Generated?"}
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
            
            # 生成
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # greedy decoding
            )
            
            # 解码
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            pred_label = extract_answer(response)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            pred_label = "Unknown"
            response = str(e)
        
        y_true.append(gt_label)
        y_pred.append(pred_label)
        results.append({
            "id": sample.get("id", ""),
            "video": video_path,
            "gt": gt_label,
            "pred": pred_label,
            "response": response[:200]  # 截断保存
        })
    
    # ========== 计算指标 ==========
    # 过滤掉 Unknown
    valid_mask = [(t != "Unknown" and p != "Unknown") for t, p in zip(y_true, y_pred)]
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]
    
    print("\n" + "="*50)
    print("📊 Evaluation Results")
    print("="*50)
    print(f"Total samples: {len(y_true)}")
    print(f"Valid samples: {len(y_true_valid)}")
    print(f"Unknown predictions: {len(y_true) - len(y_true_valid)}")
    
    if len(y_true_valid) > 0:
        acc = accuracy_score(y_true_valid, y_pred_valid)
        
        # 以 Generated 为正类
        precision = precision_score(y_true_valid, y_pred_valid, pos_label="Generated", zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, pos_label="Generated", zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, pos_label="Generated", zero_division=0)
        
        print(f"\n✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"🔍 Recall:    {recall:.4f}")
        print(f"📈 F1 Score:  {f1:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_true_valid, y_pred_valid, digits=4))
        
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=["Real", "Generated"])
        print(f"              Pred Real  Pred Generated")
        print(f"  True Real      {cm[0,0]:5d}       {cm[0,1]:5d}")
        print(f"  True Generated {cm[1,0]:5d}       {cm[1,1]:5d}")
    
    # 保存详细结果
    output_file = "eval_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": {
                "accuracy": acc if len(y_true_valid) > 0 else 0,
                "precision": precision if len(y_true_valid) > 0 else 0,
                "recall": recall if len(y_true_valid) > 0 else 0,
                "f1": f1 if len(y_true_valid) > 0 else 0,
            },
            "predictions": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed results saved to {output_file}")

if __name__ == "__main__":
    evaluate()