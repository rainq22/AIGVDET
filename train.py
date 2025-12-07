import torch
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence
from datasets import Dataset
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    AutoTokenizer
)
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
import swanlab
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoConfig

# --- 1. 配置区域 ---
# 建议使用绝对路径
MODEL_PATH = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct" 
OUTPUT_DIR = "./output/Qwen2.5-VL-Video-SFT"
MAX_LENGTH = 4096 
FREEZE_VISION = True  # 显存优化：冻结视觉塔
USE_LORA = True

# --- 2. 核心优化：自定义 Data Collator (增强鲁棒性) ---
@dataclass
class QwenVideoDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 提取文本输入和标签
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # 2. Pad 文本部分 (batch_first=True)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100 # Ignore index for loss
        )
        
        # 3. 截断 (防止异常数据导致OOM)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        # 4. 构建 Attention Mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # 5. [优化] 鲁棒的视觉特征收集
        # 不再只检查 instances[0]，而是检查 batch 中是否存在任何视觉特征
        
        # 处理图片 (Pixel Values)
        if any("pixel_values" in inst for inst in instances):
            pixel_values = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
            image_grid_thw = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            
            if len(pixel_values) > 0:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)
                batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        # 处理视频 (Pixel Values Videos) - Qwen2.5-VL 核心
        # 兼容 pixel_values_videos 和 video_pixel_values 两种命名
        video_keys = ["pixel_values_videos", "video_pixel_values"]
        target_key = next((k for k in video_keys if any(k in inst for inst in instances)), None)

        if target_key:
            pv_videos = [inst[target_key] for inst in instances if target_key in inst]
            video_grid_thw = [inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst]
            
            if len(pv_videos) > 0:
                # 官方模型 forward 默认使用 'pixel_values_videos'
                batch["pixel_values_videos"] = torch.cat(pv_videos, dim=0)
                batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        return batch

# --- 3. 数据处理函数 ---
def process_func(example, processor, tokenizer):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": example["conversations"][0]["value"]},
                {"type": "text", "text": "Analyze the video. Is it Real or Generated?"}
            ]
        }
    ]
    
    # 预处理视觉信息
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 输入 Processor
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=False, # [重要] padding 交给 Collator，节省处理时间
        return_tensors="pt"
    )
    
    # 处理 Label (Answer)
    response = example["conversations"][1]["value"]
    resp_tokens = tokenizer.encode(response, add_special_tokens=False)
    
    # 构建 Input IDs 和 Labels
    input_ids = inputs["input_ids"][0].tolist() + resp_tokens + [tokenizer.pad_token_id]
    labels = [-100] * len(inputs["input_ids"][0]) + resp_tokens + [tokenizer.pad_token_id]
    
    final_dict = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    
    # 提取视觉特征并移除 batch 维度 (processor 输出通常带 batch=1)
    if "pixel_values" in inputs:
        final_dict["pixel_values"] = inputs["pixel_values"] 
        final_dict["image_grid_thw"] = inputs["image_grid_thw"] # shape: (1, 3)
        
    if "pixel_values_videos" in inputs:
        final_dict["pixel_values_videos"] = inputs["pixel_values_videos"]
        final_dict["video_grid_thw"] = inputs["video_grid_thw"] # shape: (1, 3)
    elif "video_pixel_values" in inputs:
        final_dict["pixel_values_videos"] = inputs["video_pixel_values"]
        final_dict["video_grid_thw"] = inputs["video_grid_thw"]
            
    return final_dict

# --- 4. 主程序 ---
def train():
    # 初始化 Processor
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28,
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "sdpa"
    # 加载模型
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        config=config,
        device_map=None 
    )

    # [优化] 冻结视觉塔 (参考官方逻辑)
    if FREEZE_VISION:
        print("❄️ Freezing Vision Tower (saving ~30% memory)...")
        # Qwen2.5-VL 的视觉部分通常在 model.visual
        for param in model.visual.parameters():
            param.requires_grad = False
        # 确保 LLM 部分参与训练

    # LoRA 配置
    if USE_LORA:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=64, 
            lora_alpha=16, 
            lora_dropout=0.05, 
            bias="none",
            modules_to_save=[] # 不保存 embedding/head，只保存 adapter，减小权重体积
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 准备数据
    if not os.path.exists("train.json"): 
        raise FileNotFoundError("Run gen_cot_data.py first!")
    
    train_ds = Dataset.from_json("train.json")
    print(f"Loaded {len(train_ds)} samples from train.json")
    # 包装 process_func
    def _process(x): return process_func(x, processor, tokenizer)
    
    # 预处理数据 (Map)
    print("Processing dataset...")
    train_dataset = train_ds.map(_process, remove_columns=train_ds.column_names)
    
    eval_dataset = None
    if os.path.exists("test.json"):
        eval_ds = Dataset.from_json("test.json").select(range(50)) # 少量验证
        eval_dataset = eval_ds.map(_process, remove_columns=eval_ds.column_names)

    # SwanLab 配置
    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-VL-Video-Detection",
        experiment_name="Custom-Train-SFT",
        config={"freeze_vision": FREEZE_VISION, "max_length": MAX_LENGTH}
    )

    # 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # 视频显存大，建议保持1
        gradient_accumulation_steps=8, # 累计梯度，等效 batch=8
        num_train_epochs=3,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=True, # 必须开启 bf16
        gradient_checkpointing=True, # 必须开启显存优化
        dataloader_pin_memory=True,
        remove_unused_columns=False, # [重要] 防止 Collator 需要的自定义 key 被删除
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        report_to="none", # 关闭默认wandb，只用SwanLab
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenVideoDataCollator(tokenizer),
        callbacks=[swanlab_callback],
    )

    print("🚀 Starting training...")
    trainer.train()
    
    # 结束与保存
    swanlab.finish()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    processor.save_pretrained(f"{OUTPUT_DIR}/final") # 同时保存 processor 配置
    print(f"Training finished. Model saved to {OUTPUT_DIR}/final")

if __name__ == "__main__":
    train()

# torchrun --nproc_per_node=auto --master_port=29500 train.py --deepspeed qwen-vl-finetune/scripts/zero3.json