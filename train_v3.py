# ╔══════════════════════════════════════════════════════════════════╗
# ║  train_v3.py - 修复版训练脚本                                     ║
# ║  修复:                                                            ║
# ║    1. 精确的 token 位置定位                                        ║
# ║    2. 动态 rationale dropout (训练时决定)                          ║
# ║    3. 更简洁的实现                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import sys
import warnings
import time
import re
import random
from datetime import datetime
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*fast processor.*")

import torch
import json
from dataclasses import dataclass
from typing import Dict, Optional, List, Sequence
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        配置                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class TrainConfig:
    model_path: str = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "./output/Qwen2.5-VL-Video-SFT-v3"
    
    train_json: str = "train_v2.json"
    test_json: str = "test_v2.json"
    train_cache: str = "./cache/train_v3_pt"
    eval_cache: str = "./cache/eval_v3_pt"
    
    max_length: int = 8192
    freeze_vision: bool = True
    use_lora: bool = True
    
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # Loss 模式 (修复: rationale_dropout 在 Collator 中动态执行)
    loss_mode: str = "full"  # full / answer_only / rationale_dropout
    dropout_prob: float = 0.5
    
    per_device_batch_size: int = 1
    gradient_accumulation: int = 8
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    
    num_workers: int = 4
    cpu_threads: int = 8
    
    eval_samples: int = 100
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10

def parse_args():
    config = TrainConfig()
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--loss_mode" and i + 1 < len(args):
            config.loss_mode = args[i + 1]
            i += 2
        elif args[i] == "--dropout_prob" and i + 1 < len(args):
            config.dropout_prob = float(args[i + 1])
            i += 2
        elif args[i] == "--preprocess":
            i += 1
        elif args[i] in ["--help", "-h"]:
            i += 1
        else:
            i += 1
    return config

CONFIG = parse_args()

os.environ["OMP_NUM_THREADS"] = str(CONFIG.cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(CONFIG.cpu_threads)
torch.set_num_threads(CONFIG.cpu_threads)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        工具函数                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    return get_rank() == 0

def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_banner(text, char="═", width=60):
    if not is_main_process():
        return
    border = char * width
    padding = (width - len(text) - 2) // 2
    print(f"\n╔{border}╗")
    print(f"║{' ' * padding}{text}{' ' * (width - padding - len(text))}║")
    print(f"╚{border}╝\n")

def print_config():
    if not is_main_process():
        return
    print("\n" + "=" * 60)
    print("📋 训练配置 (v3 - 修复版)")
    print("=" * 60)
    print(f"  模型路径:      {CONFIG.model_path}")
    print(f"  输出目录:      {CONFIG.output_dir}")
    print(f"  使用 LoRA:     r={CONFIG.lora_r}, alpha={CONFIG.lora_alpha}")
    print("-" * 60)
    print(f"  🔬 Loss 模式:   {CONFIG.loss_mode}")
    if CONFIG.loss_mode == "rationale_dropout":
        print(f"     Dropout率:  {CONFIG.dropout_prob} (动态)")
    print("-" * 60)
    print(f"  Batch Size:    {CONFIG.per_device_batch_size} x 4卡 x {CONFIG.gradient_accumulation}累积")
    print(f"  学习率:        {CONFIG.learning_rate}")
    print(f"  训练轮数:      {CONFIG.num_epochs}")
    print("=" * 60 + "\n")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    回调                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if is_main_process():
            print_banner("🚀 开始训练 (v3)")
            print(f"  📊 总步数: {state.max_steps}")
            print(f"  🔬 Loss 模式: {CONFIG.loss_mode}")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        if is_main_process():
            epoch = int(state.epoch) + 1 if state.epoch else 1
            print(f"\n{'─' * 50}")
            print(f"📅 Epoch {epoch}/{args.num_train_epochs}")
            print(f"{'─' * 50}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if is_main_process() and logs:
            step = state.global_step
            log_parts = [f"Step {step:5d}"]
            if "loss" in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")
            if "learning_rate" in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")
            if "eval_loss" in logs:
                log_parts.append(f"Eval: {logs['eval_loss']:.4f}")
            print(f"  {'  |  '.join(log_parts)}")
    
    def on_train_end(self, args, state, control, **kwargs):
        if is_main_process() and self.start_time:
            print_banner("✅ 训练完成")
            print(f"  ⏱️  总耗时: {format_time(time.time() - self.start_time)}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  修复 1: 精确的 Token 边界定位                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_segment_token_ranges(response_text: str, tokenizer) -> dict:
    """
    精确计算 response 中各段的 token 范围
    
    Returns:
        {
            "think": (start, end),      # <think>...</think> 的 token 范围
            "answer": (start, end),     # <answer>...</answer> 的 token 范围
            "source": (start, end),     # <source>...</source> 的 token 范围
        }
    """
    # 分段提取
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    source_match = re.search(r'<source>(.*?)</source>', response_text, re.DOTALL)
    
    result = {}
    current_pos = 0
    
    # 按顺序 tokenize 每个部分
    segments = []
    
    if think_match:
        # <think> 标签
        pre_think = response_text[:think_match.start()]
        think_content = response_text[think_match.start():think_match.end()]
        segments.append(("pre_think", pre_think))
        segments.append(("think", think_content))
    
    if answer_match:
        # 找到 think 结束到 answer 开始之间的内容
        if think_match:
            between = response_text[think_match.end():answer_match.start()]
            segments.append(("between1", between))
        answer_content = response_text[answer_match.start():answer_match.end()]
        segments.append(("answer", answer_content))
    
    if source_match:
        if answer_match:
            between = response_text[answer_match.end():source_match.start()]
            segments.append(("between2", between))
        source_content = response_text[source_match.start():source_match.end()]
        segments.append(("source", source_content))
    
    # 计算每段的 token 范围
    current_token_pos = 0
    for seg_name, seg_text in segments:
        if not seg_text:
            continue
        seg_tokens = tokenizer.encode(seg_text, add_special_tokens=False)
        seg_len = len(seg_tokens)
        
        if seg_name in ["think", "answer", "source"]:
            result[seg_name] = (current_token_pos, current_token_pos + seg_len)
        
        current_token_pos += seg_len
    
    return result

def apply_label_mask_precise(response_text: str, response_token_ids: List[int], 
                             tokenizer, loss_mode: str, dropout_prob: float = 0.5) -> List[int]:
    """
    精确的 Label Mask（修复版）
    """
    if loss_mode == "full":
        return response_token_ids.copy()
    
    # 获取精确的 token 范围
    ranges = get_segment_token_ranges(response_text, tokenizer)
    
    masked_labels = response_token_ids.copy()
    
    if "think" not in ranges:
        return masked_labels
    
    think_start, think_end = ranges["think"]
    
    if loss_mode == "answer_only":
        # mask 掉整个 <think> 部分
        for i in range(think_start, min(think_end, len(masked_labels))):
            masked_labels[i] = -100
            
    elif loss_mode == "rationale_dropout":
        # 动态决定是否 mask
        if random.random() < dropout_prob:
            for i in range(think_start, min(think_end, len(masked_labels))):
                masked_labels[i] = -100
    
    return masked_labels

# ╔══════════════════════════════════════════════════════════════════╗
# ║  修复 2: 动态 Dropout 的 Data Collator                            ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class DynamicMaskDataCollator:
    """
    支持动态 rationale dropout 的 Data Collator
    """
    tokenizer: transformers.PreTrainedTokenizer
    loss_mode: str = "full"
    dropout_prob: float = 0.5
    max_length: int = 8192

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        
        for inst in instances:
            input_ids = inst["input_ids"]
            
            # 修复: 如果是 rationale_dropout 模式，在这里动态决定
            if self.loss_mode == "rationale_dropout" and "response_text" in inst:
                response_text = inst["response_text"]
                response_start = inst.get("response_start", 0)
                
                # 获取 response 部分的 token
                response_tokens = input_ids[response_start:].tolist()
                
                # 动态 mask
                masked_response = apply_label_mask_precise(
                    response_text, response_tokens, self.tokenizer,
                    self.loss_mode, self.dropout_prob
                )
                
                # 重建 labels
                labels = [-100] * response_start + masked_response
                labels = torch.tensor(labels, dtype=torch.long)
            else:
                labels = inst["labels"]
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        # 视频数据
        video_key = next(
            (k for k in ["pixel_values_videos", "video_pixel_values"] 
             if any(k in inst for inst in instances)), None
        )
        if video_key:
            pv_videos = [inst[video_key] for inst in instances if video_key in inst]
            video_grid_thw = [inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst]
            if pv_videos:
                batch["pixel_values_videos"] = torch.cat(pv_videos, dim=0)
                batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        return batch

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     数据处理                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def process_func(example, processor, tokenizer, loss_mode="full", dropout_prob=0.5):
    """处理单个样本"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": example["conversations"][0]["value"]},
                {"type": "text", "text": "Analyze the video. Is it Real or Generated? Also identify the source."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=False,
        return_tensors="pt"
    )
    
    response = example["conversations"][1]["value"]
    resp_tokens = tokenizer.encode(response, add_special_tokens=False)
    
    prompt_len = len(inputs["input_ids"][0])
    
    # 构建完整序列
    input_ids = inputs["input_ids"][0].tolist() + resp_tokens + [tokenizer.pad_token_id]
    
    # 对于 answer_only 模式，在预处理时就 mask
    # 对于 rationale_dropout 模式，保存原始 labels，在 Collator 中动态 mask
    if loss_mode == "answer_only":
        masked_resp = apply_label_mask_precise(response, resp_tokens, tokenizer, loss_mode, dropout_prob)
        labels = [-100] * prompt_len + masked_resp + [tokenizer.pad_token_id]
    else:
        # full 或 rationale_dropout: 保存原始 labels
        labels = [-100] * prompt_len + resp_tokens + [tokenizer.pad_token_id]
    
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_text": response,          # 保存原始文本，供动态 mask 使用
        "response_start": prompt_len,       # response 开始位置
    }
    
    if "pixel_values_videos" in inputs:
        result["pixel_values_videos"] = inputs["pixel_values_videos"]
        result["video_grid_thw"] = inputs["video_grid_thw"]
    elif "video_pixel_values" in inputs:
        result["pixel_values_videos"] = inputs["video_pixel_values"]
        result["video_grid_thw"] = inputs["video_grid_thw"]
            
    return result

# ╔══════════════════════════════════════════════════════════════════╗
# ║                  Dataset                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "index.json"), "r") as f:
            self.index = json.load(f)
        self.length = len(self.index)
        print(f"  加载数据集: {self.length} 样本")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        pt_file = os.path.join(self.cache_dir, self.index[idx])
        return torch.load(pt_file, weights_only=False)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     预处理                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def preprocess_data():
    print_banner("📦 数据预处理 (v3)")
    
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path, 
        min_pixels=128*28*28,
        max_pixels=256*28*28,
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
    
    def process_and_save(json_file, cache_dir, desc, max_samples=None):
        if os.path.exists(os.path.join(cache_dir, "index.json")):
            print(f"✅ 缓存已存在: {cache_dir}")
            return
        
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"📂 加载数据: {json_file}, 样本数: {len(data)}")
        
        index = []
        failed = []
        
        for i, sample in enumerate(tqdm(data, desc=desc)):
            try:
                result = process_func(sample, processor, tokenizer, CONFIG.loss_mode, CONFIG.dropout_prob)
                pt_filename = f"sample_{i:06d}.pt"
                torch.save(result, os.path.join(cache_dir, pt_filename))
                index.append(pt_filename)
            except Exception as e:
                failed.append((i, str(e)))
                print(f"\n⚠️ 样本 {i} 失败: {e}")
        
        with open(os.path.join(cache_dir, "index.json"), "w") as f:
            json.dump(index, f)
        
        print(f"\n✅ 完成! 成功: {len(index)}, 失败: {len(failed)}")
    
    process_and_save(CONFIG.train_json, CONFIG.train_cache, "处理训练集")
    if os.path.exists(CONFIG.test_json):
        process_and_save(CONFIG.test_json, CONFIG.eval_cache, "处理验证集", CONFIG.eval_samples)
    
    print(f"\n🎉 预处理完成! 运行: torchrun --nproc_per_node=4 train_v3.py --loss_mode {CONFIG.loss_mode}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     训练                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def train():
    print_config()
    
    print_main("📥 加载模型...")
    processor = AutoProcessor.from_pretrained(CONFIG.model_path, min_pixels=128*28*28, max_pixels=256*28*28, padding_side="right")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
    
    config = AutoConfig.from_pretrained(CONFIG.model_path)
    config._attn_implementation = "sdpa"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG.model_path, torch_dtype=torch.bfloat16, config=config, device_map=None, low_cpu_mem_usage=True
    )
    
    if CONFIG.freeze_vision:
        print_main("❄️  冻结 Vision Tower")
        for param in model.visual.parameters():
            param.requires_grad = False

    if CONFIG.use_lora:
        print_main("🔧 应用 LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=CONFIG.lora_r, lora_alpha=CONFIG.lora_alpha, lora_dropout=CONFIG.lora_dropout, bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()

    if not os.path.exists(os.path.join(CONFIG.train_cache, "index.json")):
        raise FileNotFoundError(f"\n❌ 请先运行: python train_v3.py --preprocess\n")
    
    print_main(f"📂 加载数据...")
    train_dataset = VideoDataset(CONFIG.train_cache)
    eval_dataset = VideoDataset(CONFIG.eval_cache) if os.path.exists(os.path.join(CONFIG.eval_cache, "index.json")) else None

    callbacks = [
        TrainingMonitorCallback(),
        SwanLabCallback(
            project="Qwen2.5-VL-Video-Detection-v3",
            experiment_name=f"{CONFIG.loss_mode}-{datetime.now().strftime('%m%d-%H%M')}",
            config={"loss_mode": CONFIG.loss_mode, "lora_r": CONFIG.lora_r, "lr": CONFIG.learning_rate}
        ),
    ]

    training_args = TrainingArguments(
        output_dir=CONFIG.output_dir,
        per_device_train_batch_size=CONFIG.per_device_batch_size,
        per_device_eval_batch_size=CONFIG.per_device_batch_size,
        gradient_accumulation_steps=CONFIG.gradient_accumulation,
        num_train_epochs=CONFIG.num_epochs,
        learning_rate=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
        warmup_steps=CONFIG.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        dataloader_num_workers=CONFIG.num_workers,
        logging_steps=CONFIG.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=CONFIG.eval_steps,
        save_strategy="steps",
        save_steps=CONFIG.save_steps,
        save_total_limit=CONFIG.save_total_limit,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    # 使用动态 mask 的 Collator
    data_collator = DynamicMaskDataCollator(
        tokenizer=tokenizer,
        loss_mode=CONFIG.loss_mode,
        dropout_prob=CONFIG.dropout_prob,
        max_length=CONFIG.max_length
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()
    
    if is_main_process():
        final_path = f"{CONFIG.output_dir}/final-{CONFIG.loss_mode}"
        print(f"\n💾 保存模型到: {final_path}")
        trainer.save_model(final_path)
        processor.save_pretrained(final_path)
        with open(f"{final_path}/train_config.json", "w") as f:
            json.dump(vars(CONFIG), f, indent=2, ensure_ascii=False)
        swanlab.finish()
        print("🎉 训练完成!")

if __name__ == "__main__":
    if "--preprocess" in sys.argv:
        preprocess_data()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("""
train_v3.py - 修复版训练脚本
============================

修复内容:
  1. 精确的 token 边界定位（使用分段 tokenize）
  2. 动态 rationale dropout（在 Collator 中每次决定）

使用:
  python gen_cot_data_v2.py              # 生成数据
  python train_v3.py --preprocess        # 预处理
  torchrun --nproc_per_node=4 train_v3.py --loss_mode full
  torchrun --nproc_per_node=4 train_v3.py --loss_mode answer_only
  torchrun --nproc_per_node=4 train_v3.py --loss_mode rationale_dropout --dropout_prob 0.5
        """)
    else:
        train()