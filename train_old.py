# ╔══════════════════════════════════════════════════════════════════╗
# ║  train_v2.py - 增强版训练脚本                                     ║
# ║  创新点:                                                          ║
# ║    A) Label Mask 策略: full / answer_only / rationale_dropout    ║
# ║    B) 多任务学习: 同时预测 Real/Generated + 生成器来源             ║
# ╚══════════════════════════════════════════════════════════════════╝

# 使用方法:
#   python train_v2.py --preprocess                    # 预处理数据
#   torchrun --nproc_per_node=4 train_v2.py            # 训练 (默认 full loss)
#   torchrun --nproc_per_node=4 train_v2.py --loss_mode answer_only
#   torchrun --nproc_per_node=4 train_v2.py --loss_mode rationale_dropout --dropout_prob 0.5

import os
import sys
import warnings
import time
import re
import random
from datetime import datetime
from tqdm import tqdm

# ========== 环境配置 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*fast processor.*")

import torch
import json
from dataclasses import dataclass, field
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
    HfArgumentParser,
)
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
import swanlab
from swanlab.integration.transformers import SwanLabCallback


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

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        配置区域                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class TrainConfig:
    """训练配置"""
    # 模型路径
    model_path: str = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir_base: str = "/data1/srq/Qwen/Qwen2.5-VL/output/v2-old"
    output_dir: str = ""
    
    # 数据路径 (使用 v2 版本数据)
    train_json: str = os.path.join(DATASET_DIR, "train_v2.json")
    test_json: str = os.path.join(DATASET_DIR, "test_v2.json")
    train_cache: str = "/data1/srq/Qwen/Qwen2.5-VL/cache/train_v2_pt"
    eval_cache: str = "/data1/srq/Qwen/Qwen2.5-VL/cache/eval_v2_pt"
    
    # 模型配置
    max_length: int = 8192
    freeze_vision: bool = True
    use_lora: bool = True
    
    # LoRA 配置 (减小防止过拟合)
    lora_r: int = 32                    # ← 从 64 减到 32
    lora_alpha: int = 64                # ← 2x r
    lora_dropout: float = 0.1           # ← 从 0.05 提高到 0.1
    
    # ═══════════════════════════════════════════════════════════════
    # 创新点 A: Loss Mask 策略
    # ═══════════════════════════════════════════════════════════════
    loss_mode: str = "full"             # full / answer_only / rationale_dropout
    dropout_prob: float = 0.5           # rationale_dropout 模式下的丢弃概率
    
    # 训练超参数
    per_device_batch_size: int = 1
    gradient_accumulation: int = 8
    num_epochs: int = 5                 # ← 从 20 减到 5
    learning_rate: float = 1e-4         # ← 从 2e-4 减到 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    
    # 资源控制
    num_workers: int = 4
    cpu_threads: int = 8
    
    # 评估与保存
    eval_samples: int = 100
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10

# 解析命令行参数
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    config.output_dir = f"{config.output_dir_base}/{timestamp}-{config.loss_mode}"
    return config

CONFIG = parse_args()

# 设置 CPU 线程
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

def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return "N/A"
    info = []
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info.append(f"GPU{i}: {used:.1f}/{total:.1f}GB")
    return " | ".join(info)

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
    print("📋 训练配置 (v2 - 创新版)")
    print("=" * 60)
    print(f"  模型路径:      {CONFIG.model_path}")
    print(f"  输出目录:      {CONFIG.output_dir}")
    print(f"  冻结视觉塔:    {'✅ 是' if CONFIG.freeze_vision else '❌ 否'}")
    print(f"  使用 LoRA:     r={CONFIG.lora_r}, alpha={CONFIG.lora_alpha}")
    print("-" * 60)
    print(f"  🔬 Loss 模式:   {CONFIG.loss_mode}")
    if CONFIG.loss_mode == "rationale_dropout":
        print(f"     Dropout率:  {CONFIG.dropout_prob}")
    print("-" * 60)
    print(f"  Batch Size:    {CONFIG.per_device_batch_size} x 4卡 x {CONFIG.gradient_accumulation}累积")
    print(f"  学习率:        {CONFIG.learning_rate}")
    print(f"  训练轮数:      {CONFIG.num_epochs}")
    print("=" * 60 + "\n")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    自定义回调                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if is_main_process():
            print_banner("🚀 开始训练 (v2)")
            print(f"  📊 总步数: {state.max_steps}")
            print(f"  🔬 Loss 模式: {CONFIG.loss_mode}")
            print()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
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
            if step % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                log_parts.append(f"Mem: {mem:.1f}GB")
            print(f"  {'  |  '.join(log_parts)}")
    
    def on_train_end(self, args, state, control, **kwargs):
        if is_main_process() and self.start_time:
            total_time = time.time() - self.start_time
            print_banner("✅ 训练完成")
            print(f"  ⏱️  总耗时: {format_time(total_time)}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     Data Collator                                ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class QwenVideoDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int = 8192

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if any("pixel_values" in inst for inst in instances):
            pixel_values = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
            image_grid_thw = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            if pixel_values:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)
                batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

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
# ║           创新点 A: Label Mask 策略实现                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def apply_label_mask(response_text: str, response_token_ids: List[int], 
                     tokenizer, loss_mode: str, dropout_prob: float = 0.5) -> List[int]:
    """
    根据 loss_mode 对 response 的 labels 进行 mask
    
    Args:
        response_text: 原始响应文本 (包含 <think>...</think><answer>...</answer><source>...</source>)
        response_token_ids: response 的 token ids
        tokenizer: tokenizer
        loss_mode: "full" / "answer_only" / "rationale_dropout"
        dropout_prob: rationale_dropout 模式下的丢弃概率
    
    Returns:
        masked_labels: mask 后的 labels (-100 表示不计算 loss)
    """
    if loss_mode == "full":
        # 全部计算 loss
        return response_token_ids.copy()
    
    # 找到各部分的位置
    think_start = response_text.find("<think>")
    think_end = response_text.find("</think>")
    answer_start = response_text.find("<answer>")
    answer_end = response_text.find("</answer>")
    source_start = response_text.find("<source>")
    source_end = response_text.find("</source>")
    
    # 如果格式不对，退回到 full 模式
    if think_start == -1 or think_end == -1 or answer_start == -1:
        return response_token_ids.copy()
    
    # 用字符位置估算 token 位置（近似方法）
    # 更精确的方法需要逐段 tokenize，但这里用比例估算
    total_chars = len(response_text)
    total_tokens = len(response_token_ids)
    
    def char_to_token_pos(char_pos):
        return int((char_pos / total_chars) * total_tokens)
    
    think_start_tok = char_to_token_pos(think_start)
    think_end_tok = char_to_token_pos(think_end + len("</think>"))
    answer_start_tok = char_to_token_pos(answer_start)
    answer_end_tok = char_to_token_pos(answer_end + len("</answer>"))
    source_start_tok = char_to_token_pos(source_start) if source_start != -1 else total_tokens
    source_end_tok = char_to_token_pos(source_end + len("</source>")) if source_end != -1 else total_tokens
    
    masked_labels = response_token_ids.copy()
    
    if loss_mode == "answer_only":
        # 只对 <answer> 和 <source> 部分计算 loss，mask 掉 <think>
        for i in range(think_start_tok, min(think_end_tok, len(masked_labels))):
            masked_labels[i] = -100
            
    elif loss_mode == "rationale_dropout":
        # 随机决定是否 mask <think> 部分
        if random.random() < dropout_prob:
            for i in range(think_start_tok, min(think_end_tok, len(masked_labels))):
                masked_labels[i] = -100
    
    return masked_labels

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     数据处理函数                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def process_func(example, processor, tokenizer, loss_mode="full", dropout_prob=0.5):
    """处理单个样本，支持 Label Mask 策略"""
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
    
    # 处理响应
    response = example["conversations"][1]["value"]
    resp_tokens = tokenizer.encode(response, add_special_tokens=False)
    
    # 应用 Label Mask 策略 (创新点 A)
    masked_resp_labels = apply_label_mask(
        response, resp_tokens, tokenizer, 
        loss_mode=loss_mode, dropout_prob=dropout_prob
    )
    
    # 构建完整序列
    input_ids = inputs["input_ids"][0].tolist() + resp_tokens + [tokenizer.pad_token_id]
    labels = [-100] * len(inputs["input_ids"][0]) + masked_resp_labels + [tokenizer.pad_token_id]
    
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    
    if "pixel_values" in inputs:
        result["pixel_values"] = inputs["pixel_values"] 
        result["image_grid_thw"] = inputs["image_grid_thw"]
        
    if "pixel_values_videos" in inputs:
        result["pixel_values_videos"] = inputs["pixel_values_videos"]
        result["video_grid_thw"] = inputs["video_grid_thw"]
    elif "video_pixel_values" in inputs:
        result["pixel_values_videos"] = inputs["video_pixel_values"]
        result["video_grid_thw"] = inputs["video_grid_thw"]
            
    return result

# ╔══════════════════════════════════════════════════════════════════╗
# ║                  自定义 Dataset                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.index_file = os.path.join(cache_dir, "index.json")
        
        with open(self.index_file, "r") as f:
            self.index = json.load(f)
        
        self.length = len(self.index)
        print(f"  加载数据集: {self.length} 样本")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        pt_file = os.path.join(self.cache_dir, self.index[idx])
        return torch.load(pt_file, weights_only=False)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     数据预处理                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def preprocess_data():
    print_banner("📦 数据预处理 (v2)")
    
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path, 
        min_pixels=128*28*28,
        max_pixels=256*28*28,
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
    
    def process_and_save(json_file, cache_dir, desc, max_samples=None):
        index_path = os.path.join(cache_dir, "index.json")
        if os.path.exists(index_path):
            try:
                if os.path.getsize(index_path) > 0:
                    with open(index_path, "r") as f:
                        existing_index = json.load(f)
                    if isinstance(existing_index, list) and len(existing_index) > 0:
                        print(f"✅ 缓存已存在: {cache_dir}")
                        return
            except Exception:
                pass
            print(f"⚠️ 发现无效缓存，重新生成: {cache_dir}")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"📂 加载数据: {json_file}")
        print(f"   样本数: {len(data)}")
        print(f"   Loss 模式: {CONFIG.loss_mode}")
        
        index = []
        failed = []
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(data, desc=desc)):
            try:
                result = process_func(
                    sample, processor, tokenizer,
                    loss_mode=CONFIG.loss_mode,
                    dropout_prob=CONFIG.dropout_prob
                )
                
                pt_filename = f"sample_{i:06d}.pt"
                pt_path = os.path.join(cache_dir, pt_filename)
                torch.save(result, pt_path)
                index.append(pt_filename)
                
            except Exception as e:
                failed.append((i, str(e)))
                print(f"\n⚠️ 样本 {i} 处理失败: {e}")
                continue
        
        with open(os.path.join(cache_dir, "index.json"), "w") as f:
            json.dump(index, f)
        
        elapsed = time.time() - start_time
        print(f"\n✅ 处理完成! 成功: {len(index)}, 失败: {len(failed)}")
        print(f"   耗时: {format_time(elapsed)}")
    
    process_and_save(CONFIG.train_json, CONFIG.train_cache, "处理训练集")
    
    if os.path.exists(CONFIG.test_json):
        process_and_save(CONFIG.test_json, CONFIG.eval_cache, "处理验证集", 
                        max_samples=CONFIG.eval_samples)
    
    print("\n🎉 预处理完成! 运行训练:")
    print(f"   torchrun --nproc_per_node=4 train_v2.py --loss_mode {CONFIG.loss_mode}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     主训练函数                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def train():
    print_config()
    
    print_main("📥 加载模型...")
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path, 
        min_pixels=128*28*28,
        max_pixels=256*28*28,
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
    
    config = AutoConfig.from_pretrained(CONFIG.model_path)
    config._attn_implementation = "sdpa"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG.model_path, 
        torch_dtype=torch.bfloat16,
        config=config,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    
    if CONFIG.freeze_vision:
        print_main("❄️  冻结 Vision Tower")
        for param in model.visual.parameters():
            param.requires_grad = False

    if CONFIG.use_lora:
        print_main("🔧 应用 LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            r=CONFIG.lora_r, 
            lora_alpha=CONFIG.lora_alpha, 
            lora_dropout=CONFIG.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()

    train_index = os.path.join(CONFIG.train_cache, "index.json")
    if not os.path.exists(train_index):
        raise FileNotFoundError(
            f"\n❌ 未找到预处理数据!\n"
            f"   请先运行: python train_v2.py --preprocess\n"
        )
    
    print_main(f"📂 加载预处理数据...")
    train_dataset = VideoDataset(CONFIG.train_cache)
    
    eval_dataset = None
    eval_index = os.path.join(CONFIG.eval_cache, "index.json")
    if os.path.exists(eval_index):
        eval_dataset = VideoDataset(CONFIG.eval_cache)

    callbacks = [
        TrainingMonitorCallback(),
        SwanLabCallback(
            project="Qwen2.5-VL-Video-Detection-v2",
            experiment_name=f"{CONFIG.loss_mode}-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "model": "Qwen2.5-VL-7B",
                "loss_mode": CONFIG.loss_mode,
                "lora_r": CONFIG.lora_r,
                "learning_rate": CONFIG.learning_rate,
            }
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenVideoDataCollator(tokenizer, CONFIG.max_length),
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

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        入口点                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    if "--preprocess" in sys.argv:
        preprocess_data()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("""
Qwen2.5-VL 视频检测训练脚本 v2 (创新版)
========================================

创新点:
  A) Label Mask 策略: 防止模板记忆
  B) 多任务学习: Real/Generated + 生成器归因

使用方法:
  1. 生成 v2 数据 (先运行):
     python gen_cot_data_v2.py

  2. 预处理数据:
     python train_v2.py --preprocess

  3. 训练 (选择 loss 模式):
     torchrun --nproc_per_node=4 train_v2.py --loss_mode full
     torchrun --nproc_per_node=4 train_v2.py --loss_mode answer_only
     torchrun --nproc_per_node=4 train_v2.py --loss_mode rationale_dropout --dropout_prob 0.5

Loss 模式说明:
  - full:              对全部输出计算 loss (baseline)
  - answer_only:       只对 <answer> 和 <source> 计算 loss
  - rationale_dropout: 以 dropout_prob 概率丢弃 <think> 的 loss
        """)
    else:
        train()