# # 查看帮助
# python train.py --help

# # 第一步：预处理数据
# python train.py --preprocess

# # 第二步：4卡训练
# torchrun --nproc_per_node=4 train.py
import os
import sys
import warnings
import time
from datetime import datetime
from tqdm import tqdm
import pickle

# ========== 环境配置（必须放最前面） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 抑制警告
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*fast processor.*")

import torch
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Sequence
from datasets import Dataset, load_from_disk
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
# ║                        配置区域                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class TrainConfig:
    """训练配置 - 集中管理所有参数"""
    # 模型路径
    model_path: str = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "./output/Qwen2.5-VL-Video-SFT"
    
    # 数据路径
    train_json: str = "train.json"
    test_json: str = "test.json"
    train_cache: str = "./cache/train_pt"      # 改为 pt 格式
    eval_cache: str = "./cache/eval_pt"
    
    # 模型配置
    max_length: int = 8192              # ← 增大，避免截断视频 token
    freeze_vision: bool = True
    use_lora: bool = True
    
    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # ═══════════════════════════════════════════════════════════════
    # 训练超参数 - 优化配置
    # ═══════════════════════════════════════════════════════════════
    per_device_batch_size: int = 1      # ← 改为 1，视频必须单样本
    gradient_accumulation: int = 8       # ← 增加到 8，有效 batch = 1×4×8=32
    num_epochs: int = 20
    learning_rate: float = 2e-4          # ← 稍微提高
    weight_decay: float = 0.01
    warmup_steps: int = 100              # ← 用 warmup_steps 替代 warmup_ratio
    
    # ═══════════════════════════════════════════════════════════════
    # 资源控制 - 关键优化
    # ═══════════════════════════════════════════════════════════════
    num_workers: int = 8                 # ← 改为 0！避免多进程开销
    cpu_threads: int = 8                 # ← 降低
    
    # 评估与保存
    eval_samples: int = 50
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10

CONFIG = TrainConfig()

# 设置 CPU 线程
os.environ["OMP_NUM_THREADS"] = str(CONFIG.cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(CONFIG.cpu_threads)
torch.set_num_threads(CONFIG.cpu_threads)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        工具函数                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_rank():
    """获取当前进程 rank"""
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    """是否为主进程"""
    return get_rank() == 0

def print_main(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)

def format_time(seconds):
    """格式化时间"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_gpu_memory_info():
    """获取 GPU 显存信息"""
    if not torch.cuda.is_available():
        return "N/A"
    
    info = []
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info.append(f"GPU{i}: {used:.1f}/{total:.1f}GB")
    return " | ".join(info)

def print_banner(text, char="═", width=60):
    """打印美观的横幅"""
    if not is_main_process():
        return
    border = char * width
    padding = (width - len(text) - 2) // 2
    print(f"\n╔{border}╗")
    print(f"║{' ' * padding}{text}{' ' * (width - padding - len(text))}║")
    print(f"╚{border}╝\n")

def print_config():
    """打印配置信息"""
    if not is_main_process():
        return
    
    print("\n" + "=" * 60)
    print("📋 训练配置")
    print("=" * 60)
    print(f"  模型路径:      {CONFIG.model_path}")
    print(f"  输出目录:      {CONFIG.output_dir}")
    print(f"  冻结视觉塔:    {'✅ 是' if CONFIG.freeze_vision else '❌ 否'}")
    print(f"  使用 LoRA:     {'✅ 是' if CONFIG.use_lora else '❌ 否'}")
    if CONFIG.use_lora:
        print(f"    - r={CONFIG.lora_r}, alpha={CONFIG.lora_alpha}")
    print("-" * 60)
    print(f"  Batch Size:    {CONFIG.per_device_batch_size} x 4卡 x {CONFIG.gradient_accumulation}累积 = {CONFIG.per_device_batch_size * 4 * CONFIG.gradient_accumulation}")
    print(f"  学习率:        {CONFIG.learning_rate}")
    print(f"  训练轮数:      {CONFIG.num_epochs}")
    print(f"  Warmup:        {CONFIG.warmup_steps}")
    print("-" * 60)
    print(f"  DataLoader:    {CONFIG.num_workers} workers")
    print(f"  CPU 线程:      {CONFIG.cpu_threads}")
    print("=" * 60 + "\n")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    自定义回调 - 训练监控                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class TrainingMonitorCallback(TrainerCallback):
    """训练过程监控回调"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if is_main_process():
            print_banner("🚀 开始训练")
            print(f"  📊 总步数: {state.max_steps}")
            print(f"  📈 总样本: {state.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps * 4}")
            print(f"  🖥️  显存: {get_gpu_memory_info()}")
            print()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        if is_main_process():
            epoch = int(state.epoch) + 1 if state.epoch else 1
            print(f"\n{'─' * 50}")
            print(f"📅 Epoch {epoch}/{args.num_train_epochs}")
            print(f"{'─' * 50}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if is_main_process() and self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            print(f"  ⏱️  Epoch 耗时: {format_time(epoch_time)}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if is_main_process() and logs:
            step = state.global_step
            
            # 构建日志行
            log_parts = [f"Step {step:5d}"]
            
            if "loss" in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")
            if "learning_rate" in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")
            if "eval_loss" in logs:
                log_parts.append(f"Eval Loss: {logs['eval_loss']:.4f}")
            
            # 添加显存信息（每100步）
            if step % 100 == 0:
                mem_used = torch.cuda.max_memory_allocated() / 1024**3
                log_parts.append(f"Mem: {mem_used:.1f}GB")
            
            print(f"  {'  |  '.join(log_parts)}")
    
    def on_save(self, args, state, control, **kwargs):
        if is_main_process():
            print(f"  💾 模型已保存 (Step {state.global_step})")
    
    def on_train_end(self, args, state, control, **kwargs):
        if is_main_process() and self.start_time:
            total_time = time.time() - self.start_time
            print_banner("✅ 训练完成")
            print(f"  ⏱️  总耗时: {format_time(total_time)}")
            print(f"  📊 最终 Loss: {state.log_history[-1].get('loss', 'N/A')}")
            print(f"  🖥️  峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
            print()

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     Data Collator                                ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class QwenVideoDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int = 8192              # ← 增大

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 不截断 input_ids，避免 token/feature 不匹配
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        # 移除截断，或者设置一个很大的值
        # input_ids = input_ids[:, :self.max_length]  # 注释掉或删除
        # labels = labels[:, :self.max_length]        # 注释掉或删除
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        # 处理图像数据
        if any("pixel_values" in inst for inst in instances):
            pixel_values = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
            image_grid_thw = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            if pixel_values:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)
                batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        # 处理视频数据
        video_key = next(
            (k for k in ["pixel_values_videos", "video_pixel_values"] 
             if any(k in inst for inst in instances)), 
            None
        )
        if video_key:
            pv_videos = [inst[video_key] for inst in instances if video_key in inst]
            video_grid_thw = [inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst]
            if pv_videos:
                batch["pixel_values_videos"] = torch.cat(pv_videos, dim=0)
                batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        return batch

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     数据处理函数                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def process_func(example, processor, tokenizer):
    """处理单个样本"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": example["conversations"][0]["value"]},
                {"type": "text", "text": "Analyze the video. Is it Real or Generated?"}
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
    
    # 构建完整序列
    input_ids = inputs["input_ids"][0].tolist() + resp_tokens + [tokenizer.pad_token_id]
    labels = [-100] * len(inputs["input_ids"][0]) + resp_tokens + [tokenizer.pad_token_id]
    
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    
    # 添加视觉特征
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
# ║                  自定义 Dataset（从 .pt 加载）                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class VideoDataset(torch.utils.data.Dataset):
    """从预处理的 .pt 文件加载数据"""
    
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
# ║                     数据预处理（保存为 .pt）                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def preprocess_data():
    """单独预处理数据，保存为单独的 .pt 文件"""
    print_banner("📦 数据预处理")
    
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path, 
        # ═══════════════════════════════════════════════════════════════
        # 关键修改：大幅降低视频分辨率
        # ═══════════════════════════════════════════════════════════════
        min_pixels=128*28*28,       # ← 降低：128×28×28 = 100,352
        max_pixels=256*28*28,       # ← 大幅降低：256×28×28 = 200,704（原来 1280×28×28）
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
    
    def process_and_save(json_file, cache_dir, desc, max_samples=None):
        """处理数据并保存为 .pt 文件"""
        if os.path.exists(os.path.join(cache_dir, "index.json")):
            print(f"✅ 缓存已存在: {cache_dir}")
            return
        
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"📂 加载数据: {json_file}")
        print(f"   样本数: {len(data)}")
        print("🔄 处理中...")
        
        index = []
        failed = []
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(data, desc=desc)):
            try:
                result = process_func(sample, processor, tokenizer)
                
                # 保存为单独的 .pt 文件
                pt_filename = f"sample_{i:06d}.pt"
                pt_path = os.path.join(cache_dir, pt_filename)
                torch.save(result, pt_path)
                index.append(pt_filename)
                
            except Exception as e:
                failed.append((i, str(e)))
                print(f"\n⚠️ 样本 {i} 处理失败: {e}")
                continue
        
        # 保存索引文件
        with open(os.path.join(cache_dir, "index.json"), "w") as f:
            json.dump(index, f)
        
        elapsed = time.time() - start_time
        print(f"\n✅ 处理完成!")
        print(f"   成功: {len(index)}, 失败: {len(failed)}")
        print(f"   耗时: {format_time(elapsed)}")
        print(f"   保存到: {cache_dir}")
        
        if failed:
            with open(os.path.join(cache_dir, "failed.json"), "w") as f:
                json.dump(failed, f, indent=2)
            print(f"   失败记录: {os.path.join(cache_dir, 'failed.json')}")
    
    # 处理训练集
    process_and_save(CONFIG.train_json, CONFIG.train_cache, "处理训练集")
    
    # 处理验证集
    if os.path.exists(CONFIG.test_json):
        process_and_save(CONFIG.test_json, CONFIG.eval_cache, "处理验证集", 
                        max_samples=CONFIG.eval_samples)
    
    print("\n" + "=" * 50)
    print("🎉 预处理完成! 现在可以运行训练:")
    print("   torchrun --nproc_per_node=4 train.py")
    print("=" * 50)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     主训练函数                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def train():
    """主训练流程"""
    
    # 打印配置（只在主进程）
    print_config()
    
    # 加载 tokenizer 和 processor
    print_main("📥 加载 Processor 和 Tokenizer...")
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path, 
        min_pixels=128*28*28,       # ← 保持一致
        max_pixels=256*28*28,       # ← 保持一致
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)
    
    # 加载模型
    print_main("📥 加载模型...")
    config = AutoConfig.from_pretrained(CONFIG.model_path)
    config._attn_implementation = "sdpa"  # 使用高效注意力
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG.model_path, 
        torch_dtype=torch.bfloat16,
        config=config,
        device_map=None,
        # 显存优化
        low_cpu_mem_usage=True,
    )
    
    # 冻结视觉塔
    if CONFIG.freeze_vision:
        print_main("❄️  冻结 Vision Tower (节省 ~30% 显存)")
        for param in model.visual.parameters():
            param.requires_grad = False

    # 应用 LoRA
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

    # 检查缓存
    train_index = os.path.join(CONFIG.train_cache, "index.json")
    if not os.path.exists(train_index):
        raise FileNotFoundError(
            f"\n❌ 未找到预处理数据!\n"
            f"   请先运行: python train.py --preprocess\n"
        )
    
    # 加载数据集
    print_main(f"📂 加载预处理数据...")
    train_dataset = VideoDataset(CONFIG.train_cache)
    
    eval_dataset = None
    eval_index = os.path.join(CONFIG.eval_cache, "index.json")
    if os.path.exists(eval_index):
        eval_dataset = VideoDataset(CONFIG.eval_cache)

    # 回调函数
    callbacks = [
        TrainingMonitorCallback(),
        SwanLabCallback(
            project="Qwen2.5-VL-Video-Detection",
            experiment_name=f"SFT-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "model": "Qwen2.5-VL-7B",
                "freeze_vision": CONFIG.freeze_vision,
                "lora_r": CONFIG.lora_r,
                "batch_size": CONFIG.per_device_batch_size * 4 * CONFIG.gradient_accumulation,
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
        warmup_steps=CONFIG.warmup_steps,    # ← 用 warmup_steps
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,          # ← 改为 False
        dataloader_num_workers=CONFIG.num_workers,             # ← 改为 0
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
    
    # 保存最终模型
    if is_main_process():
        final_path = f"{CONFIG.output_dir}/final"
        print(f"\n💾 保存最终模型到: {final_path}")
        trainer.save_model(final_path)
        processor.save_pretrained(final_path)
        
        # 保存训练配置
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
Qwen2.5-VL 视频检测训练脚本
===========================

使用方法:
  1. 预处理数据 (首次运行):
     python train.py --preprocess

  2. 开始训练 (4卡):
     torchrun --nproc_per_node=4 train.py

  3. 单卡训练 (调试):
     python train.py

选项:
  --preprocess    预处理数据集
  --help, -h      显示帮助信息
        """)
    else:
        train()