# ╔══════════════════════════════════════════════════════════════════╗
# ║  train_v4.py - 改进版训练脚本                                     ║
# ║  改进:                                                           ║
# ║    1) 统一使用权重式 loss (支持 answer_only / rationale_dropout)  ║
# ║    2) 预处理阶段缓存 token 范围，减少训练时重复 tokenize          ║
# ║    3) 提示词多样化，降低 prompt 记忆                              ║
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
import torch.nn.functional as F
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
)
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
import swanlab
from swanlab.integration.transformers import SwanLabCallback


@dataclass
class TrainConfig:
    model_path: str = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "./output/Qwen2.5-VL-Video-SFT-v4"

    train_json: str = "train_v4.json"
    test_json: str = "test_v4.json"
    train_cache: str = "./cache/train_v4_pt"
    eval_cache: str = "./cache/eval_v4_pt"

    max_length: int = 8192
    freeze_vision: bool = True
    use_lora: bool = True

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    loss_mode: str = "full"  # full / answer_only / rationale_dropout / rationale_weighted
    dropout_prob: float = 0.5
    think_weight: float = 0.3

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

    seed: int = 42
    prompt_templates: List[str] = field(
        default_factory=lambda: [
            "Analyze the video. Is it Real or Generated? Also identify the source.",
            "Determine whether the video is real or generated, and name the generator if any.",
            "Judge if this video is authentic or AI-generated, then provide the source.",
            "Inspect the video for authenticity and specify the generator if it is fake.",
        ]
    )


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
        elif args[i] == "--think_weight" and i + 1 < len(args):
            config.think_weight = float(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            config.seed = int(args[i + 1])
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
    print("训练配置 (v4 - 改进版)")
    print("=" * 60)
    print(f"  模型路径:      {CONFIG.model_path}")
    print(f"  输出目录:      {CONFIG.output_dir}")
    print(f"  使用 LoRA:     r={CONFIG.lora_r}, alpha={CONFIG.lora_alpha}")
    print("-" * 60)
    print(f"  Loss 模式:     {CONFIG.loss_mode}")
    if CONFIG.loss_mode == "rationale_dropout":
        print(f"  Dropout率:     {CONFIG.dropout_prob} (动态)")
    if CONFIG.loss_mode == "rationale_weighted":
        print(f"  Think 权重:    {CONFIG.think_weight}")
    print("-" * 60)
    print(f"  Batch Size:    {CONFIG.per_device_batch_size} x 4卡 x {CONFIG.gradient_accumulation}累积")
    print(f"  学习率:        {CONFIG.learning_rate}")
    print(f"  训练轮数:      {CONFIG.num_epochs}")
    print("=" * 60 + "\n")


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if is_main_process():
            print_banner("开始训练 (v4)")
            print(f"  总步数: {state.max_steps}")
            print(f"  Loss 模式: {CONFIG.loss_mode}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        if is_main_process():
            epoch = int(state.epoch) + 1 if state.epoch else 1
            print(f"\n{'-' * 50}")
            print(f"Epoch {epoch}/{args.num_train_epochs}")
            print(f"{'-' * 50}")

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
            print_banner("训练完成")
            print(f"  总耗时: {format_time(time.time() - self.start_time)}")


def get_segment_token_ranges(response_text: str, tokenizer) -> dict:
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    source_match = re.search(r"<source>(.*?)</source>", response_text, re.DOTALL)

    segments = []
    if think_match:
        segments.append(("pre_think", response_text[:think_match.start()]))
        segments.append(("think", response_text[think_match.start():think_match.end()]))
    if answer_match:
        if think_match:
            segments.append(("between1", response_text[think_match.end():answer_match.start()]))
        segments.append(("answer", response_text[answer_match.start():answer_match.end()]))
    if source_match:
        if answer_match:
            segments.append(("between2", response_text[answer_match.end():source_match.start()]))
        segments.append(("source", response_text[source_match.start():source_match.end()]))

    ranges = {}
    current_pos = 0
    for seg_name, seg_text in segments:
        if not seg_text:
            continue
        seg_tokens = tokenizer.encode(seg_text, add_special_tokens=False)
        seg_len = len(seg_tokens)
        if seg_name in ["think", "answer", "source"]:
            ranges[seg_name] = (current_pos, current_pos + seg_len)
        current_pos += seg_len

    return ranges


@dataclass
class DynamicWeightedDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    loss_mode: str = "full"
    dropout_prob: float = 0.5
    think_weight: float = 0.3
    max_length: int = 8192

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        weights_list = []

        for inst in instances:
            input_ids = inst["input_ids"]
            labels = inst["labels"]
            weights = torch.zeros_like(labels, dtype=torch.float)
            weights[labels != -100] = 1.0

            response_start = inst.get("response_start", 0)
            ranges = inst.get("response_segment_ranges")
            if ranges is None and "response_text" in inst:
                ranges = get_segment_token_ranges(inst["response_text"], self.tokenizer)

            if ranges and "think" in ranges:
                think_start = response_start + ranges["think"][0]
                think_end = response_start + ranges["think"][1]

                if self.loss_mode == "answer_only":
                    weights[think_start:think_end] = 0.0
                elif self.loss_mode == "rationale_dropout":
                    if random.random() < self.dropout_prob:
                        weights[think_start:think_end] = 0.0
                elif self.loss_mode == "rationale_weighted":
                    weights[think_start:think_end] = self.think_weight

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            weights_list.append(weights)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        loss_weights = torch.nn.utils.rnn.pad_sequence(
            weights_list, batch_first=True, padding_value=0.0
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

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


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is None or loss_weights is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        flat_weights = shift_weights.view(-1)
        flat_labels = shift_labels.view(-1)
        valid = flat_labels != -100
        weighted_loss = loss * flat_weights
        denom = flat_weights[valid].sum().clamp_min(1.0)
        loss = weighted_loss[valid].sum() / denom

        return (loss, outputs) if return_outputs else loss


def process_func(example, processor, tokenizer):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": example["conversations"][0]["value"]},
                {"type": "text", "text": random.choice(CONFIG.prompt_templates)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )

    response = example["conversations"][1]["value"]
    resp_tokens = tokenizer.encode(response, add_special_tokens=False)
    prompt_len = len(inputs["input_ids"][0])

    input_ids = inputs["input_ids"][0].tolist() + resp_tokens + [tokenizer.pad_token_id]
    labels = [-100] * prompt_len + resp_tokens + [tokenizer.pad_token_id]

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_text": response,
        "response_start": prompt_len,
        "response_segment_ranges": get_segment_token_ranges(response, tokenizer),
    }

    if "pixel_values_videos" in inputs:
        result["pixel_values_videos"] = inputs["pixel_values_videos"]
        result["video_grid_thw"] = inputs["video_grid_thw"]
    elif "video_pixel_values" in inputs:
        result["pixel_values_videos"] = inputs["video_pixel_values"]
        result["video_grid_thw"] = inputs["video_grid_thw"]

    return result


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


def preprocess_data():
    print_banner("数据预处理 (v4)")

    random.seed(CONFIG.seed)

    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
        padding_side="right",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)

    def process_and_save(json_file, cache_dir, desc, max_samples=None):
        if os.path.exists(os.path.join(cache_dir, "index.json")):
            print(f"缓存已存在: {cache_dir}")
            return

        os.makedirs(cache_dir, exist_ok=True)

        with open(json_file, "r") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        print(f"加载数据: {json_file}, 样本数: {len(data)}")

        index = []
        failed = []

        for i, sample in enumerate(tqdm(data, desc=desc)):
            try:
                result = process_func(sample, processor, tokenizer)
                pt_filename = f"sample_{i:06d}.pt"
                torch.save(result, os.path.join(cache_dir, pt_filename))
                index.append(pt_filename)
            except Exception as e:
                failed.append((i, str(e)))
                print(f"\n样本 {i} 失败: {e}")

        with open(os.path.join(cache_dir, "index.json"), "w") as f:
            json.dump(index, f)

        print(f"完成: 成功 {len(index)}, 失败 {len(failed)}")

    process_and_save(CONFIG.train_json, CONFIG.train_cache, "处理训练集")
    if os.path.exists(CONFIG.test_json):
        process_and_save(CONFIG.test_json, CONFIG.eval_cache, "处理验证集", CONFIG.eval_samples)

    print(f"预处理完成! 运行: torchrun --nproc_per_node=4 train_v4.py --loss_mode {CONFIG.loss_mode}")


def train():
    print_config()

    print_main("加载模型...")
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
        padding_side="right",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)

    config = AutoConfig.from_pretrained(CONFIG.model_path)
    config._attn_implementation = "sdpa"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG.model_path, torch_dtype=torch.bfloat16, config=config, device_map=None, low_cpu_mem_usage=True
    )

    if CONFIG.freeze_vision:
        print_main("冻结 Vision Tower")
        for param in model.visual.parameters():
            param.requires_grad = False

    if CONFIG.use_lora:
        print_main("应用 LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=CONFIG.lora_r,
            lora_alpha=CONFIG.lora_alpha,
            lora_dropout=CONFIG.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()

    if not os.path.exists(os.path.join(CONFIG.train_cache, "index.json")):
        raise FileNotFoundError("请先运行: python train_v4.py --preprocess")

    print_main("加载数据...")
    train_dataset = VideoDataset(CONFIG.train_cache)
    eval_dataset = VideoDataset(CONFIG.eval_cache) if os.path.exists(os.path.join(CONFIG.eval_cache, "index.json")) else None

    callbacks = [
        TrainingMonitorCallback(),
        SwanLabCallback(
            project="Qwen2.5-VL-Video-Detection-v4",
            experiment_name=f"{CONFIG.loss_mode}-{datetime.now().strftime('%m%d-%H%M')}",
            config={"loss_mode": CONFIG.loss_mode, "lora_r": CONFIG.lora_r, "lr": CONFIG.learning_rate},
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
        seed=CONFIG.seed,
    )

    data_collator = DynamicWeightedDataCollator(
        tokenizer=tokenizer,
        loss_mode=CONFIG.loss_mode,
        dropout_prob=CONFIG.dropout_prob,
        think_weight=CONFIG.think_weight,
        max_length=CONFIG.max_length,
    )

    trainer = WeightedLossTrainer(
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
        print(f"保存模型到: {final_path}")
        trainer.save_model(final_path)
        processor.save_pretrained(final_path)
        with open(f"{final_path}/train_config.json", "w") as f:
            json.dump(vars(CONFIG), f, indent=2, ensure_ascii=False)
        swanlab.finish()
        print("训练完成")


if __name__ == "__main__":
    if "--preprocess" in sys.argv:
        preprocess_data()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
train_v4.py - 改进版训练脚本
==========================

改进内容:
  1) 统一权重式 loss (answer_only / rationale_dropout / rationale_weighted)
  2) 预处理缓存 token 范围，减少训练时重复 tokenize
  3) 提示词多样化，降低 prompt 记忆

使用:
  python gen_cot_data_v4.py
  python train_v4.py --preprocess
  torchrun --nproc_per_node=4 train_v4.py --loss_mode full
  torchrun --nproc_per_node=4 train_v4.py --loss_mode answer_only
  torchrun --nproc_per_node=4 train_v4.py --loss_mode rationale_dropout --dropout_prob 0.5
  torchrun --nproc_per_node=4 train_v4.py --loss_mode rationale_weighted --think_weight 0.3
            """
        )
    else:
        train()
