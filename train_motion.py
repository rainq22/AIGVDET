import os
import sys
import warnings
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"
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
import swanlab
from swanlab.integration.transformers import SwanLabCallback

from models.dual_stream_qwen import DualStreamQwenDeepfake


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


@dataclass
class TrainConfig:
    model_path: str = "/data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir_base: str = "/data1/srq/Qwen/Qwen2.5-VL/output/motion"
    output_dir: str = ""

    train_json: str = os.path.join(DATASET_DIR, "train_v2.json")
    test_json: str = os.path.join(DATASET_DIR, "test_v2.json")
    train_cache: str = "/data1/srq/Qwen/Qwen2.5-VL/cache/train_v3_motion_pt"
    eval_cache: str = "/data1/srq/Qwen/Qwen2.5-VL/cache/eval_v3_motion_pt"
    flow_dir: str = "/data1/srq/Qwen/Qwen2.5-VL/cache/flow_features"

    dataset_dir: str = ""

    video_reader: str = "torchvision"
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 256 * 28 * 28
    video_max_frames: int = 0

    max_length: int = 8192
    freeze_vision: bool = True
    use_lora: bool = True

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    loss_mode: str = "full"
    dropout_prob: float = 0.5

    per_device_batch_size: int = 1
    gradient_accumulation: int = 8
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50

    num_workers: int = 4
    cpu_threads: int = 8
    preprocess_workers: int = 1
    preprocess_save_every: int = 200

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
        elif args[i] == "--preprocess_workers" and i + 1 < len(args):
            config.preprocess_workers = int(args[i + 1])
            i += 2
        elif args[i] == "--preprocess_save_every" and i + 1 < len(args):
            config.preprocess_save_every = int(args[i + 1])
            i += 2
        elif args[i] == "--video_reader" and i + 1 < len(args):
            config.video_reader = args[i + 1]
            i += 2
        elif args[i] == "--min_pixels" and i + 1 < len(args):
            config.min_pixels = int(args[i + 1])
            i += 2
        elif args[i] == "--max_pixels" and i + 1 < len(args):
            config.max_pixels = int(args[i + 1])
            i += 2
        elif args[i] == "--video_max_frames" and i + 1 < len(args):
            config.video_max_frames = int(args[i + 1])
            i += 2
        elif args[i] == "--cpu_threads" and i + 1 < len(args):
            config.cpu_threads = int(args[i + 1])
            i += 2
        elif args[i] == "--dataset_dir" and i + 1 < len(args):
            config.dataset_dir = args[i + 1]
            i += 2
        elif args[i] in ["--help", "-h"]:
            i += 1
        else:
            i += 1
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    config.output_dir = f"{config.output_dir_base}/{timestamp}-{config.loss_mode}"
    return config

CONFIG = parse_args()

if CONFIG.dataset_dir:
    CONFIG.train_json = os.path.join(CONFIG.dataset_dir, "train_v2.json")
    CONFIG.test_json = os.path.join(CONFIG.dataset_dir, "test_v2.json")

os.environ["FORCE_QWENVL_VIDEO_READER"] = CONFIG.video_reader

from qwen_vl_utils import process_vision_info

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


def print_banner(text, char="=", width=60):
    if not is_main_process():
        return
    border = char * width
    padding = (width - len(text) - 2) // 2
    print(f"\n[{border}]")
    print(f"[{(' ' * padding) + text + (' ' * (width - padding - len(text)))}]")
    print(f"[{border}]\n")


def print_config():
    if not is_main_process():
        return
    print("\n" + "=" * 60)
    print("Train config (motion)")
    print("=" * 60)
    print(f"model_path: {CONFIG.model_path}")
    print(f"output_dir: {CONFIG.output_dir}")
    print(f"flow_dir:   {CONFIG.flow_dir}")
    print(f"loss_mode:  {CONFIG.loss_mode}")
    print("=" * 60 + "\n")


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if is_main_process():
            print_banner("Start training")
            print(f"steps: {state.max_steps}")
            print(f"loss_mode: {CONFIG.loss_mode}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        if is_main_process():
            epoch = int(state.epoch) + 1 if state.epoch else 1
            print(f"\nEpoch {epoch}/{args.num_train_epochs}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if is_main_process() and self.epoch_start_time:
            print(f"epoch time: {format_time(time.time() - self.epoch_start_time)}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if is_main_process() and logs:
            step = state.global_step
            parts = [f"step {step:5d}"]
            if "loss" in logs:
                parts.append(f"loss {logs['loss']:.4f}")
            if "learning_rate" in logs:
                parts.append(f"lr {logs['learning_rate']:.2e}")
            if "eval_loss" in logs:
                parts.append(f"eval {logs['eval_loss']:.4f}")
            print(" | ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        if is_main_process() and self.start_time:
            print_banner("Training done")
            print(f"total time: {format_time(time.time() - self.start_time)}")


def apply_label_mask(response_text: str, response_token_ids: List[int],
                     tokenizer, loss_mode: str, dropout_prob: float = 0.5) -> List[int]:
    if loss_mode == "full":
        return response_token_ids.copy()

    think_start = response_text.find("<think>")
    think_end = response_text.find("</think>")

    if think_start == -1 or think_end == -1:
        return response_token_ids.copy()

    total_chars = len(response_text)
    total_tokens = len(response_token_ids)

    def char_to_token_pos(char_pos):
        return int((char_pos / total_chars) * total_tokens)

    think_start_tok = char_to_token_pos(think_start)
    think_end_tok = char_to_token_pos(think_end + len("</think>"))

    masked_labels = response_token_ids.copy()

    if loss_mode == "answer_only":
        for i in range(think_start_tok, min(think_end_tok, len(masked_labels))):
            masked_labels[i] = -100
    elif loss_mode == "rationale_dropout":
        if random.random() < dropout_prob:
            for i in range(think_start_tok, min(think_end_tok, len(masked_labels))):
                masked_labels[i] = -100

    return masked_labels


def process_func(example, processor, tokenizer, loss_mode="full", dropout_prob=0.5):
    video_item = {"type": "video", "video": example["conversations"][0]["value"]}
    if CONFIG.video_max_frames and CONFIG.video_max_frames > 0:
        video_item["max_frames"] = CONFIG.video_max_frames

    messages = [
        {
            "role": "user",
            "content": [
                video_item,
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

    masked_resp_labels = apply_label_mask(
        response, resp_tokens, tokenizer,
        loss_mode=loss_mode, dropout_prob=dropout_prob
    )

    input_ids = inputs["input_ids"][0].tolist() + resp_tokens + [tokenizer.pad_token_id]
    labels = [-100] * len(inputs["input_ids"][0]) + masked_resp_labels + [tokenizer.pad_token_id]

    sample_id = example.get("id") or os.path.splitext(os.path.basename(example["conversations"][0]["value"]))[0]
    flow_path = os.path.join(CONFIG.flow_dir, f"{sample_id}_flow.pt")
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow residual not found: {flow_path}")

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flow_residuals": torch.load(flow_path),
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
        self.index_file = os.path.join(cache_dir, "index.json")

        with open(self.index_file, "r") as f:
            self.index = json.load(f)

        self.length = len(self.index)
        print(f"Loaded dataset: {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pt_file = os.path.join(self.cache_dir, self.index[idx])
        return torch.load(pt_file, weights_only=False)


@dataclass
class QwenVideoDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

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

        if any("flow_residuals" in inst for inst in instances):
            flows = [inst["flow_residuals"] for inst in instances]
            batch["flow_residuals"] = torch.stack(flows, dim=0)

        return batch


def preprocess_data():
    print_banner("Preprocess data")

    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path,
        min_pixels=CONFIG.min_pixels,
        max_pixels=CONFIG.max_pixels,
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)

    def process_and_save(json_file, cache_dir, desc, max_samples=None):
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, "index.json")

        existing_index = []
        if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
            try:
                with open(index_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_index = loaded
            except Exception:
                existing_index = []

        existing_index = [
            name for name in existing_index
            if os.path.exists(os.path.join(cache_dir, name))
        ]
        existing_set = set(existing_index)
        if existing_index:
            print(f"Resume cache: {cache_dir} ({len(existing_index)} samples)")

        with open(json_file, "r") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        print(f"Load data: {json_file}")
        print(f"samples: {len(data)}")
        print(f"loss_mode: {CONFIG.loss_mode}")

        index = list(existing_index)
        index_set = set(existing_index)
        failed = []
        start_time = time.time()

        def process_one(i, sample):
            pt_filename = f"sample_{i:06d}.pt"
            pt_path = os.path.join(cache_dir, pt_filename)
            if pt_filename in existing_set and os.path.exists(pt_path):
                return "skip", pt_filename, None
            try:
                with torch.inference_mode():
                    result = process_func(
                        sample, processor, tokenizer,
                        loss_mode=CONFIG.loss_mode,
                        dropout_prob=CONFIG.dropout_prob
                    )
                torch.save(result, pt_path)
                return "ok", pt_filename, None
            except Exception as exc:
                return "fail", pt_filename, str(exc)

        def maybe_flush():
            if CONFIG.preprocess_save_every and len(index) % CONFIG.preprocess_save_every == 0:
                with open(index_path, "w") as f:
                    json.dump(index, f)

        if CONFIG.preprocess_workers > 1:
            max_workers = max(1, CONFIG.preprocess_workers)
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, sample in enumerate(data):
                    futures.append(executor.submit(process_one, i, sample))
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    status, pt_filename, err = fut.result()
                    if status in {"ok", "skip"}:
                        if pt_filename not in index_set:
                            index.append(pt_filename)
                            index_set.add(pt_filename)
                            maybe_flush()
                    else:
                        failed.append((pt_filename, err))
                        print(f"Sample {pt_filename} failed: {err}")
        else:
            for i, sample in enumerate(tqdm(data, desc=desc)):
                status, pt_filename, err = process_one(i, sample)
                if status in {"ok", "skip"}:
                    if pt_filename not in index_set:
                        index.append(pt_filename)
                        index_set.add(pt_filename)
                        maybe_flush()
                else:
                    failed.append((i, err))
                    print(f"Sample {i} failed: {err}")

        with open(index_path, "w") as f:
            json.dump(index, f)

        elapsed = time.time() - start_time
        print(f"Done. Success: {len(index)}, Failed: {len(failed)}, Time: {format_time(elapsed)}")

    process_and_save(CONFIG.train_json, CONFIG.train_cache, "train")

    if os.path.exists(CONFIG.test_json):
        process_and_save(CONFIG.test_json, CONFIG.eval_cache, "eval", max_samples=CONFIG.eval_samples)

    print("Preprocess done")


def train():
    print_config()

    print_main("Load model...")
    processor = AutoProcessor.from_pretrained(
        CONFIG.model_path,
        min_pixels=CONFIG.min_pixels,
        max_pixels=CONFIG.max_pixels,
        padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_path)

    model = DualStreamQwenDeepfake(CONFIG.model_path)

    if CONFIG.freeze_vision:
        print_main("Freeze vision tower")
        for param in model.qwen.visual.parameters():
            param.requires_grad = False

    if CONFIG.use_lora:
        print_main("Apply LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=CONFIG.lora_r,
            lora_alpha=CONFIG.lora_alpha,
            lora_dropout=CONFIG.lora_dropout,
            bias="none",
        )
        model.qwen = get_peft_model(model.qwen, lora_config)
        if is_main_process():
            model.qwen.print_trainable_parameters()

    train_index = os.path.join(CONFIG.train_cache, "index.json")
    if not os.path.exists(train_index):
        raise FileNotFoundError("Please run: python train_motion.py --preprocess")

    train_dataset = VideoDataset(CONFIG.train_cache)
    eval_dataset = None
    eval_index = os.path.join(CONFIG.eval_cache, "index.json")
    if os.path.exists(eval_index):
        eval_dataset = VideoDataset(CONFIG.eval_cache)

    callbacks = [
        TrainingMonitorCallback(),
        SwanLabCallback(
            project="Qwen2.5-VL-Video-Detection-motion",
            experiment_name=f"{CONFIG.loss_mode}-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "loss_mode": CONFIG.loss_mode,
                "lora_r": CONFIG.lora_r,
                "learning_rate": CONFIG.learning_rate,
            },
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
        data_collator=QwenVideoDataCollator(tokenizer),
        callbacks=callbacks,
    )

    trainer.train()

    if is_main_process():
        final_path = f"{CONFIG.output_dir}/final-{CONFIG.loss_mode}"
        print(f"Save model to: {final_path}")
        trainer.save_model(final_path)
        processor.save_pretrained(final_path)
        with open(f"{final_path}/train_config.json", "w") as f:
            json.dump(vars(CONFIG), f, indent=2, ensure_ascii=False)
        swanlab.finish()
        print("Done")


def main():
    if "--preprocess" in sys.argv:
        preprocess_data()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("""
train_motion.py

Usage:
  python train_motion.py --preprocess [--cpu_threads 16]
  torchrun --nproc_per_node=4 train_motion.py --loss_mode full
  torchrun --nproc_per_node=4 train_motion.py --loss_mode answer_only
  torchrun --nproc_per_node=4 train_motion.py --loss_mode rationale_dropout --dropout_prob 0.5
""")
    else:
        train()


if __name__ == "__main__":
    main()
