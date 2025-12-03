import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration, # 替换为 Qwen2.5
    AutoProcessor,
)
import swanlab
import json
import os

# --- 配置部分 ---
# 根据你的文件结构，修改模型路径为相对路径或绝对路径
# 你的结构: Qwen/Qwen2.5-VL/train.py (当前位置) -> Qwen/Qwen/Qwen2.5-VL-7B-Instruct (模型位置)
local_model_path = "../Qwen/Qwen2.5-VL-7B-Instruct"  # 或者使用绝对路径 /data/srq/Qwen/Qwen/Qwen2.5-VL-7B-Instruct
train_dataset_json_path = "train.json"
output_dir = "./output/Qwen2.5-VL-VidGuard-CoT" # 修改输出目录名以区分
MAX_LENGTH = 4096

# [创新点] 引入 VidGuard-R1 的 CoT 系统提示词
# 引导模型在给出 Real/Generated 结论前，先进行 <think> 推理
SYSTEM_PROMPT = """Analyze the input video to determine if it is real or AI-generated.
Your reasoning should focus on four key diagnostic categories:
1. Motion Consistency: Check for unnatural movements or floating objects.
2. Lighting Consistency: Check for shadows and light sources that match the environment.
3. Texture Artifacts: Look for overly smooth, plastic-like surfaces or jagged edges.
4. Physics Violations: Ensure gravity and object interactions obey physical laws.

Format your response as:
<think>
[Detailed reasoning covering the 4 categories]
</think>
<answer> [Real or Generated] </answer>"""

# 如果显存紧张，可以降低分辨率限制
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28 

def process_func(example):
    """
    预处理函数：支持视频和图片混合输入
    """
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    
    # 约定：JSON中第一条由用户提供文件路径，第二条是助手回答
    file_path = conversation[0]["value"]
    output_content = conversation[1]["value"]
    
    # 根据后缀判断是视频还是图片
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        content_item = {
            "type": "video",
            "video": file_path,
            # "max_pixels": 360 * 420, # 可选：限制视频帧分辨率以节省显存
            # "fps": 1.0, # 可选：设置采样帧率
        }
        default_prompt = SYSTEM_PROMPT
    else:
        content_item = {
            "type": "image",
            "image": file_path,
        }
        default_prompt = "Describe this image."

    # 构造 Qwen 格式的消息
    messages = [
        {
            "role": "user",
            "content": [
                content_item,
                {"type": "text", "text": default_prompt},
            ],
        }
    ]

    # 1. 应用聊天模板获取文本
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 2. 处理视觉信息 (加载图片/视频)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 3. 输入 Processor
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 转换为 dict 并移除 batch 维度
    inputs = {key: value.tolist() for key, value in inputs.items()} 
    instruction = inputs

    # 4. 处理 Label (回答部分)
    response = tokenizer(f"{output_content}", add_special_tokens=False)

    # 5. 拼接 Input IDs 和 Labels
    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    
    # Label 中 Instruction 部分设为 -100 (不计算 Loss)
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # 构造返回字典，保留 Qwen2.5-VL 特有的视觉特征字段
    final_dict = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }
    
    # 动态添加存在的视觉字段
    for key in ['pixel_values', 'image_grid_thw', 'video_pixel_values', 'video_grid_thw']:
        if key in inputs:
            # 部分字段可能需要 squeeze 去掉 batch 维度
            tensor_val = torch.tensor(inputs[key])
            if key in ['image_grid_thw', 'video_grid_thw']:
                tensor_val = tensor_val.squeeze(0)
            final_dict[key] = tensor_val
            
    return final_dict

# --- 主流程 ---

# 1. 加载 Processor
processor = AutoProcessor.from_pretrained(
    local_model_path, 
    min_pixels=MIN_PIXELS, 
    max_pixels=MAX_PIXELS
)

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path, 
    use_fast=False, 
    trust_remote_code=True
)

# 3. 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
)
model.enable_input_require_grads()

# 4. 加载并处理数据
# 确保你的 train.json 格式正确
if not os.path.exists(train_dataset_json_path):
    raise FileNotFoundError(f"找不到数据集文件: {train_dataset_json_path}")

train_ds = Dataset.from_json(train_dataset_json_path)
train_dataset = train_ds.map(process_func)

# 5. 配置 LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)
train_peft_model = get_peft_model(model, config)
train_peft_model.print_trainable_parameters()

# 6. 训练参数配置
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,     # 视频显存占用大，建议为1
    gradient_accumulation_steps=4,     # 累积4步相当于batch_size=4
    logging_steps=5,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,       # 显存优化：必须开启
    report_to="none",                  # 默认不上传 wandb 等
    bf16=True,                         # Qwen2.5 推荐 bf16
    dataloader_pin_memory=False,       # 避免多模态数据加载卡死
    remove_unused_columns=False,       # 必须设为False，否则自定义字段会被过滤
)

# 7. SwanLab 回调 (如果你需要可视化训练曲线)
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-VL-Video-Detection",
    experiment_name="run-v1",
    config={
        "model_path": local_model_path,
        "lora_rank": 64,
    },
)

# 8. 开始训练
trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

print("开始训练...")
trainer.train()