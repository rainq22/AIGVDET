毕设/论文方案：光流二阶残差 + 双流 VLM 的 AIGC 视频检测

1. 研究动机与核心假设
- 现有 VLM 对空间语义强，但对视频中微小、非物理的运动异常不敏感。
- 一阶光流表示速度，二阶光流残差近似加速度变化。AIGC 视频的加速度场不稳定，残差图上更容易出现高频异常。
- 方案：在 Qwen2.5-VL 主干上引入 Motion 旁路，显式注入物理运动特征。

2. 工程落地结构（基于当前仓库）
建议在 Qwen2.5-VL 下新增：

Plaintext
Qwen2.5-VL/
├── data_prep/
│   └── extract_flow_residual.py   # 离线提取二阶残差
├── models/
│   ├── __init__.py
│   ├── motion_adapter.py          # 运动适配器
│   └── dual_stream_qwen.py        # 双流封装
├── train_motion.py                    # 双流训练脚本 (基于 train_old.py)
└── README_V3.md

3. 阶段一：离线数据预处理（光流二阶残差）
目标：将每个视频预先转为残差张量，避免训练时重复计算。

建议输出目录：/data1/srq/Qwen/Qwen2.5-VL/cache/flow_features

关键依赖：torchvision (RAFT), decord, tqdm

示例脚本（extract_flow_residual.py）：

Python
import os
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from decord import VideoReader, cpu
from tqdm import tqdm

NUM_FRAMES = 16
RESIZE_HW = (256, 256)


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = F.resize(frames, size=list(RESIZE_HW))
    return frames.cuda()


def extract_second_order_residual(video_path, model):
    frames = load_video_frames(video_path)
    flows = []
    with torch.no_grad():
        for i in range(frames.shape[0] - 1):
            img1, img2 = frames[i:i + 1], frames[i + 1:i + 2]
            flow_list = model(img1, img2)
            flows.append(flow_list[-1])
    flows = torch.cat(flows, dim=0)              # [T-1, 2, H, W]
    flow_residuals = flows[1:] - flows[:-1]      # [T-2, 2, H, W]
    return flow_residuals.cpu()


if __name__ == "__main__":
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).cuda().eval()

    # 从 train_v2.json / test_v2.json 读取视频路径
    # 建议使用 /data1/srq/Qwen/Qwen2.5-VL 下的数据文件
    json_path = "/data1/srq/Qwen/Qwen2.5-VL/train_v2.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    output_dir = "/data1/srq/Qwen/Qwen2.5-VL/cache/flow_features"
    os.makedirs(output_dir, exist_ok=True)

    for sample in tqdm(data):
        vid_path = sample["conversations"][0]["value"]
        vid_id = os.path.basename(vid_path).split(".")[0]
        flow_res = extract_second_order_residual(vid_path, model)
        torch.save(flow_res, os.path.join(output_dir, f"{vid_id}_flow.pt"))

说明：生成的 flow_residuals 形状为 [T-2, 2, H, W]。

4. 阶段二：模型结构（双流注入）
目标：将残差张量映射为若干 motion tokens，与文本 token 拼接送入 Qwen 主干。

models/motion_adapter.py（示例）
- 3D CNN 编码残差，输出固定 token 数
- 线性映射到 Qwen hidden size

models/dual_stream_qwen.py（示例）
- 加载 Qwen2.5-VL
- 前向时将 motion_embeds 拼接到 inputs_embeds 头部
- labels 对齐：motion token 部分设为 -100

5. 阶段三：训练脚本改造（train_motion.py）
基于 train_old.py 改造，保持 LoRA + label mask 机制。

必须改动：
- process_func：读取 flow_residuals 并写入样本缓存
- DataCollator：对 flow_residuals 批量拼接
- model：替换为 DualStreamQwenDeepfake

关键示意：

Python
# Dataset
result["flow_residuals"] = torch.load(flow_path)

# Collator
flows = [inst["flow_residuals"] for inst in instances]
batch["flow_residuals"] = torch.stack(flows, dim=0)

# Model
model = DualStreamQwenDeepfake(CONFIG.model_path)
for p in model.qwen.visual.parameters():
    p.requires_grad = False
for p in model.motion_adapter.parameters():
    p.requires_grad = True

6. 训练与评测建议
- 训练入口仍沿用 train_old.py 的 loss_mode 设计。
- 评测可先复用 eval_v2.py（确保 LORA_PATH 指向当前训练输出）。
- 核心指标：Binary Acc / F1 / Source Attribution Acc。

7. 实验计划（精简版）
- E1: v2 baseline（无 motion）
- E2: motion + full loss
- E3: motion + answer_only
- E4: motion + rationale_dropout

8. 工作量与论文贡献点（简述）
- 光流二阶残差作为物理先验输入
- 双流注入架构与 motion adapter
- 与 v2 基线的系统消融对比
