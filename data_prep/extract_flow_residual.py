import os
import json
import argparse
from typing import List

import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from decord import VideoReader, cpu
from tqdm import tqdm


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract second-order optical flow residuals")
    parser.add_argument(
        "--json",
        default=None,
        help="Path to a single json file (overrides --jsons)",
    )
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=None,
        help=(
            "One or more json files. Default: train_v2.json and test_v2.json in the latest dataset dir."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Dataset directory containing train_v2.json/test_v2.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/data1/srq/Qwen/Qwen2.5-VL/cache/flow_features",
        help="Directory to save .pt residual tensors",
    )
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample")
    parser.add_argument("--resize", type=int, default=256, help="Resize shorter side to this size")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output exists")
    return parser.parse_args()


def load_video_frames(video_path: str, num_frames: int, resize_hw: int, device: str) -> torch.Tensor:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = F.resize(frames, size=[resize_hw, resize_hw])
    return frames.to(device)


def extract_second_order_residual(video_path: str, model, num_frames: int, resize_hw: int, device: str) -> torch.Tensor:
    frames = load_video_frames(video_path, num_frames, resize_hw, device)
    flows = []
    with torch.no_grad():
        for i in range(frames.shape[0] - 1):
            img1, img2 = frames[i : i + 1], frames[i + 1 : i + 2]
            flow_list = model(img1, img2)
            flows.append(flow_list[-1])
    flows = torch.cat(flows, dim=0)
    flow_residuals = flows[1:] - flows[:-1]
    return flow_residuals.cpu()


def load_video_list(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def main() -> None:
    args = parse_args()

    if args.json:
        json_paths = [args.json]
    elif args.jsons:
        json_paths = args.jsons
    else:
        dataset_dir = args.dataset_dir or DATASET_DIR
        json_paths = [
            os.path.join(dataset_dir, "train_v2.json"),
            os.path.join(dataset_dir, "test_v2.json"),
        ]

    for json_path in json_paths:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(args.device).eval()

    for json_path in json_paths:
        data = load_video_list(json_path)
        if args.max_samples:
            data = data[: args.max_samples]

        manifest = []
        for sample in tqdm(data, desc=f"Extract flow residuals: {os.path.basename(json_path)}"):
            video_path = sample["conversations"][0]["value"]
            sample_id = sample.get("id") or os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(args.output_dir, f"{sample_id}_flow.pt")

            if args.skip_existing and os.path.exists(out_path):
                manifest.append({"id": sample_id, "video": video_path, "flow_path": out_path})
                continue

            try:
                flow_res = extract_second_order_residual(
                    video_path, model, args.num_frames, args.resize, args.device
                )
                torch.save(flow_res, out_path)
                manifest.append({"id": sample_id, "video": video_path, "flow_path": out_path})
            except Exception as exc:
                manifest.append({"id": sample_id, "video": video_path, "flow_path": None, "error": str(exc)})

        stem = os.path.splitext(os.path.basename(json_path))[0]
        manifest_path = os.path.join(args.output_dir, f"flow_manifest_{stem}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
