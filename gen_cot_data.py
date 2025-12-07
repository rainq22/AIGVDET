import json
import os
import random

# --- 配置参数 ---
# 数据集根目录
DATASET_ROOT = "/data/srq/Qwen/GenBuster-200K-mini"
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

# 定义不同 Split 的输出文件名
SPLITS = {
    "train": "train.json",
    "test": "test.json",
    "benchmark": "benchmark.json"
}

# CoT 模板 (保持不变，或根据需要扩充)
REAL_COT_TEMPLATES = [
    """<think>\n1. Motion Consistency: The movement of objects obeys physics perfectly; inertia and gravity are consistent.\n2. Lighting Consistency: Shadows cast by the objects accurately reflect the environmental light sources.\n3. Texture Artifacts: High-frequency details like skin pores or surface noise are natural and irregular.\n4. Physics Violations: No warping or morphing of rigid objects observed.\nConclusion: The video exhibits all characteristics of natural footage.\n</think>\n<answer> Real </answer>"""
]

FAKE_COT_TEMPLATES = [
    """<think>\n1. Motion Consistency: Some background elements remain static or move disjointedly from the foreground.\n2. Lighting Consistency: The lighting on the subject seems flat or inconsistent with the background scene.\n3. Texture Artifacts: Surfaces appear overly smooth, lacking natural grain; some edges are blurry or jagged.\n4. Physics Violations: Minor temporal inconsistencies observed, suggesting frame-by-frame generation.\nConclusion: These artifacts are indicative of AI synthesis.\n</think>\n<answer> Generated </answer>"""
]

def get_category_from_path(path):
    """从路径中提取生成器名称，例如 .../fake/sora/1.mp4 -> sora"""
    parts = path.split(os.sep)
    if 'real' in parts:
        return 'real'
    try:
        # 假设结构是 .../fake/{generator_name}/video.mp4
        fake_index = parts.index('fake')
        if fake_index + 1 < len(parts):
            return parts[fake_index + 1] # 返回 sora, kling 等
    except ValueError:
        pass
    return 'unknown_fake'

def process_split(split_name, output_file):
    root_dir = os.path.join(DATASET_ROOT, split_name)
    if not os.path.exists(root_dir):
        print(f"跳过 {split_name}: 目录不存在")
        return

    print(f"正在扫描 {split_name} 数据集...")
    data_list = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(VIDEO_EXTS):
                full_path = os.path.join(root, file)
                
                # 判定标签
                if '/real' in full_path:
                    label = "Real"
                    template = random.choice(REAL_COT_TEMPLATES)
                else:
                    label = "Generated"
                    template = random.choice(FAKE_COT_TEMPLATES)
                
                # 提取细分类型 (用于后续分析)
                category = get_category_from_path(full_path)
                
                # 构造 ID，包含分类信息方便 debug
                # 例如: train_fake_sora_001
                unique_id = f"{split_name}_{label.lower()}_{category}_{len(data_list)}"
                
                entry = {
                    "id": unique_id,
                    "conversations": [
                        {
                            "role": "user",
                            "value": full_path
                        },
                        {
                            "role": "assistant",
                            "value": template
                        }
                    ],
                    # [创新点] 保留元数据，虽然训练时不直接用，但方便人工检查
                    "meta": {
                        "category": category,
                        "split": split_name
                    }
                }
                data_list.append(entry)

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"--> 已生成 {output_file}: 共 {len(data_list)} 条数据")

if __name__ == "__main__":
    for split, filename in SPLITS.items():
        process_split(split, filename)