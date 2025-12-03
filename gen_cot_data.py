import json
import os

# 假设你有一个简单的列表，包含视频路径和真实标签
# 你也可以从 csv 读取
raw_data = [
    {"path": "/data/srq/dataset/video1.mp4", "label": "Real"},
    {"path": "/data/srq/dataset/video2.mp4", "label": "Generated"},
]

# 简单的思维链模板（你可以手动修改或用大模型API替换生成）
def create_cot_response(label):
    if label == "Generated":
        return """<think>
1. Motion Consistency: The object movement appears slightly floating and lacks weight.
2. Lighting Consistency: Some shadows are missing or inconsistent with the light source.
3. Texture Artifacts: Surfaces appear overly smooth or have digital artifacts.
4. Physics Violations: Minor gravity inconsistencies observed.
These signs point to synthetic generation.
</think>
<answer> Generated </answer>"""
    else:
        return """<think>
1. Motion Consistency: Movement is fluid, natural, and obeys inertia.
2. Lighting Consistency: Lighting is complex but physically correct.
3. Texture Artifacts: High-frequency details and natural noise are present.
4. Physics Violations: No violations of physical laws observed.
The video exhibits all characteristics of real footage.
</think>
<answer> Real </answer>"""

train_data = []
for idx, item in enumerate(raw_data):
    entry = {
        "id": f"vid_{idx}",
        "conversations": [
            {
                "role": "user",
                "value": item["path"]
            },
            {
                "role": "assistant",
                "value": create_cot_response(item["label"])
            }
        ]
    }
    train_data.append(entry)

# 保存为 train.json
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

print(f"生成的 train.json 包含 {len(train_data)} 条 CoT 数据。")