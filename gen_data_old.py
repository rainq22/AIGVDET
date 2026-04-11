"""
gen_cot_data_v2.py - 增强版数据生成脚本
创新点:
  1. 多任务: 同时输出 Real/Generated + 生成器来源
  2. 模板多样化: 多个 CoT 模板防止模型记忆
  3. 保留 meta 信息用于细粒度评测
"""
import json
import os
import random
from datetime import datetime

# --- 配置参数 ---
DATASET_ROOT = "/data/srq/Qwen/GenBuster-200K-mini"
BASE_OUTPUT_DIR = "/data1/srq/Qwen/Qwen2.5-VL/datasets"
DATE_DIR = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, DATE_DIR)
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

SPLITS = {
    "train": "train_v2.json",
    "test": "test_v2.json",
    "benchmark": "benchmark_v2.json"
}

# ══════════════════════════════════════════════════════════════════
# 创新点 1: 多样化 CoT 模板池（防止模板记忆）
# ══════════════════════════════════════════════════════════════════

REAL_COT_TEMPLATES = [
    # 模板 1: 强调运动一致性
    """<think>
1. Motion Analysis: Object movements follow natural physics - acceleration, deceleration, and inertia are consistent throughout.
2. Temporal Coherence: Frame-to-frame transitions are smooth with no sudden jumps or artifacts.
3. Texture Quality: Fine details like fabric weave, skin texture, and environmental noise appear naturally irregular.
4. Lighting Fidelity: Shadows and highlights respond realistically to scene geometry and light sources.
Conclusion: All visual cues indicate authentic captured footage.
</think>
<answer>Real</answer>
<source>real</source>""",

    # 模板 2: 强调纹理细节
    """<think>
1. Surface Details: Microscopic irregularities in textures (pores, grain, dust) are present and natural.
2. Motion Dynamics: Moving objects exhibit proper motion blur and temporal consistency.
3. Environmental Interaction: Reflections, shadows, and occlusions behave physically correctly.
4. Compression Artifacts: Only standard video compression patterns, no AI-specific distortions.
Conclusion: The video shows characteristics of real-world capture.
</think>
<answer>Real</answer>
<source>real</source>""",

    # 模板 3: 强调物理一致性
    """<think>
1. Physics Compliance: Gravity, momentum, and object interactions follow natural laws.
2. Geometric Stability: Object shapes remain consistent; no warping or morphing detected.
3. Noise Patterns: Sensor noise is uniform and matches typical camera characteristics.
4. Temporal Flow: Motion is continuous without frame-level inconsistencies.
Conclusion: Evidence strongly suggests genuine video recording.
</think>
<answer>Real</answer>
<source>real</source>""",

    # 模板 4: 简洁版
    """<think>
1. Natural motion with proper physics and inertia.
2. Authentic textures with realistic micro-details.
3. Consistent lighting and shadow behavior.
4. No temporal or spatial generation artifacts.
Conclusion: This is authentic footage.
</think>
<answer>Real</answer>
<source>real</source>""",

    # 模板 5: 强调对比分析
    """<think>
1. Compared to AI-generated videos, motion trajectories follow realistic physics without sudden changes.
2. High-frequency details are preserved without the smoothing typical of generative models.
3. Background elements move consistently with camera motion, not frozen or independent.
4. No characteristic flickering or temporal instability in edges and fine structures.
Conclusion: The video exhibits authentic capture characteristics.
</think>
<answer>Real</answer>
<source>real</source>""",
]

# 通用 Fake 模板（适用于所有生成器）
FAKE_COT_TEMPLATES_GENERIC = [
    # 模板 1
    """<think>
1. Motion Artifacts: Some movements appear unnatural - objects may float, slide, or ignore physics.
2. Temporal Issues: Frame-to-frame consistency is imperfect; subtle flickering in edges and textures.
3. Texture Smoothness: Surfaces lack natural micro-details; an overly smooth, painted appearance.
4. Geometric Instability: Object shapes may subtly warp or morph between frames.
Conclusion: Multiple indicators suggest AI-generated content.
</think>
<answer>Generated</answer>
<source>{source}</source>""",

    # 模板 2
    """<think>
1. Unnatural Dynamics: Motion lacks proper acceleration curves and physical momentum.
2. Detail Degradation: Fine textures are blurred or replaced with synthetic patterns.
3. Lighting Inconsistency: Shadows don't perfectly align with light source positions.
4. Boundary Artifacts: Edges between objects show subtle blending or halo effects.
Conclusion: The video exhibits AI generation characteristics.
</think>
<answer>Generated</answer>
<source>{source}</source>""",

    # 模板 3
    """<think>
1. Physics Violations: Objects don't interact naturally; some movements defy gravity or inertia.
2. Temporal Flickering: Subtle changes in texture and color between consecutive frames.
3. Over-smoothing: Natural noise and grain are absent; surfaces appear artificially clean.
4. Semantic Errors: Minor logical inconsistencies in object behavior or scene composition.
Conclusion: Evidence indicates synthetic video generation.
</think>
<answer>Generated</answer>
<source>{source}</source>""",

    # 模板 4: 简洁版
    """<think>
1. Unnatural motion patterns inconsistent with real physics.
2. Lack of authentic high-frequency texture details.
3. Temporal instability in edges and fine structures.
4. Subtle geometric distortions in moving objects.
Conclusion: This is AI-generated content.
</think>
<answer>Generated</answer>
<source>{source}</source>""",

    # 模板 5
    """<think>
1. Motion analysis reveals non-physical trajectories and velocity changes.
2. Texture inspection shows synthetic smoothness without natural irregularities.
3. Frame comparison detects inconsistent details across temporal sequence.
4. Edge analysis finds characteristic generation artifacts.
Conclusion: The video is synthetically generated.
</think>
<answer>Generated</answer>
<source>{source}</source>""",
]

# ══════════════════════════════════════════════════════════════════
# 创新点 2: 生成器特定模板（更精细的归因描述）
# ══════════════════════════════════════════════════════════════════

GENERATOR_SPECIFIC_TEMPLATES = {
    "cogvideox": [
        """<think>
1. Motion Pattern: Characteristic smooth but sometimes physically implausible movements typical of diffusion-based video generation.
2. Temporal Coherence: Generally consistent but with occasional subtle flickering in detailed regions.
3. Texture Style: Slightly over-smoothed with the artistic quality common in CogVideoX outputs.
4. Generation Signature: Frame-level patterns consistent with transformer-based video diffusion.
Conclusion: Visual patterns match CogVideoX generation characteristics.
</think>
<answer>Generated</answer>
<source>cogvideox</source>""",
    ],
    "easyanimate": [
        """<think>
1. Motion Dynamics: Smooth interpolation but occasional unnatural acceleration patterns.
2. Visual Quality: High aesthetic quality but lacking authentic camera noise and grain.
3. Temporal Flow: Generally stable with minor inconsistencies in complex motion regions.
4. Style Signature: Characteristic rendering style of EasyAnimate's generation pipeline.
Conclusion: Artifacts consistent with EasyAnimate video synthesis.
</think>
<answer>Generated</answer>
<source>easyanimate</source>""",
    ],
    "hunyuanvideo": [
        """<think>
1. Motion Characteristics: Fluid movements but with subtle physics inconsistencies.
2. Detail Rendering: Good overall quality but synthetic texture patterns in close inspection.
3. Temporal Stability: Minor flickering artifacts in high-frequency detail areas.
4. Generation Pattern: Visual signatures matching HunyuanVideo's diffusion architecture.
Conclusion: The video shows HunyuanVideo generation patterns.
</think>
<answer>Generated</answer>
<source>hunyuanvideo</source>""",
    ],
    "ltxvideo": [
        """<think>
1. Motion Quality: Generally smooth but occasional frame-level inconsistencies.
2. Texture Appearance: Characteristic smoothness with reduced natural noise.
3. Temporal Coherence: Good overall but detectable artifacts in detailed motion.
4. Style Markers: Visual patterns consistent with LTX-Video generation.
Conclusion: Artifacts indicate LTX-Video synthesis.
</think>
<answer>Generated</answer>
<source>ltxvideo</source>""",
    ],
    # 通用 fallback
    "luma": [
        """<think>
1. Motion Analysis: Smooth but occasionally physics-defying movements.
2. Visual Quality: High aesthetic quality but lacking authentic imperfections.
3. Temporal Flow: Minor inconsistencies between frames in detailed regions.
4. Generation Signature: Patterns consistent with Luma video generation.
Conclusion: The video exhibits Luma generation characteristics.
</think>
<answer>Generated</answer>
<source>luma</source>""",
    ],
    "sora": [
        """<think>
1. Motion Dynamics: Highly realistic but with subtle temporal inconsistencies.
2. Physics Simulation: Generally good but occasional violations in complex interactions.
3. Detail Quality: Impressive but lacking authentic sensor noise patterns.
4. Generation Pattern: Consistent with Sora's advanced video synthesis.
Conclusion: Visual analysis suggests Sora-generated content.
</think>
<answer>Generated</answer>
<source>sora</source>""",
    ],
}

def get_category_from_path(path):
    """从路径中提取生成器名称"""
    parts = path.split(os.sep)
    if 'real' in parts:
        return 'real'
    try:
        fake_index = parts.index('fake')
        if fake_index + 1 < len(parts):
            return parts[fake_index + 1].lower()
    except ValueError:
        pass
    return 'unknown'

def get_response_template(label, category):
    """根据标签和类别选择模板"""
    if label == "Real":
        return random.choice(REAL_COT_TEMPLATES)
    else:
        # 50% 概率使用生成器特定模板，50% 使用通用模板
        if category in GENERATOR_SPECIFIC_TEMPLATES and random.random() < 0.5:
            return random.choice(GENERATOR_SPECIFIC_TEMPLATES[category])
        else:
            template = random.choice(FAKE_COT_TEMPLATES_GENERIC)
            return template.format(source=category)

def process_split(split_name, output_file):
    root_dir = os.path.join(DATASET_ROOT, split_name)
    if not os.path.exists(root_dir):
        print(f"跳过 {split_name}: 目录不存在")
        return

    print(f"正在扫描 {split_name} 数据集...")
    data_list = []
    category_stats = {}
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(VIDEO_EXTS):
                full_path = os.path.join(root, file)
                
                if '/real' in full_path:
                    label = "Real"
                else:
                    label = "Generated"
                
                category = get_category_from_path(full_path)
                category_stats[category] = category_stats.get(category, 0) + 1
                
                response = get_response_template(label, category)
                
                unique_id = f"{split_name}_{label.lower()}_{category}_{len(data_list)}"
                
                entry = {
                    "id": unique_id,
                    "conversations": [
                        {"role": "user", "value": full_path},
                        {"role": "assistant", "value": response}
                    ],
                    "meta": {
                        "label": label,
                        "category": category,
                        "split": split_name
                    }
                }
                data_list.append(entry)

    # 打乱顺序
    random.shuffle(data_list)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"--> 已生成 {output_file}: 共 {len(data_list)} 条数据")
    print(f"    类别统计: {category_stats}")

if __name__ == "__main__":
    random.seed(42)  # 保证可复现
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split, filename in SPLITS.items():
        output_path = os.path.join(OUTPUT_DIR, filename)
        process_split(split, output_path)
    print("\n✅ 数据生成完成! 新增特性:")
    print("   - 多任务输出: <answer> + <source>")
    print("   - 模板多样化: 5+ 个 CoT 模板")
    print("   - 生成器特定描述")