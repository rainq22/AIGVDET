"""
+gen_cot_data_v4.py - 改进版数据生成脚本
+改进:
+  1) 模板数量增加 + 轻量扰动
+  2) generator 名称规范化，减少 source 噪声
+  3) 维持统一格式，兼容 v4 的精确 mask
+"""
+import json
+import os
+import random
+
+DATASET_ROOT = "/data/srq/Qwen/GenBuster-200K-mini"
+VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
+
+SPLITS = {
+    "train": "train_v4.json",
+    "test": "test_v4.json",
+    "benchmark": "benchmark_v4.json",
+}
+
+REAL_TEMPLATES = [
+    """<think>
+1. Motion Analysis: Movements obey inertia and acceleration patterns consistent with real physics.
+2. Temporal Coherence: Frame transitions are smooth without unnatural flicker.
+3. Texture Fidelity: Fine details and sensor noise appear naturally irregular.
+4. Lighting Consistency: Shadows and highlights align with scene geometry.
+Conclusion: The video appears to be authentic.
+</think>
+<answer>Real</answer>
+<source>real</source>""",
+    """<think>
+1. Texture Details: Microscopic irregularities are present without oversmoothing.
+2. Motion Dynamics: Motion blur looks natural and temporally consistent.
+3. Environmental Interaction: Reflections and occlusions behave physically.
+Conclusion: This is genuine captured footage.
+</think>
+<answer>Real</answer>
+<source>real</source>""",
+    """<think>
+1. Physics Compliance: Object trajectories follow real-world constraints.
+2. Geometric Stability: Shapes remain stable across frames.
+3. Noise Patterns: Sensor noise is consistent with camera characteristics.
+Conclusion: Evidence supports a real video.
+</think>
+<answer>Real</answer>
+<source>real</source>""",
+    """<think>
+1. Natural motion and realistic temporal flow.
+2. Authentic textures with subtle imperfections.
+Conclusion: Authentic video.
+</think>
+<answer>Real</answer>
+<source>real</source>""",
+]
+
+FAKE_GENERIC_TEMPLATES = [
+    """<think>
+1. Motion Artifacts: Movements show non-physical trajectories.
+2. Temporal Issues: Subtle flicker in edges and textures across frames.
+3. Texture Smoothness: Surfaces look overly clean with reduced fine detail.
+Conclusion: This video is AI-generated.
+</think>
+<answer>Generated</answer>
+<source>{source}</source>""",
+    """<think>
+1. Unnatural Dynamics: Acceleration patterns appear inconsistent.
+2. Detail Degradation: High-frequency textures look blurred or synthetic.
+3. Boundary Artifacts: Edges show faint blending or halo effects.
+Conclusion: The content appears generated.
+</think>
+<answer>Generated</answer>
+<source>{source}</source>""",
+    """<think>
+1. Physics Violations: Object interactions do not fully follow real laws.
+2. Temporal Flicker: Micro-changes in color and texture between frames.
+3. Over-smoothing: Natural noise is missing.
+Conclusion: Indicators suggest synthetic generation.
+</think>
+<answer>Generated</answer>
+<source>{source}</source>""",
+    """<think>
+1. Subtle geometry warping during motion.
+2. Texture regularity inconsistent with real capture.
+Conclusion: AI-generated video.
+</think>
+<answer>Generated</answer>
+<source>{source}</source>""",
+]
+
+GENERATOR_SPECIFIC_TEMPLATES = {
+    "cogvideox": [
+        """<think>
+1. Motion Pattern: Smooth but with diffusion-like temporal artifacts.
+2. Texture Style: Slightly over-smoothed with a stylized look.
+3. Temporal Coherence: Minor flicker in detailed regions.
+Conclusion: Matches CogVideoX generation patterns.
+</think>
+<answer>Generated</answer>
+<source>cogvideox</source>""",
+    ],
+    "easyanimate": [
+        """<think>
+1. Motion Dynamics: Smooth interpolation with occasional unnatural acceleration.
+2. Visual Quality: High aesthetic quality but weak sensor noise.
+Conclusion: Consistent with EasyAnimate outputs.
+</think>
+<answer>Generated</answer>
+<source>easyanimate</source>""",
+    ],
+    "hunyuanvideo": [
+        """<think>
+1. Temporal Stability: Mostly stable with fine-detail flicker.
+2. Texture Rendering: Synthetic micro-patterns on close inspection.
+Conclusion: Aligns with HunyuanVideo artifacts.
+</think>
+<answer>Generated</answer>
+<source>hunyuanvideo</source>""",
+    ],
+    "ltxvideo": [
+        """<think>
+1. Motion Quality: Smooth but with frame-level inconsistencies.
+2. Texture Appearance: Reduced natural grain.
+Conclusion: Typical of LTX-Video generation.
+</think>
+<answer>Generated</answer>
+<source>ltxvideo</source>""",
+    ],
+    "sora": [
+        """<think>
+1. Motion Dynamics: Highly realistic but slight temporal drift in details.
+2. Physics Simulation: Good overall with minor violations in complex scenes.
+Conclusion: Resembles Sora-generated video.
+</think>
+<answer>Generated</answer>
+<source>sora</source>""",
+    ],
+    "luma": [
+        """<think>
+1. Motion Analysis: Smooth trajectories with occasional physics slips.
+2. Visual Quality: Clean textures lacking real sensor noise.
+Conclusion: Consistent with Luma generation.
+</think>
+<answer>Generated</answer>
+<source>luma</source>""",
+    ],
+}
+
+SOURCE_NORMALIZATION = {
+    "cogvideox": "cogvideox",
+    "easyanimate": "easyanimate",
+    "hunyuanvideo": "hunyuanvideo",
+    "ltxvideo": "ltxvideo",
+    "sora": "sora",
+    "luma": "luma",
+    "real": "real",
+}
+
+
+def normalize_source(name: str) -> str:
+    name = (name or "").lower()
+    return SOURCE_NORMALIZATION.get(name, "unknown")
+
+
+def get_category_from_path(path: str) -> str:
+    parts = path.split(os.sep)
+    if "real" in parts:
+        return "real"
+    try:
+        fake_index = parts.index("fake")
+        if fake_index + 1 < len(parts):
+            return parts[fake_index + 1].lower()
+    except ValueError:
+        pass
+    return "unknown"
+
+
+def maybe_shuffle_think(template: str) -> str:
+    if random.random() > 0.3:
+        return template
+    lines = template.splitlines()
+    head = []
+    body = []
+    tail = []
+    in_think = False
+    for line in lines:
+        if line.strip() == "<think>":
+            in_think = True
+            head.append(line)
+        elif line.strip() == "</think>":
+            in_think = False
+            tail.append(line)
+        elif in_think:
+            body.append(line)
+        else:
+            tail.append(line)
+    if len(body) > 2:
+        random.shuffle(body)
+    return "\n".join(head + body + tail)
+
+
+def get_response_template(label: str, category: str) -> str:
+    if label == "Real":
+        template = random.choice(REAL_TEMPLATES)
+        return maybe_shuffle_think(template)
+
+    source = normalize_source(category)
+    use_specific = source in GENERATOR_SPECIFIC_TEMPLATES and random.random() < 0.5
+    if use_specific:
+        template = random.choice(GENERATOR_SPECIFIC_TEMPLATES[source])
+        return maybe_shuffle_think(template)
+
+    template = random.choice(FAKE_GENERIC_TEMPLATES)
+    return maybe_shuffle_think(template.format(source=source))
+
+
+def process_split(split_name: str, output_file: str):
+    root_dir = os.path.join(DATASET_ROOT, split_name)
+    if not os.path.exists(root_dir):
+        print(f"跳过 {split_name}: 目录不存在")
+        return
+
+    print(f"正在扫描 {split_name} 数据集...")
+    data_list = []
+    category_stats = {}
+
+    for root, _, files in os.walk(root_dir):
+        for file in files:
+            if file.lower().endswith(VIDEO_EXTS):
+                full_path = os.path.join(root, file)
+                label = "Real" if "/real" in full_path else "Generated"
+                category = normalize_source(get_category_from_path(full_path))
+                category_stats[category] = category_stats.get(category, 0) + 1
+
+                response = get_response_template(label, category)
+                unique_id = f"{split_name}_{label.lower()}_{category}_{len(data_list)}"
+
+                entry = {
+                    "id": unique_id,
+                    "conversations": [
+                        {"role": "user", "value": full_path},
+                        {"role": "assistant", "value": response},
+                    ],
+                    "meta": {"label": label, "category": category, "split": split_name},
+                }
+                data_list.append(entry)
+
+    random.shuffle(data_list)
+
+    with open(output_file, "w", encoding="utf-8") as f:
+        json.dump(data_list, f, indent=2, ensure_ascii=False)
+
+    print(f"--> 已生成 {output_file}: 共 {len(data_list)} 条数据")
+    print(f"    类别统计: {category_stats}")
+
+
+if __name__ == "__main__":
+    random.seed(42)
+    for split, filename in SPLITS.items():
+        process_split(split, filename)
+    print("\n数据生成完成 (v4)")
