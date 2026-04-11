import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

from .motion_adapter import MotionAdapter


class DualStreamQwenDeepfake(nn.Module):
    def __init__(self, qwen_model_path: str):
        super().__init__()
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_path, torch_dtype=torch.bfloat16
        )
        llm_dim = self.qwen.config.hidden_size
        self.motion_adapter = MotionAdapter(llm_hidden_size=llm_dim)

    def forward(
        self,
        input_ids,
        pixel_values_videos,
        video_grid_thw,
        flow_residuals,
        labels=None,
        **kwargs,
    ):
        embedder = getattr(self.qwen, 'get_input_embeddings', None)
        if embedder is not None:
            inputs_embeds = embedder()(input_ids)
        else:
            base = getattr(self.qwen, 'base_model', None)
            if base is None or not hasattr(base, 'get_input_embeddings'):
                raise AttributeError('qwen model does not expose input embeddings')
            inputs_embeds = base.get_input_embeddings()(input_ids)
        motion_embeds = self.motion_adapter(flow_residuals)

        inputs_embeds = torch.cat([motion_embeds, inputs_embeds], dim=1)

        if labels is not None:
            motion_labels = torch.full(
                (labels.shape[0], motion_embeds.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([motion_labels, labels], dim=1)

        motion_len = motion_embeds.shape[1]
        attn = kwargs.get("attention_mask")
        if attn is not None:
            prefix = torch.ones((attn.shape[0], motion_len), device=attn.device, dtype=attn.dtype)
            kwargs["attention_mask"] = torch.cat([prefix, attn], dim=1)

        pos = kwargs.get("position_ids")
        if pos is not None:
            if pos.dim() == 2:
                prefix = torch.arange(motion_len, device=pos.device).unsqueeze(0).expand(pos.size(0), -1)
                kwargs["position_ids"] = torch.cat([prefix, pos + motion_len], dim=1)
            elif pos.dim() == 3:
                prefix = torch.arange(motion_len, device=pos.device).view(1, 1, -1)
                prefix = prefix.expand(pos.size(0), pos.size(1), -1)
                kwargs["position_ids"] = torch.cat([prefix, pos + motion_len], dim=2)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            **kwargs,
        )
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        target = getattr(self.qwen, "gradient_checkpointing_enable", None)
        if target is not None:
            target(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            return
        base = getattr(self.qwen, "base_model", None)
        if base is not None and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        target = getattr(self.qwen, "gradient_checkpointing_disable", None)
        if target is not None:
            target()
            return
        base = getattr(self.qwen, "base_model", None)
        if base is not None and hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
