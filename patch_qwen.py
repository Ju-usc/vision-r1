import torch
import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import rotate_half


# def custom_apply_multimodal_rotary_pos_emb(
#     q, k, cos, sin, mrope_section, unsqueeze_dim=1
# ):
#     # Removed: mrope_section = mrope_section * 2 otherwise will cause error
#     cos = torch.cat(
#         [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
#     ).unsqueeze(unsqueeze_dim)
#     sin = torch.cat(
#         [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
#     ).unsqueeze(unsqueeze_dim)

#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed
def custom_apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section, unsqueeze_dim=1
):
    # Double the mrope_section values to match the head dimension (128)
    mrope_section = [s * 2 for s in mrope_section]  # This makes it [32, 48, 48] which sums to 128
    
    cos_chunks = cos.split(mrope_section, dim=-1)
    sin_chunks = sin.split(mrope_section, dim=-1)
    
    cos = torch.cat(
        [m[i % len(cos_chunks)] for i, m in enumerate(cos_chunks)], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % len(sin_chunks)] for i, m in enumerate(sin_chunks)], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Monkey patching the function
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
    custom_apply_multimodal_rotary_pos_emb
)
