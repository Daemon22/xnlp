"""
XNLP Core Large Language Model (LLM)
=====================================
A LLaMA-style transformer architecture optimized for isiXhosa language processing.

Features:
- Rotary Position Embeddings (RoPE)
- SwiGLU Activation Function
- RMSNorm (Root Mean Square Normalization)
- Pre-Normalization Architecture
- KV-Cache for Efficient Inference
- Grouped Query Attention (GQA) Support

Author: XNLP Team
Version: 2.0.0 (Core LLM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class XNLPConfig:
    """Configuration for XNLP Core LLM."""
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1280
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_type: str = "xnlp_llm"
    version: str = "2.0.0"
    language: str = "isiXhosa"
    
    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads
    
    @property
    def total_params(self) -> int:
        params = self.vocab_size * self.hidden_size
        q_params = self.hidden_size * self.hidden_size
        kv_params = 2 * self.hidden_size * (self.num_key_value_heads * self.head_dim)
        attention_output_params = self.hidden_size * self.hidden_size
        ffn_params = 3 * self.hidden_size * self.intermediate_size
        per_layer = q_params + kv_params + attention_output_params + ffn_params + 2 * self.hidden_size
        params += per_layer * self.num_hidden_layers
        params += self.hidden_size + self.vocab_size * self.hidden_size
        return params


class RotaryEmbedding(nn.Module):
    def __init__(self, config: XNLPConfig):
        super().__init__()
        self.head_dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, seq_len: int, device, dtype=torch.float32):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, config: XNLPConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: XNLPConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(config)
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                position_ids=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        seq_len = q_len
        if past_key_value is not None:
            seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(seq_len, device=hidden_states.device, dtype=hidden_states.dtype)

        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos[-q_len:].unsqueeze(0).unsqueeze(0)
            sin = sin[-q_len:].unsqueeze(0).unsqueeze(0)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class XNLPDecoderLayer(nn.Module):
    def __init__(self, config: XNLPConfig):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLUFFN(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class XNLPCoreLLM(nn.Module):
    def __init__(self, config: XNLPConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([XNLPDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()
        self.post_init()
        self.gradient_checkpointing = False
    
    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight
    
    def post_init(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def _prepare_decoder_attention_mask(self, input_shape, device, dtype, past_key_values_length=0):
        """Build a causal mask for full-sequence and cached decoding."""
        _, query_len = input_shape
        total_key_len = past_key_values_length + query_len
        mask = torch.triu(
            torch.full(
                (query_len, total_key_len),
                torch.finfo(dtype).min,
                device=device,
                dtype=dtype,
            ),
            diagonal=past_key_values_length + 1,
        )
        return mask.unsqueeze(0)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=False, return_dict=True):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_len, _ = inputs_embeds.shape
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                device=input_ids.device,
            ).unsqueeze(0)
        
        if attention_mask is not None:
            combined_attention_mask = self._prepare_decoder_attention_mask(
                (batch_size, seq_len), device=input_ids.device,
                dtype=inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )
            if attention_mask.dim() == 2:
                expanded_mask = attention_mask[:, None, None, :]
                expanded_mask = (1.0 - expanded_mask.to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
                combined_attention_mask = expanded_mask + combined_attention_mask
        else:
            combined_attention_mask = self._prepare_decoder_attention_mask(
                (batch_size, seq_len), device=input_ids.device,
                dtype=inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )
        
        hidden_states = inputs_embeds
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                next_decoder_cache += (present_key_value,)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (next_decoder_cache,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return {"loss": loss, "logits": logits, "past_key_values": next_decoder_cache}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50,
                top_p=0.9, repetition_penalty=1.1, do_sample=True,
                pad_token_id=None, eos_token_id=None, streamer=None):
        self.eval()

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input_ids must contain at least one token")
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Prompt length ({seq_len}) exceeds the model context window "
                f"({self.config.max_position_embeddings})."
            )
        if seq_len + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError(
                f"Requested generation needs {seq_len + max_new_tokens} positions, "
                f"but the model context window supports only "
                f"{self.config.max_position_embeddings}. "
                "Reduce max_new_tokens or shorten the prompt."
            )

        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for step in range(max_new_tokens):
            if past_key_values is not None:
                current_input = generated_ids[:, -1:]
            else:
                current_input = generated_ids
            
            outputs = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            next_token_logits = logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                if repetition_penalty <= 0:
                    raise ValueError("repetition_penalty must be > 0")
                for batch_idx in range(batch_size):
                    for prev_token in generated_ids[batch_idx].unique():
                        if next_token_logits[batch_idx, prev_token] > 0:
                            next_token_logits[batch_idx, prev_token] /= repetition_penalty
                        else:
                            next_token_logits[batch_idx, prev_token] *= repetition_penalty
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
            
            if streamer is not None:
                streamer.put(next_token.cpu())
        
        if streamer is not None:
            streamer.end()
        
        return generated_ids


# Preset configurations
XNLP_PRESETS = {
    "tiny": dict(hidden_size=256, intermediate_size=640, num_hidden_layers=6,
                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512),
    "small": dict(hidden_size=512, intermediate_size=1280, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=1024),
    "medium": dict(hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
                   num_attention_heads=12, num_key_value_heads=6, max_position_embeddings=2048),
}


def create_preset_model(preset_name="small", **overrides):
    if preset_name not in XNLP_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    config_dict = XNLP_PRESETS[preset_name].copy()
    config_dict.update(overrides)
    config = XNLPConfig(**config_dict)
    model = XNLPCoreLLM(config)
    return config, model
