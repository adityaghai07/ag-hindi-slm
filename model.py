import math
import torch
import torch.nn as nn

import torch.nn.functional as F


from config import Config



class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):

        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        xf = x.float()

        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()

        return (xf * rms).to(x.dtype) * self.weight



def precompute_rope(head_dim: int, seq_len: int, theta: float, device):

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    t = torch.arange(seq_len, device=device).float()

    freqs = torch.outer(t, inv_freq)

    return freqs.cos(), freqs.sin()



def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:

    x1, x2 = x[..., ::2], x[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)

    sin = sin.unsqueeze(0).unsqueeze(0)

    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)

    cos_e = torch.stack([cos, cos], dim=-1).flatten(-2)

    sin_e = torch.stack([sin, sin], dim=-1).flatten(-2)

    return x * cos_e + x_rot * sin_e



class MLAttention(nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim


        self.q_proj  = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)

        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_lora_rank, bias=False)

        self.kv_up   = nn.Linear(cfg.kv_lora_rank, 2 * cfg.n_heads * cfg.head_dim, bias=False)

        self.o_proj  = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)


    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:

        B, T, _ = x.shape

        H, D = self.n_heads, self.head_dim


        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)


        kv_lat = self.kv_down(x)

        kv = self.kv_up(kv_lat).view(B, T, 2, H, D)

        k, v = kv.unbind(2)

        k = k.transpose(1, 2)

        v = v.transpose(1, 2)


        q = apply_rope(q, cos, sin)

        k = apply_rope(k, cos, sin)


        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)

        return self.o_proj(out)



class SwiGLU(nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()

        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)

        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)

        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.down(F.silu(self.gate(x)) * self.up(x))



def block_attn_res(

    blocks: list[torch.Tensor],

    partial: torch.Tensor,

    proj: nn.Linear,

    norm: RMSNorm,

) -> torch.Tensor:

    V = torch.stack(blocks + [partial], dim=0)

    K = norm(V)

    w = proj.weight.squeeze(0)

    logits = torch.einsum("d, nbtd -> nbt", w, K)

    weights = logits.softmax(dim=0)

    return torch.einsum("nbt, nbtd -> btd", weights, V)



class TransformerBlock(nn.Module):

    def __init__(self, cfg: Config, layer_idx: int):

        super().__init__()

        self.layer_idx = layer_idx

        self.block_size = cfg.attn_res_block_size


        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.ffn_norm  = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.attn      = MLAttention(cfg)

        self.ffn       = SwiGLU(cfg)


        self.ar_proj_attn = nn.Linear(cfg.d_model, 1, bias=False)

        self.ar_proj_ffn  = nn.Linear(cfg.d_model, 1, bias=False)

        self.ar_norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.ar_norm_ffn  = RMSNorm(cfg.d_model, cfg.norm_eps)


    def _is_boundary(self) -> bool:

        return (self.layer_idx + 1) % (self.block_size // 2) == 0


    def forward(
        self,

        blocks: list[torch.Tensor],

        partial: torch.Tensor,

        cos: torch.Tensor,

        sin: torch.Tensor,

    ) -> tuple[list[torch.Tensor], torch.Tensor]:

        h = block_attn_res(blocks, partial, self.ar_proj_attn, self.ar_norm_attn)

        partial = partial + self.attn(self.attn_norm(h), cos, sin)


        if self._is_boundary():

            blocks.append(partial)


        h = block_attn_res(blocks, partial, self.ar_proj_ffn, self.ar_norm_ffn)

        partial = partial + self.ffn(self.ffn_norm(h))


        return blocks, partial


class AGHindiSLM(nn.Module):

    def __init__(self, cfg: Config, device=None):

        super().__init__()
        self.cfg = cfg

        _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.embed    = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.layers   = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])

        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight


        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, _device)

        self.register_buffer("rope_cos", cos)

        self.register_buffer("rope_sin", sin)


        self.apply(self._base_init)

        self._scaled_output_init()
        self._attn_res_init()


    def _base_init(self, m):

        if isinstance(m, (nn.Linear, nn.Embedding)):

            nn.init.normal_(m.weight, std=0.02)


    def _scaled_output_init(self):

        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)

        for layer in self.layers:

            nn.init.normal_(layer.attn.o_proj.weight, std=std)

            nn.init.normal_(layer.ffn.down.weight, std=std)

    def _attn_res_init(self):

        for layer in self.layers:

            nn.init.normal_(layer.ar_proj_attn.weight, std=0.001)

            nn.init.normal_(layer.ar_proj_ffn.weight, std=0.001)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):

        _, T = idx.shape

        x = self.embed(idx)

        cos = self.rope_cos[:T]

        sin = self.rope_sin[:T]


        blocks: list[torch.Tensor] = [x]

        partial: torch.Tensor = x


        for layer in self.layers:
            if self.training:

                n_blocks = len(blocks)


                def make_ckpt(lyr, n):

                    def _fn(partial_, cos_, sin_, *block_tensors):

                        blks = list(block_tensors)

                        blks_out, part_out = lyr(blks, partial_, cos_, sin_)

                        new_snaps = blks_out[n:]

                        return (part_out, *new_snaps)

                    return _fn


                fn = make_ckpt(layer, n_blocks)

                result = torch.utils.checkpoint.checkpoint(

                    fn, partial, cos, sin, *blocks, use_reentrant=False
                )

                partial = result[0]

                blocks = blocks + list(result[1:])
            else:

                blocks, partial = layer(blocks, partial, cos, sin)


        logits = self.lm_head(self.norm_out(partial))


        loss = None

        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(-1, self.cfg.vocab_size),

                targets.reshape(-1),

                ignore_index=-1,
            )

        return logits, loss


    def num_params(self) -> int:

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @torch.no_grad()
    def generate(
        self,

        prompt_ids: torch.Tensor,

        max_new: int = 100,

        temperature: float = 0.8,

        top_k: int = 50,

        tokenizer=None,

    ) -> torch.Tensor:

        self.eval()

        idx = prompt_ids.clone()

        eos_id = tokenizer.eos_token_id if tokenizer else None


        for _ in range(max_new):

            idx_ctx = idx[:, -self.cfg.max_seq_len:]

            logits, _ = self(idx_ctx)

            logits = logits[:, -1, :] / temperature


            if top_k:

                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

                logits[logits < v[:, [-1]]] = float("-inf")


            next_id = torch.multinomial(logits.softmax(-1), 1)

            idx = torch.cat([idx, next_id], dim=1)


            if eos_id is not None and next_id.item() == eos_id:

                break

        self.train()

        return idx



def build_optimizer(model: AGHindiSLM, cfg: Config) -> torch.optim.AdamW:

    decay, no_decay = [], []

    no_decay_keys = {"norm", "bias", "embed", "rope", "ar_proj"}

    for name, param in model.named_parameters():

        if not param.requires_grad:

            continue

        if param.dim() >= 2 and not any(k in name for k in no_decay_keys):

            decay.append(param)
        else:

            no_decay.append(param)


    return torch.optim.AdamW(

        [{"params": decay, "weight_decay": cfg.weight_decay},

         {"params": no_decay, "weight_decay": 0.0}],

        lr=cfg.lr_max, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )

