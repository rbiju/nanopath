# ViT: clean ViT + 4 register tokens that loads Meta's `dinov2_vit{s,b,l,g}14_reg`
# pretrained weights via state_dict (no xformers, no dinov2 codebase imports).
# Attention runs on `F.scaled_dot_product_attention` so we get FlashAttention-2
# on H100 bf16 with no third-party kernel dependency. Module names below match
# Meta's checkpoint key layout exactly, so `load_pretrained(model)` does
# a strict load.
#
# SpecializedViT is the same ViT with CLS/patch weight specialization (Marouani et al.).
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS self-distillation loss. It is intentionally trivial
# (~15 lines) so we have zero runtime dependency on the dinov2 codebase.

import math
from copy import deepcopy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# (dim, depth, heads, pretrain_grid, patch, ffn, pos_has_cls, weight URL[, registers]) per variant.
VIT_VARIANTS = {
    "dinov2_vits14_reg": (384, 12, 6, 37, 14, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"),
    "dinov2_vitb14_reg": (768, 12, 12, 37, 14, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"),
    "dinov2_vitl14_reg": (1024, 24, 16, 37, 14, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"),
    "dinov2_vitg14_reg": (1536, 40, 24, 37, 14, "swiglu", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"),
}


def probe_transforms():
    # Default for Nanopath-trained checkpoints; baseline scripts override this in their request config.
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    return transform, transform


# Stochastic depth: keep_prob bernoulli on the residual branch, scaled to preserve mean.
class DropPath(nn.Module):
    def __init__(self, p): super().__init__(); self.p = float(p)
    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


# Per-channel learnable scale on residual branches; matches Meta's `ls1.gamma`/`ls2.gamma`.
class LayerScale(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.gamma


# FINO gradient gate: identity forward, scales the gradient by `scale` on backward. sign>0 encourages the
# encoder to predict a metadata factor (M+); sign<0 reverses the gradient to suppress it (M-, DANN-style).
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale): ctx.scale = scale; return x
    @staticmethod
    def backward(ctx, g): return g * ctx.scale, None


# Attention with single qkv Linear + F.scaled_dot_product_attention (Flash-2 backend on H100 bf16).
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden, bias=True)
        self.w3 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


# Standard pre-LN block: attn + ls1 + drop_path, then mlp + ls2 + drop_path.
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop_path_p, ffn="mlp"):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, hidden) if ffn == "swiglu" else nn.Sequential()
        if ffn == "mlp":
            self.mlp.fc1 = nn.Linear(dim, hidden, bias=True)
            self.mlp.fc2 = nn.Linear(hidden, dim, bias=True)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path_p)

    def _ff(self, x): return self.mlp(x) if isinstance(self.mlp, SwiGLU) else self.mlp.fc2(F.gelu(self.mlp.fc1(x)))

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self._ff(self.norm2(x))))
        return x


# ViT-S/B-14 with 4 register tokens; key layout matches Meta's DINOv2 register checkpoints
# (cls_token, register_tokens, pos_embed (1, 1+37^2, dim), mask_token (1, dim), patch_embed.proj,
# blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,ls1,ls2,mlp.fc1,mlp.fc2}, norm).
# Pos embed is bicubically interpolated at runtime to the current patch grid.
# Meta DINOv2 includes a cls pos and uses 37x37 patches; variant_cfg can override this for other ViTs.
class ViT(nn.Module):
    pos_interpolation_antialias = True

    def __init__(self, variant="dinov2_vits14_reg", drop_path_rate=0.0, variant_cfg=None):
        super().__init__()
        cfg = variant_cfg or VIT_VARIANTS[variant]
        dim, depth, heads, pretrain_grid, patch, ffn, pos_has_cls, self.pretrained_url = cfg[:8]
        mlp_ratio, registers = 4.0, cfg[8] if len(cfg) > 8 else 4
        self.variant = variant
        self.patch_size, self.registers, self.embed_dim = patch, registers, dim
        self._pretrain_grid, self._pos_has_cls = pretrain_grid, pos_has_cls
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, int(self._pos_has_cls) + self._pretrain_grid**2, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, p, ffn=ffn) for p in rates)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Which blocks probe_features concatenates. A persistent buffer, so the choice travels in the
        # checkpoint and probe.py picks it up through its existing strict load without knowing about it.
        mask = torch.zeros(depth, dtype=torch.bool)
        mask[[i for i in (4, 6, 8, 11) if i < depth]] = True
        self.register_buffer("probe_layer_mask", mask)

    # Bicubic resample of the checkpoint patch-pos grid to the current (h, w) grid.
    def _interpolate_pos_embed(self, h, w):
        cls_pos = self.pos_embed[:, :1] if self._pos_has_cls else None
        g = self._pretrain_grid
        patch_pos = self.pos_embed[:, int(self._pos_has_cls):].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        # antialias=True matches Meta's default for DINOv2 `_reg` variants.
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False, antialias=self.pos_interpolation_antialias)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(self.pos_embed.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1) if cls_pos is not None else patch_pos

    # Build [cls, registers, patches] tokens. `keep_idx` (B, V) selects which patch positions survive:
    # MAE-style, masked patches are dropped before the blocks rather than replaced by a mask token, so
    # the encoder never spends compute on them. Positional embeddings are added BEFORE the gather, so the
    # surviving tokens carry their true grid positions and absence is what marks a hole.
    # `mask_token` is consequently unused (it stays only because Meta's checkpoint carries it and
    # load_pretrained is a strict load); it gets no gradient and AdamW skips it.
    def _prepare_tokens(self, x, keep_idx=None):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        pos = self._interpolate_pos_embed(h, w)
        cls_pos, patch_pos = (pos[:, :1], pos[:, 1:]) if self._pos_has_cls else (0.0, pos)
        x = x + patch_pos
        if keep_idx is not None:
            x = x.gather(1, keep_idx[..., None].expand(-1, -1, x.shape[-1]))
        cls = self.cls_token.expand(B, -1, -1) + cls_pos
        return torch.cat([cls, self.register_tokens.expand(B, -1, -1), x], dim=1)

    # Return semantic token groups used by train.py and probe.py.
    # With `keep_idx` set, `patches` holds only the V kept patches, in grid order.
    # `checkpoint=True` re-runs each block under torch.utils.checkpoint to trade compute for memory;
    # useful when the 1-GPU batch of 128 (2 globals + 8 locals) does not fit in 80 GB.
    # align_from: also return the normed CLS of every block from that index on, as (N, L, D), for the
    # cross-view feature-alignment term. LayerNorm is per token, so norming CLS alone matches norm(x)[:, 0].
    def forward(self, x, keep_idx=None, checkpoint=False, align_from=None):
        x = self._prepare_tokens(x, keep_idx)
        cls_layers = []
        for i, blk in enumerate(self.blocks):
            if checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if align_from is not None and i >= align_from:
                cls_layers.append(self.norm(x[:, :1])[:, 0])
        x = self.norm(x)
        out = {
            "cls": x[:, 0],
            "registers": x[:, 1 : 1 + self.registers],
            "patches": x[:, 1 + self.registers :],
        }
        if align_from is not None:
            out["cls_layers"] = torch.stack(cls_layers, dim=1)  # (N, L, D)
        return out

    # Segmentation readout: last-4-block normalized patch tokens fused on the channel axis, at the
    # native patch grid. The probe contract is patches only, so registers are dropped here; probe.py
    # area-pools any non-native grid back, which is why upsampling in this method would be wasted work.
    def encode_image(self, x, checkpoint=False):
        xt, feats = self._prepare_tokens(x), []
        for i, blk in enumerate(self.blocks):
            xt = torch.utils.checkpoint.checkpoint(blk, xt, use_reentrant=False) if checkpoint and self.training else blk(xt)
            if i >= len(self.blocks) - 4:
                feats.append(self.norm(xt)[:, 1 + self.registers:])
        return torch.cat(feats, -1)

    # Selects blocks via probe_layer_mask; set_probe_layers configures it before the checkpoint is written.
    def probe_features(self, x):
        xt, feats = self._prepare_tokens(x), []
        for i, blk in enumerate(self.blocks):
            xt = blk(xt)
            if bool(self.probe_layer_mask[i]):
                feats.append(self.norm(xt[:, :1])[:, 0])
        return torch.cat(feats, dim=-1)

    def set_probe_layers(self, layers):
        depth = len(self.blocks)
        bad = [i for i in layers if not 0 <= int(i) < depth]
        if bad or not len(layers):
            raise ValueError(f"model.probe_layers={list(layers)} invalid: need a non-empty subset of 0..{depth - 1}")
        self.probe_layer_mask.zero_()
        self.probe_layer_mask[[int(i) for i in layers]] = True
        return self

    # probe.py is a locked path: it builds a plain ViT and strict-loads the run's checkpoint.
    # CLS/patch specialization is purely structural -- no forward override -- so a plain ViT rewrapped to
    # match the incoming keys *is* the specialized model. Adopting that structure from the state dict lets
    # specialized checkpoints load through the locked path; a plain checkpoint rewraps nothing.
    def load_state_dict(self, state_dict, *args, **kwargs):
        blocks = range(len(self.blocks))
        if any(f"blocks.{i}.norm1.cls.weight" in state_dict for i in blocks):
            specialize_cls_weights(self, sum(f"blocks.{i}.attn.qkv.cls.weight" in state_dict for i in blocks))
        return super().load_state_dict(state_dict, *args, **kwargs)


# Runs `cls` weights over the leading [CLS] token and `patch` weights over the register+patch tail,
# then rejoins the sequence. The token layout is [cls, registers, patches], so the split is the single
# slice at index 1. Wrapping a token-wise layer (LayerNorm, LayerScale, qkv Linear) this way means
# Block and Attention need no forward changes, and attention still sees one unmasked sequence so SDPA
# keeps its Flash-2 path. The two copies are loaded from the same pretrained tensor, so they only
# diverge through training. Splitting a Linear costs no extra FLOPs: every token still passes through
# exactly one.
class Specialized(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.patch, self.cls = layer, deepcopy(layer)

    def forward(self, x):
        return torch.cat([self.cls(x[:, :1]), self.patch(x[:, 1:])], dim=1)


# CLS/patch weight specialization (Marouani et al.): [CLS] carries global semantics while patch tokens
# carry local ones, so give them their own LayerNorm and LayerScale weights in every block (the paper's
# "specialized normalization") plus their own qkv projection in the first `qkv_blocks` blocks, where
# specialization pays off.
# Registers group with the patches: they are filled from the patch field and carry its activation
# statistics, so only the readout token [CLS] gets the specialized weights.
# Idempotent, so it is a no-op on an already-specialized model (resume, `load_pretrained`).
def specialize_cls_weights(model, qkv_blocks):
    for i, blk in enumerate(model.blocks):
        if isinstance(blk.norm1, Specialized):
            continue
        blk.norm1, blk.norm2 = Specialized(blk.norm1), Specialized(blk.norm2)
        blk.ls1, blk.ls2 = Specialized(blk.ls1), Specialized(blk.ls2)
        if i < qkv_blocks:
            blk.attn.qkv = Specialized(blk.attn.qkv)
    return model


class SpecializedViT(ViT):
    def __init__(self, variant="dinov2_vits14_reg", drop_path_rate=0.0, qkv_blocks=4, variant_cfg=None):
        super().__init__(variant, drop_path_rate, variant_cfg)
        specialize_cls_weights(self, qkv_blocks)


# Strict-load the model's declared pretrained weights; incompatible layouts fail loudly.
# The rewrite points both halves of every Specialized layer at the one pretrained tensor, so a fresh
# SpecializedViT is numerically identical to DINOv2 at step 0; it is a no-op for a plain ViT.
def load_pretrained(model):
    state = torch.hub.load_state_dict_from_url(model.pretrained_url, progress=False, map_location="cpu")
    own = model.state_dict()
    # probe_layer_mask is ours; carry it through so the load can stay strict on everything Meta ships.
    state = {k: own[k] if k == "probe_layer_mask" else state[k.replace(".cls.", ".").replace(".patch.", ".")] for k in own}
    model.load_state_dict(state, strict=True)
    return model


# DINO projection head: 3-layer MLP (in -> hidden -> hidden -> bottleneck) + L2 norm +
# weight-normed Linear(bottleneck -> n_prototypes) with weight_g frozen at 1, matching the
# behaviour of dinov2.layers.DINOHead. Standalone reimplementation (no xformers, no fvcore).
# Optional penalty on the prototype bank itself, run once per step from compute_losses and added to the
# total. Any weighting belongs inside the regularizer so the train script just adds the returned scalar.
class PrototypeRegularizer(nn.Module):
    # protos: the raw bank, (n_prototypes, D). Terms that only need direction normalise it themselves.
    def forward(self, protos):
        raise NotImplementedError


class NoPrototypeReg(PrototypeRegularizer):
    def forward(self, protos):
        return protos.new_zeros(())


# MCR^2 coding rate, negated so minimising spreads the bank across its subspace.
class CodingRateReg(PrototypeRegularizer):
    def __init__(self, weight, eps=0.5):
        super().__init__()
        self.weight, self.eps = float(weight), float(eps)

    def forward(self, protos):
        with torch.autocast(device_type=protos.device.type, enabled=False):
            p = F.normalize(protos.float(), dim=-1, p=2)
            k, d = p.shape
            gram = p @ p.T if k < d else p.T @ p  # det(I + AB) = det(I + BA), so take the smaller side
            eye = torch.eye(gram.shape[-1], device=p.device, dtype=p.dtype)
            return -self.weight * 0.5 * torch.logdet(eye + (d / (k * self.eps ** 2)) * gram)


# Sliced-Wasserstein normality test on (G, B, D): standardise, project onto K random directions, and match
# each projection's order statistics to the standard-normal quantiles.
class VISReg(nn.Module):
    def __init__(self, num_projections: int = 256, scale_weight: float = 1.0, shape_weight: float = 1.0, center_weight: float = 1.0):
        super().__init__()
        self.K = num_projections
        self._cached_B = -1
        self._cached_target = None
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight

    def _get_target(self, B: int, device) -> torch.Tensor:
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return self.scale_weight * scale_loss + self.shape_weight * shape_loss + self.center_weight * center_loss


# Drives the raw bank toward N(0, I), whose directions are uniform on the sphere once the head normalises.
class SlicedWassersteinReg(PrototypeRegularizer):
    def __init__(self, weight, projections=256, scale_weight=1.0, shape_weight=1.0, center_weight=1.0):
        super().__init__()
        self.weight = float(weight)
        self.vis = VISReg(int(projections), float(scale_weight), float(shape_weight), float(center_weight))

    def forward(self, protos):
        with torch.autocast(device_type=protos.device.type, enabled=False):
            return self.weight * self.vis(protos.float().unsqueeze(0))


# VISReg over a FIFO queue of past CLS embeddings, so the normality statistic sees far more samples than
# one batch holds. Only the current rows carry gradient; queued rows are detached and stale.
class QueuedVISReg(nn.Module):
    def __init__(self, dim, weight, queue_size=4096, num_projections=256, scale_weight=1.0, shape_weight=1.0, center_weight=1.0):
        super().__init__()
        self.weight = float(weight)
        self.vis = VISReg(int(num_projections), float(scale_weight), float(shape_weight), float(center_weight))
        self.register_buffer("queue", torch.zeros(int(queue_size), dim), persistent=False)
        self.register_buffer("fill", torch.zeros((), dtype=torch.long), persistent=False)
        self.ptr = 0

    def _push(self, z):
        k = self.queue.shape[0]
        idx = (torch.arange(z.shape[0], device=z.device) + self.ptr) % k
        self.queue.index_copy_(0, idx, z.to(self.queue.dtype))
        self.ptr = (self.ptr + z.shape[0]) % k
        self.fill.fill_(min(k, int(self.fill) + z.shape[0]))

    def forward(self, z, enqueue=True):
        with torch.autocast(device_type=z.device.type, enabled=False):
            n = int(self.fill)
            pool = torch.cat([z.float(), self.queue[:n]]) if n else z.float()
            loss = self.weight * self.vis(pool.unsqueeze(0))
        if enqueue:
            self._push(z.detach())
        return loss


PROTOTYPE_REGULARIZERS = {"none": NoPrototypeReg, "coding_rate": CodingRateReg, "swd": SlicedWassersteinReg}


# Keys under dino.prototype_reg map straight to constructor kwargs; "none" ignores whatever is left set.
def make_prototype_regularizer(kind, **kwargs):
    if kind not in PROTOTYPE_REGULARIZERS:
        raise ValueError(f"unknown dino.prototype_reg.kind={kind!r}; expected one of {sorted(PROTOTYPE_REGULARIZERS)}")
    return NoPrototypeReg() if kind == "none" else PROTOTYPE_REGULARIZERS[kind](**kwargs)


class FactoredDINOHead(nn.Module):
    def __init__(self, in_dim, n_prototypes, hidden_dim=2048, bottleneck_dim=384, nlayers=3, n_factors=1, regularizer=None):
        super().__init__()
        if bottleneck_dim % n_factors:
            raise ValueError("Bottleneck dim must be divisible by n_factors")

        if n_prototypes % n_factors:
            raise ValueError("Number of prototypes must be divisible by n_factors")

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        # nlayers includes the first and bottleneck layers
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, bottleneck_dim // n_factors))
        self.n_factors = n_factors
        self.regularizer = regularizer or NoPrototypeReg()

    # L2-normalised bank, split into one codebook per factor.
    def _chunked_prototypes(self):
        return torch.tensor_split(F.normalize(self.prototypes, dim=-1, p=2), self.n_factors, dim=0)

    # Prototype-bank penalty over the WHOLE bank, (n_prototypes, D), not per factor: spreading every
    # prototype against every other is what pushes the factors' codebooks apart, so the chunks are pressed
    # toward encoding mutually exclusive concepts rather than re-deriving each other's.
    # Depends only on the parameter, never on x, so the train script calls it once per step instead of
    # picking it up as a side effect of whichever forward happened to run last.
    def prototype_reg(self):
        return self.regularizer(self.prototypes)

    def forward(self, x):
        # x: [B E]
        x = self.mlp(x)
        chunks = einops.rearrange(F.normalize(x, dim=-1, p=2), 'b (n k) -> b n k', n=self.n_factors)
        chunked_prototypes = self._chunked_prototypes()
        out = torch.stack([chunks[:, i, :] @ chunked_prototypes[i].T for i in range(self.n_factors)], dim=1)

        return out  # B N K_P


# CrossMAE decoder block: queries attend into `ctx` and never to each other, so a masked position's only
# route to information is the visible field -- there is no self-attention path by which masked positions
# could pool each other's guesses. `ctx` is visible-only by construction (the encoder dropped the rest),
# so no key mask is needed and SDPA keeps its Flash backend. No LayerScale/DropPath: the wrapped Block
# ran with drop_path=0.0 and gamma=1, so both were identities here.
class CrossBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.heads = heads
        self.norm1, self.norm_ctx, self.norm2 = (nn.LayerNorm(dim, eps=1e-6) for _ in range(3))
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, dim * 2), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x, ctx):
        B, N, C = x.shape
        q = self.q(self.norm1(x)).reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        k, v = self.kv(self.norm_ctx(ctx)).reshape(B, ctx.shape[1], 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(attn)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


# I-JEPA predictor head: regresses EMA-teacher patch representations at masked target blocks from the
# student's visible-only encoding. CrossMAE-style: the K masked positions are the queries, the V encoded
# visible tokens are the keys/values, held fixed at the encoder output for every block rather than re-read
# from the evolving query stream. The encoder never sees masked positions, so queries cannot be gathered
# from it -- they are built here as a shared learned token plus a positional embedding indexed by
# `mask_idx`, which is the only thing telling a query which patch it is responsible for.
# FINO/JEPA-T option: n_cond>0 adds a learned per-class embedding (idx 0 = missing/-1)
# of a discrete metadata factor to both sides, so the latent-regression target is metadata-aware
# (a dense-path alternative to CLS-token steering). n_cond=0 is plain I-JEPA.
class JEPAPredictor(nn.Module):
    def __init__(self, dim, n_pos, depth=4, width=0, heads=6, n_cond=0):
        super().__init__()
        width = width or dim
        self.proj_in = nn.Linear(dim, width) if width != dim else nn.Identity()
        self.cond_emb = nn.Embedding(n_cond + 1, width) if n_cond else None
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, n_pos, width))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(CrossBlock(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, dim, bias=True)

    def forward(self, visible_tokens, mask_idx, cond=None):
        ctx = self.proj_in(visible_tokens)  # (B, V, width)
        q = self.query + self.pos.expand(ctx.shape[0], -1, -1).gather(1, mask_idx[..., None].expand(-1, -1, ctx.shape[-1]))
        if self.cond_emb is not None and cond is not None:
            c = self.cond_emb(cond + 1).unsqueeze(1)  # broadcast factor embedding over tokens; cond=-1 -> idx 0
            ctx, q = ctx + c, q + c
        for blk in self.blocks:
            q = blk(q, ctx)
        return self.proj(self.norm(q))
