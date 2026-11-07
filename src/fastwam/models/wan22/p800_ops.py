"""Optional P800 training operators. Original model paths remain the fallback."""

import os
import sys

import torch


_reported = set()


def report_hit(name):
    if name not in _reported:
        _reported.add(name)
        print(f"P800_OPERATOR_HIT {name}", flush=True)


def enabled(name):
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def supported(x):
    # This deployment exposes Kunlunxin through CUDA compatibility, not torch.xpu.
    return (
        "torch_xmlir" in sys.modules
        and x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and x.shape[-1] == 3072
        and x.is_contiguous()
    )


def rmsnorm(x, weight, eps, scale):
    from torch_xmlir.nn.rms_norm import RMSNormFunction

    # Preserve upstream's cast to input dtype before the learned weight.
    report_hit("rmsnorm")
    return RMSNormFunction.apply(x, eps, scale, False, False) * weight


def rotary(x, freqs, num_heads):
    from torch_xmlir.nn.rotary_no_freqs_pos_emb import RotaryNoFreqsPosEmbABABFunction

    version = None if freqs.is_inference() else freqs._version
    cached = None if freqs.is_inference() else getattr(freqs, "_p800_rotations", None)
    if cached is None or cached[0] != version:
        # Full head width, interleaved (x0,x1), with no trigonometry on device.
        with torch.inference_mode(False):
            cosine = freqs.real.float().repeat_interleave(2, -1).unsqueeze(0).contiguous()
            sine = freqs.imag.float().repeat_interleave(2, -1).unsqueeze(0).contiguous()
        cached = (version, cosine, sine)
        if not freqs.is_inference():
            freqs._p800_rotations = cached
    _, cosine, sine = cached
    report_hit("rope")
    batch, sequence, hidden = x.shape
    y = RotaryNoFreqsPosEmbABABFunction.apply(
        x.float().reshape(batch, sequence, num_heads, hidden // num_heads),
        sine,
        cosine,
    )
    return y.reshape_as(x).to(x.dtype)
