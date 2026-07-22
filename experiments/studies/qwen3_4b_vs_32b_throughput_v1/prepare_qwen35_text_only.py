#!/usr/bin/env python3
"""Export the verified Qwen3.5 text policy as a standalone causal-LM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    loaded = AutoModelForCausalLM.from_pretrained(
        args.source,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Transformers did not return model loading information")
    model, loading_info = loaded
    if model.__class__.__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(f"expected Qwen3_5ForCausalLM, got {model.__class__.__name__}")

    contract = {
        "missing_keys": list(loading_info.get("missing_keys") or []),
        "unexpected_keys": list(loading_info.get("unexpected_keys") or []),
        "mismatched_keys": list(loading_info.get("mismatched_keys") or []),
        "error_msgs": list(loading_info.get("error_msgs") or []),
    }
    if any(contract.values()):
        raise RuntimeError(f"weight loading contract failed: {contract}")

    visual_modules = [name for name, _ in model.named_modules() if name == "visual" or name.startswith("visual.")]
    if visual_modules:
        raise RuntimeError(f"text model unexpectedly contains visual modules: {visual_modules[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False, use_fast=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output)

    manifest = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "model_class": model.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dtype_set": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "visual_module_count": len(visual_modules),
        "loading_contract": contract,
        "config_architectures": model.config.architectures,
        "config_model_type": model.config.model_type,
        "source_is_unchanged": True,
    }
    (args.output / "CAPA_TEXT_ONLY_EXPORT.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
