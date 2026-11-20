#!/usr/bin/env python3
"""Thin launcher around cap-x's SAM3 FastAPI server that accepts an explicit checkpoint.

cap-x's `capx.serving.launch_sam3_server.main` always calls
`build_sam3_image_model(enable_inst_interactivity=True)`, which downloads the gated
facebook/sam3 checkpoint from HF. We patch the builder so a local file can be used
and the request/response contract stays byte-identical to what the harness expects.

--cache-images N (default 0 = off, the release behaviour): cap-x runs the SAM3 image encoder (set_image) on every
/segment and /segment_point call, and one observation image is usually segmented several times (segment,
mark_point -> segment_point, measure_object_geometry on the other camera, ...). With N > 0 the encoder output is kept
for the N most recently used images (LRU keyed by the decoded pixels, size and mode). Every call gets a fresh state
dict with a shallow copy of the cached backbone outputs; set_text_prompt only adds text entries to that copy, so the
cached tensors are never modified and responses are identical to the uncached server. GET /cache_stats reports hits,
misses and encoder seconds.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import os
import threading
import time


def install_image_cache(processor_cls, max_images: int) -> dict:
    """Wrap processor_cls.set_image with an LRU of its outputs; returns the live stats dict."""
    import numpy as np
    import PIL.Image

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    original = processor_cls.set_image
    cache: collections.OrderedDict[str, dict] = collections.OrderedDict()
    lock = threading.Lock()
    stats = {"hits": 0, "misses": 0, "encode_s": 0.0, "max_images": max_images}

    @functools.wraps(original)
    def set_image(self, image, state=None):
        if state is not None or not isinstance(image, PIL.Image.Image):
            return original(self, image, state)
        digest = hashlib.sha1(np.asarray(image).tobytes())
        digest.update(f"{image.mode}:{image.size}".encode())
        key = digest.hexdigest()
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                stats["hits"] += 1
        if hit is None:
            t0 = time.perf_counter()
            hit = original(self, image, None)
            with lock:
                stats["encode_s"] += time.perf_counter() - t0
                stats["misses"] += 1
                cache[key] = hit
                while len(cache) > max_images:
                    cache.popitem(last=False)
        state = dict(hit)
        state["backbone_out"] = dict(hit["backbone_out"])
        return state

    processor_cls.set_image = set_image
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8114)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None, help="local sam3.pt; if missing, fall back to HF download")
    ap.add_argument("--cache-images", type=int, default=0,
                    help="keep the image-encoder output of the N most recent images (0 = off, release behaviour)")
    args = ap.parse_args()
    if args.cache_images < 0:
        ap.error("--cache-images must be >= 0")

    import capx.serving.launch_sam3_server as srv
    import sam3.model_builder as mb

    ckpt = args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None
    if ckpt:
        srv.build_sam3_image_model = functools.partial(  # type: ignore[attr-defined]
            mb.build_sam3_image_model, checkpoint_path=ckpt, load_from_HF=False
        )
        print(f"[sam3_server] using local checkpoint {ckpt}")
    else:
        print("[sam3_server] no local checkpoint; sam3 will try HF download (needs HF_TOKEN with facebook/sam3 access)")
    if args.cache_images:
        from sam3.model.sam3_image_processor import Sam3Processor

        stats = install_image_cache(Sam3Processor, args.cache_images)

        @srv.app.get("/cache_stats")
        def cache_stats():
            return stats

        print(f"[sam3_server] image-encoder cache of {args.cache_images} images")
    srv.main(device=args.device, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
