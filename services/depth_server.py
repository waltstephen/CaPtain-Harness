#!/usr/bin/env python3
"""Monocular metric depth HTTP service (MoGe-2) for cameras without a depth sensor.

Used only by the hardware backend (`hardware_real`) for cameras configured with `"depth": "mono"`
(`src/captain_harness/adapters/hardware/mono_depth.py`, model name "moge2"; the env server reaches it through
`--hardware-depth-url`, default $CAPTAIN_DEPTH_URL or http://127.0.0.1:8130). The simulator backends use rendered depth.

POST /estimate_depth  {"rgb_png_base64": str, "intrinsics_3x3": [9 floats] | null, "model": str | null}
  -> {"depth_npy_base64": str (float32 HxW, meters, camera optical z; NaN = invalid), "model": str,
      "uses_intrinsics": bool, "elapsed_s": float, "valid_ratio": float}
GET /health -> {"status": "ok", ...}

Inputs are the RGB frame and (optionally) the pinhole intrinsics. No simulator depth, scene state or object pose is
sent to or used by this service. With known intrinsics the horizontal FoV is passed to MoGe as fov_x, so only
depth/shift is predicted; mono_depth.py then fits one scale per frame to the support plane.

Model: Ruicheng/moge-2-vitl-normal (MoGe-2 ViT-L), the checkpoint used for the real-robot runs
(scripts/models/download_moge.sh). Needs torch, the MoGe code (github.com/microsoft/MoGe), fastapi, uvicorn, pillow.

    python services/depth_server.py --checkpoint "$MOGE_CHECKPOINT" --port 8130
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import threading
import time

import numpy as np
from PIL import Image

MODEL_NAME = "moge2"
DEFAULT_CHECKPOINT = "Ruicheng/moge-2-vitl-normal"


class MoGeEstimator:
    def __init__(self, device: str, checkpoint: str, resolution_level: int):
        import torch
        from moge.model.v2 import MoGeModel

        self.torch = torch
        self.device = torch.device(device)
        self.resolution_level = resolution_level
        # a local model.pt path or an HF repo id (downloaded into the HF cache)
        self.model = MoGeModel.from_pretrained(checkpoint).to(self.device).eval()
        self._lock = threading.Lock()

    def estimate(self, rgb: np.ndarray, intrinsics: np.ndarray | None) -> tuple[np.ndarray, bool]:
        torch = self.torch
        h, w = rgb.shape[:2]
        with self._lock, torch.inference_mode():
            image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
            fov_x = None
            if intrinsics is not None:
                fov_x = math.degrees(2.0 * math.atan(w / (2.0 * float(intrinsics[0, 0]))))
            out = self.model.infer(image, fov_x=fov_x, resolution_level=self.resolution_level, use_fp16=True)
            depth = out["depth"].float().cpu().numpy()
            mask = out.get("mask")
            if mask is not None:
                depth = np.where(mask.cpu().numpy() > 0.5, depth, np.nan)
            return depth.astype(np.float32), fov_x is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint", default=os.environ.get("MOGE_CHECKPOINT") or DEFAULT_CHECKPOINT,
                    help="local model.pt or HF repo id [$MOGE_CHECKPOINT or Ruicheng/moge-2-vitl-normal]")
    ap.add_argument("--resolution-level", type=int, default=9, help="MoGe-2 token budget level 0-9")
    args = ap.parse_args()

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    est = MoGeEstimator(args.device, args.checkpoint, args.resolution_level)

    class Request(BaseModel):
        rgb_png_base64: str
        intrinsics_3x3: list[float] | None = None
        model: str | None = None

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "default_model": MODEL_NAME, "loaded": [MODEL_NAME], "device": str(est.device)}

    @app.post("/estimate_depth")
    def estimate(req: Request):
        if req.model not in (None, MODEL_NAME):
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}; this service serves {MODEL_NAME!r}")
        t0 = time.time()
        try:
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(req.rgb_png_base64))).convert("RGB"))
            intr = None if req.intrinsics_3x3 is None else np.asarray(req.intrinsics_3x3, dtype=np.float64).reshape(3, 3)
            depth, used = est.estimate(rgb, intr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        buf = io.BytesIO()
        np.save(buf, depth.astype(np.float32), allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0)
        return {"depth_npy_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "model": MODEL_NAME, "uses_intrinsics": used,
                "elapsed_s": round(time.time() - t0, 4), "valid_ratio": float(valid.mean())}

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
