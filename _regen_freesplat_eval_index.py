"""Regenerate FreeSplat eval index JSON for the given num_context_views."""
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--num-context", type=int, default=30)
p.add_argument("--num-frames", type=int, default=140)
p.add_argument("--scene-key", default="v32_data3_00")
p.add_argument("--out-path", type=Path,
               default=Path(r"C:\Users\mgallai\Downloads\FreeSplat\assets\evaluation_index_scannet_30views.json"))
args = p.parse_args()

ctx_idx = np.linspace(0, args.num_frames - 1, args.num_context, dtype=int).tolist()
tgt_idx = [i for i in range(args.num_frames) if i not in set(ctx_idx)]
data = {args.scene_key: {"context": ctx_idx, "target": tgt_idx}}
args.out_path.parent.mkdir(parents=True, exist_ok=True)
args.out_path.write_text(json.dumps(data, indent=2))
print(f"wrote {args.out_path}: context={len(ctx_idx)}, target={len(tgt_idx)}")
