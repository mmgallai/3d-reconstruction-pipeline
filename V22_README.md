# V22 — AGS-Mesh splat experiment

V22 attempted to replace V14-V17's `splatfacto-big` with **AGS-Mesh** (DN-Splatter's surface-aligned 2D Gaussian variant with depth + normal supervision). Both expert opinions ranked this kind of change as their #1 recommendation.

## Result — regression

- ~260 K Gaussians (vs V14-V17's ~5 M)  → 20× fewer
- 10.7 MB pruned splat (vs V14-V17's ~725 MB)
- Depth and normal supervision DID run (every Gaussian was depth+normal supervised)
- Densification was DISABLED — DN-Splatter's densification code uses old gsplat 1.0 intermediate tensors (`xys`, `conics`, `max_2Dsize`) that modern gsplat 1.5 (which we need for Blackwell sm_120 GPU support) doesn't expose

Visually: V22 splat is geometrically more surface-aligned where it has coverage, but the coverage is far sparser than V6 (DA3-only) or V14-V17 (hybrid). Net: not a usable replacement for the canonical splats.

## To reproduce V22 yourself

1. Clone DN-Splatter into your project:
   ```bash
   git clone https://github.com/maturk/dn-splatter.git
   ```
2. Apply our source patches (8 changes that port the codebase from gsplat 1.0 → 1.5):
   ```bash
   cd dn-splatter
   git apply ../patches/dn_splatter_for_gsplat_15.patch
   cd ..
   ```
3. Run the pipeline with the new method:
   ```bash
   python reconstruct_realityscan.py --skip-mvs --train-method ags-mesh
   ```

The patch file `patches/dn_splatter_for_gsplat_15.patch` captures the 8 source patches we made to DN-Splatter to compile against modern gsplat:

| # | Patch | What it fixes |
|---|---|---|
| 1 | Lazy `rasterize_gaussians` import | Modern gsplat doesn't export it |
| 2 | Inline `quat_to_rotmat`, `num_sh_bases` | `gsplat.cuda_legacy.*` is gone |
| 3 | Guard `predict_normals` branch | Depends on legacy rasterize_gaussians API |
| 4 | Squeeze `normals_im` batch dim | Was BxHxWx3, downstream expects HxWx3 |
| 5 | Init `normals_im` on cuda device | Default was CPU, caused device mismatch |
| 6 | Fallback `confidence = ones_like(depth_gt)` | AGS-Mesh strategy requires confidence map |
| 7 | Fallback `gt_normal = surface_normal` | When dataparser doesn't load normals |
| 8 | Skip densification when tracking attrs are None | xys/conics/max_2Dsize no longer exposed |

## What you can ALSO change to reproduce more faithfully

`depths_2/` directory must exist next to `images_2/` (DN-Splatter's NormalNerfstudio dataparser looks for it):
```bash
cp -r colmap/dense/depths colmap/dense/depths_2
```

In `dn-splatter/dn_splatter/dn_config.py` the `dn_splatter_big` config uses `continue_cull_post_densification` which was removed in nerfstudio 1.1.5 — comment it out or accept that it errors at load time (ags-mesh config doesn't use it, so this only matters for the `dn-splatter-big` variant).

## See also

- **V6 (`git checkout v6`)** = the canonical splat (DA3-only init).
- **V17 (`git checkout v17`)** = the canonical mesh deliverable.
- **V22 (this branch)** = the AGS-Mesh experiment, kept as a reference.
