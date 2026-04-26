"""
Generate a Word document containing the V5 technical brief — for sharing
with an external expert to diagnose the V5 coordinate-system bug.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from datetime import datetime


doc = Document()
section = doc.sections[0]
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)
section.left_margin   = Cm(2.2)
section.right_margin  = Cm(2.2)

# ─── helpers ──────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def set_cell_text(cell, text, bold=False, color=None, size=10, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    para = cell.paragraphs[0]
    para.alignment = align
    run = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

def add_table(headers, rows, col_widths=None, header_color="1F4E79"):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0]
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        set_cell_bg(c, header_color)
        set_cell_text(c, h, bold=True, color=(255,255,255), size=10,
                      align=WD_ALIGN_PARAGRAPH.CENTER)

    for ri, row in enumerate(rows):
        bg = "F2F2F2" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row):
            c = table.rows[ri+1].cells[ci]
            set_cell_bg(c, bg)
            set_cell_text(c, str(val), bold=(ci==0), size=10)

    if col_widths:
        for i, w in enumerate(col_widths):
            for r in table.rows:
                r.cells[i].width = Inches(w)

    doc.add_paragraph()
    return table

def h1(text):
    p = doc.add_heading(text, level=1)
    return p

def h2(text):
    return doc.add_heading(text, level=2)

def body(text):
    p = doc.add_paragraph(text)
    for r in p.runs: r.font.size = Pt(11)
    return p

def code(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    return p

def bullet(text):
    p = doc.add_paragraph(text, style="List Bullet")
    for r in p.runs: r.font.size = Pt(11)
    return p


# ─── title ────────────────────────────────────────────────────────────────────
title = doc.add_heading("3D Reconstruction Pipeline", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph("Technical Brief — V5 Failure Diagnosis")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(14)
sub.runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
sub.runs[0].bold = True

dt = doc.add_paragraph(datetime.now().strftime("Generated: %Y-%m-%d"))
dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
dt.runs[0].font.size = Pt(10)
dt.runs[0].font.color.rgb = RGBColor(0x60, 0x60, 0x60)

doc.add_paragraph()


# ─── 1. Project Overview ──────────────────────────────────────────────────────
h1("1. Project Overview")

body("Goal: Build an open-source pipeline that converts iPhone room scans into VR-ready 3D content for Meta Quest 3.")

body("Input: A folder of HEIC images captured with the iPhone RealityScan app (185 photos of an indoor room).")

body("Output: Two artifacts:")
bullet("A Gaussian Splat (.ply) — photorealistic VR background that can be navigated in Quest 3")
bullet("A textured mesh (.obj/.ply) — physics-collidable surface for VR object interaction (Poisson reconstruction)")

body("Constraints:")
bullet("Run fully offline on a Windows 11 + WSL2 + Docker workstation (single GPU)")
bullet("No cloud services — must be reproducible on any machine with a CUDA GPU")
bullet("Must match or beat commercial apps (Polycam, RealityScan, Scaniverse) in quality")
bullet("Must use only open-source models and tools")


# ─── 2. Pipeline Architecture ─────────────────────────────────────────────────
h1("2. Pipeline Architecture (10 stages)")

code(
"HEIC images\n"
"    │\n"
"    ▼\n"
"Stage 1:  HEIC → JPEG conversion\n"
"Stage 2:  COLMAP SfM (feature extraction → sparse reconstruction\n"
"          → image undistortion)\n"
"Stage 3:  COLMAP MVS (PatchMatchStereo + StereoFusion)\n"
"          → fused.ply (~5M dense points, COLMAP scale)\n"
"Stage 4:  DA3 monocular depth generation (V5 only)\n"
"          → 130 metric depth maps + da3_init.ply\n"
"Stage 5:  Generate transforms.json (COLMAP → Nerfstudio format)\n"
"Stage 6:  Pre-generate downscaled images (images_2/)\n"
"Stage 7:  3D Gaussian Splatting training (Nerfstudio splatfacto-big,\n"
"          30,000 iterations)\n"
"Stage 8:  Mesh export (Poisson)\n"
"Stage 9:  Splat export\n"
"Stage 10: Auto-prune + version-tag outputs\n"
"    │\n"
"    ▼\n"
"output/splat_v{N}_{tag}.ply  +  output/splat_v{N}_{tag}_pruned.ply"
)


# ─── 3. Files and Code Modules ────────────────────────────────────────────────
h1("3. Files and Code Modules")

h2("Project root")
add_table(
    headers=["File", "Purpose"],
    rows=[
        ["reconstruct_realityscan.py", "Pipeline orchestrator (the 10 stages above)"],
        ["config.py", "Central config: train method, iterations, downscale factor"],
        ["generate_da3_depths.py", "NEW for V5 — runs DA3 inference, builds da3_init.ply, writes scene bounds JSON"],
        ["subsample_ply.py", "Voxel/random subsampling with σ-clipping (used for fused.ply → fused_sub.ply)"],
        ["prune_splat.py", "Post-training Gaussian removal (opacity threshold + spatial outliers)"],
        ["analyze_splat.py", "Per-file quality stats: opacity histogram, ghost ratio, density"],
        ["analyze_comparison.py", "Side-by-side comparison of multiple PLY files (Polycam vs ours)"],
        ["Dockerfile", "Builds nerfstudio-blackwell image (CUDA 12.8 + nerfstudio 1.1.5 + gsplat 1.5.3)"],
    ],
    col_widths=[2.2, 4.5],
)

h2("lib/ modules")
add_table(
    headers=["File", "Purpose"],
    rows=[
        ["colmap_pipeline.py", "Wrappers for COLMAP SfM (feature/match/sparse/undistort) and MVS — all run in Docker"],
        ["colmap_to_ns.py", "Reads cameras.bin/images.bin, applies OpenCV→OpenGL coord flip, writes transforms.json. Now also embeds depth_file_path per frame and scene_box from DA3"],
        ["nerfstudio_pipeline.py", "Builds the Docker ns-train and ns-export commands. Contains TORCH_LOAD_PATCH for PyTorch 2.6 weights_only compat"],
        ["docker_runner.py", "Subprocess wrapper for docker run with logging"],
        ["heic_converter.py", "HEIC → JPEG via pillow_heif"],
    ],
    col_widths=[2.2, 4.5],
)

h2("Models / external dependencies")
add_table(
    headers=["Component", "Version", "Used in"],
    rows=[
        ["COLMAP",            "latest (Docker colmap/colmap:latest)", "SfM + MVS"],
        ["Nerfstudio",        "1.1.5 (in nerfstudio-blackwell)",      "Training orchestration"],
        ["gsplat",            "1.5.3 (in nerfstudio-blackwell)",      "Differentiable Gaussian rasterizer"],
        ["splatfacto-big",    "Nerfstudio built-in",                  "The actual 3DGS model — high-quality variant with degree-3 SH"],
        ["Depth Anything 3",  "da3metric-large",                       "Single-view metric depth estimation, used in V5"],
        ["dn-splatter",       "cloned but unused",                    "Was supposed to add depth+normal supervision; gsplat API incompatible"],
    ],
    col_widths=[1.8, 2.3, 2.7],
)


# ─── 4. Version History ───────────────────────────────────────────────────────
h1("4. Version History")

h2("V1 (sparse init) — baseline")
bullet("Init: COLMAP sparse (~37k SfM points)")
bullet("Result: 1.20 GB raw → 659 MB pruned, 36.2% ghost ratio")
bullet("Problem: keypoint-only init, white walls/floor missed entirely → many floaters")

h2("V2 (sparse init, retry) — same as V1")
bullet("Same setup as V1, used to validate reproducibility")

h2("V3 (MVS init, hand-tuned) — first major improvement")
bullet("Init: COLMAP MVS dense → subsampled to 500k points (fused_sub.ply)")
bullet("Result: 1.14 GB raw → 668 MB pruned, 34% ghost ratio, mean opacity 0.696")
bullet("Improvement over V1: dense MVS init covers textured surfaces that sparse SfM missed")
bullet("Remaining problem: still 34% floaters; bounding volume 67× larger than Polycam")

h2("V4 (MVS init + 3σ outlier clip + cull-alpha-thresh 0.01)")
bullet("Init: same as V3 but with 3σ spatial outlier clipping (removed ~37k stray MVS points)")
bullet("Training: added --pipeline.model.cull-alpha-thresh 0.01 (5× more aggressive mid-training pruning)")
bullet("Method: splatfacto-big, 30k iters, 2× downscale")
bullet("Result: 1.22 GB raw → 753 MB pruned, similar opacity to V3")
bullet("STATUS: working, currently the best stable version")
bullet("Files on disk: output/splat_v4_mvs_500k.ply (1.22 GB), output/splat_v4_mvs_500k_pruned.ply (753 MB)")

h2("V5 (Depth Anything 3 init + scene bounds) — currently broken")
body("What changed:")
bullet("New init source: Replaced MVS fused_sub.ply with a DA3-derived point cloud da3_init.ply")
bullet("DA3 pipeline: Ran da3metric-large on each of 130 undistorted images individually, producing per-frame metric depth maps. Back-projected each pixel into world space using COLMAP camera poses. Merged all 130 frames, applied 3σ outlier clip, voxel-subsampled to 500k points.")
bullet("Scene bounds: Computed AABB from the merged point cloud, embedded as scene_box in transforms.json")
bullet("Per-frame depths: Linked each .npy depth map via depth_file_path in transforms.json (was meant for dn-splatter depth supervision; unused since dn-splatter isn't compatible with our gsplat version)")
bullet("Training: splatfacto-big (NOT dn-splatter — dn-splatter requires gsplat==1.0.0, ours is 1.5.3)")
body("Result: 657 MB raw → 247 MB pruned (63% smaller than V4 pruned).")
body("Numerical metrics looked great: mean opacity 0.706 (highest of any version), 0% ghost ratio after pruning, 11× faster training (43 min vs ~8 hr for V4).")
body("Visual quality: BROKEN — see screenshots, the splat is fragmented and doesn't represent the room.")


# ─── 5. The V5 Problem ────────────────────────────────────────────────────────
h1("5. The V5 Problem")

h2("Symptoms (from user-rendered screenshots)")
bullet("Splat is a small fragmented blob in space, not a recognizable room")
bullet("Visible halo of splats at the top — likely camera frustums rendered as Gaussians")
bullet("Zoomed view inside shows chaotic noise, no surfaces")
bullet("Inconsistent with the metrics: opacity 0.706, ghost 0% — yet visually unusable")

h2("Numerical evidence of the bug")
body("From analyze_comparison.py:")
code(
"Ours_V5_full:\n"
"  Bounding vol: 2.2 cubic units  (1.4 × 1.9 × 0.9)\n"
"  Point density: 1,275,156 pts/unit³\n\n"
"Ours_V3_pruned (MVS init):\n"
"  Bounding vol: 976.8 cubic units  (11.1 × 10.7 × 8.2)\n"
"  Point density: 2,892 pts/unit³\n\n"
"Polycam:\n"
"  Bounding vol: 304.4 cubic units  (7.8 × 4.9 × 7.9)\n"
"  Point density: 4,460 pts/unit³\n\n"
"Camera cluster from COLMAP:\n"
"  center=[0.14, -0.03, 0.1]\n"
"  p90_radius=5.862  ← cameras are 5.86 units from center"
)

body("The cameras live at radius ~5.86 from the origin. The Gaussians ended up packed into a 1.4-unit ball at the origin. The cameras are looking at a tiny shrunken scene from far away — that's why the splat looks like a fragment floating in empty space when rendered.")

h2("Root cause hypothesis: coordinate-system scale mismatch")
body("da3metric-large is a single-view metric depth model. It outputs depth in real-world meters based on its monocular prior — independent of the input scene scale.")

body("COLMAP SfM, in contrast, produces an up-to-scale reconstruction. The cameras live in an arbitrary scale set by the first camera baseline. For phone scans this is roughly metric but not exactly.")

body("When V5's generate_da3_depths.py does this loop:")
code(
"for each image:\n"
"    depth = da3_metric.inference(image)        # in metric meters (DA3 scale)\n"
"    K, w2c = colmap_intrinsics, colmap_pose    # in COLMAP scale\n"
"    pts_world = backproject(depth, K, w2c)     # mixing two scales"
)

body("…the backprojected XYZ is a hybrid: depth from DA3 (metric), camera transform from COLMAP (arbitrary scale). The resulting da3_init.ply is in neither coordinate system cleanly.")

body("When nerfstudio loaded transforms.json with this init cloud + COLMAP cameras + an embedded scene_box from DA3-scale bounds, its auto-normalizer (auto_scale_poses=True) tried to fit everything into a unit sphere. The cameras dominated the normalization (5.86 unit radius → squeezed to ≤1), but the init points were already at metric scale — so the splat collapsed to a tiny ball.")

h2("Supporting evidence")
bullet("DA3's _align_to_input_extrinsics_intrinsics uses Umeyama alignment to scale DA3 output to match input extrinsics. We tried passing extrinsics in V5 but da3metric-large (single-view) crashed when given multi-view input. So we ran it without extrinsics, which means no scale alignment was applied.")
bullet("The DA3 multi-view model (da3nested-giant-large) supports proper alignment, but it's 1.15B params and needs more careful integration.")

h2("What we tried but didn't help")
bullet("3σ outlier clip on the merged DA3 cloud — clip is in absolute units, doesn't fix scale mismatch")
bullet("scene_box in transforms.json — nerfstudio likely ignored or normalized it")
bullet("Per-frame voxel subsampling — improves uniformity but not scale")


# ─── 6. Question for the Expert ───────────────────────────────────────────────
h1("6. Question for the Expert")

body("How should we make a multi-frame Depth-Anything-3-derived point cloud that is co-registered with COLMAP camera poses for use as 3DGS initialization?")

body("Options we considered, but unsure which is best:")

bullet("Option 1 — Compute a per-image scale factor from COLMAP triangulated points: for each image, take the COLMAP 3D points that project into it, compare their COLMAP depth to DA3's predicted depth at the same pixels, take the median ratio, multiply DA3 depths by it. Concern: depends on quality of sparse COLMAP points; per-image scales may be inconsistent.")

bullet("Option 2 — Compute a global scale factor the same way but pooled across all images — single ratio for the whole dataset. Concern: assumes COLMAP scene scale is uniform (it should be, but not guaranteed).")

bullet("Option 3 — Use the DA3 multi-view model (da3nested-giant-large) which supports passing COLMAP extrinsics and uses Umeyama alignment internally. Concern: 1.15B params, GPU memory, multi-view batching.")

bullet("Option 4 — Drop DA3 entirely and just use the existing MVS fused_sub.ply — V4 is already working.")

bullet("Option 5 — Use ARKit poses + LiDAR depth from the iPhone instead of COLMAP+DA3. Requires re-capture with the 3D Scanner App. Concern: changes the input pipeline.")

body("Our preference is option 1 or 2 — they keep the existing capture pipeline and just add a scale-alignment step before back-projection.")


# ─── 7. Files the Expert May Want to See ──────────────────────────────────────
h1("7. Files the Expert May Want to See")

bullet("generate_da3_depths.py — the V5 init generation script with the scale-mismatch bug")
bullet("lib/colmap_to_ns.py — transforms.json builder")
bullet("lib/nerfstudio_pipeline.py — training command")
bullet("analyze_comparison.py output — numerical comparison")
bullet("The two V5 screenshots from SuperSplat")
bullet("output/splat_v5_da3.ply (657 MB) and output/splat_v5_da3_pruned.ply (247 MB)")
bullet("output/splat_v4_mvs_500k_pruned.ply (753 MB) — the working baseline")
bullet("colmap/dense/da3_init.ply (7 MB) — the broken init cloud")
bullet("colmap/dense/da3_bounds.json — the scene AABB used")
bullet("colmap/dense/transforms.json — what nerfstudio actually loaded")

doc.add_paragraph()
final = doc.add_paragraph("V4 is the canonical working output until V5 is fixed.")
final.alignment = WD_ALIGN_PARAGRAPH.CENTER
final.runs[0].bold = True
final.runs[0].font.size = Pt(11)
final.runs[0].font.color.rgb = RGBColor(0x7F, 0x3F, 0x00)


# ─── save ─────────────────────────────────────────────────────────────────────
out = r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\V5_Technical_Brief.docx"
doc.save(out)
print(f"Saved: {out}")
