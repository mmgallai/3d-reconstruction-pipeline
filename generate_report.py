"""
Generate a Word document summarizing the 3D reconstruction pipeline progress.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)

# ── Style helpers ─────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    return p

def body(text):
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(11)
    return p

def bullet(text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.style.font.size = Pt(11)
    return p

def add_table(headers, rows, col_widths=None, header_color="1F4E79"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        set_cell_bg(cell, header_color)
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg = "EBF3FB" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            cell.paragraphs[0].runs[0].font.size = Pt(10)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    doc.add_paragraph()  # spacer
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
title = doc.add_heading("Room-Scale VR Digital Twin — 3D Reconstruction Pipeline", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph("Lab Meeting Progress Report")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(14)
sub.runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

date_p = doc.add_paragraph(datetime.date.today().strftime("%B %d, %Y"))
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.runs[0].font.size = Pt(12)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROJECT GOAL
# ═══════════════════════════════════════════════════════════════════════════════
heading("1. Project Goal", 1)
body(
    "The goal is to build an open-source pipeline that converts a consumer phone scan "
    "(captured with RealityScan / ARKit on an iPhone) of a real room into a fully interactive "
    "VR digital twin deployable on a Meta Quest 3 headset. The system must produce two outputs:"
)
bullet("A photorealistic Gaussian Splat — used as the visual background of the room in VR.")
bullet(
    "A triangulated mesh with physics colliders — used so that users can grab, move, "
    "and interact with key objects (chairs, books, lamps, etc.) inside the VR environment."
)
body(
    "The deliberate choice was to build this pipeline entirely on open-source tools "
    "(COLMAP, Nerfstudio, Docker) rather than relying on commercial photogrammetry software "
    "such as RealityCapture, in order to maintain full control, reproducibility, and zero "
    "per-seat licensing costs."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TECHNICAL FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════
heading("2. Technical Fundamentals", 1)

heading("2.1  3D Gaussian Splatting (3DGS)", 2)
body(
    "3D Gaussian Splatting represents a scene as millions of small, semi-transparent, "
    "coloured ellipsoids (Gaussians). Unlike traditional NeRF (which integrates a neural "
    "radiance field by ray marching), Gaussians are rasterized directly onto the screen — "
    "making real-time rendering feasible at VR frame rates (72+ FPS on a Quest 3). "
    "Each Gaussian stores: position (x,y,z), orientation (quaternion), scale (3 axes), "
    "opacity, and spherical-harmonic colour coefficients for view-dependent appearance."
)

heading("2.2  Structure-from-Motion (SfM) — COLMAP", 2)
body(
    "Before training a Gaussian Splat, we need camera poses: where the camera was located "
    "and which direction it pointed for each photograph. COLMAP computes these automatically "
    "from raw images using SIFT feature detection, exhaustive feature matching across all "
    "image pairs, and sparse bundle adjustment. The result is a sparse 3D point cloud "
    "and calibrated camera poses — the essential inputs for any neural rendering pipeline."
)

heading("2.3  Multi-View Stereo (MVS) — Dense Reconstruction", 2)
body(
    "COLMAP's SfM stage gives only a sparse cloud (~30,000–100,000 points). "
    "Multi-View Stereo (specifically PatchMatchStereo) densifies this by computing "
    "per-pixel depth maps for every image using GPU-accelerated patch matching. "
    "StereoFusion then merges these depth maps into a single dense point cloud "
    "(fused.ply) with 5–30 million points. This dense initialisation is critical for "
    "high-quality Gaussian Splatting because Gaussians anchored to real geometry "
    "converge faster and produce far fewer floating artefacts."
)

heading("2.4  Coordinate System Conventions", 2)
body(
    "COLMAP uses OpenCV convention (Y-axis pointing down, Z-axis into the scene). "
    "Nerfstudio uses OpenGL convention (Y-axis up, Z-axis out of the scene). "
    "Converting between them requires flipping columns 1 and 2 of the camera-to-world "
    "rotation matrix. We implemented this conversion manually in a custom colmap_to_ns.py "
    "module that reads COLMAP binary files and outputs a transforms.json file."
)

heading("2.5  Spherical Harmonics for Colour", 2)
body(
    "Gaussian Splats store colour as spherical harmonic (SH) coefficients rather than "
    "a single RGB value. This allows the colour of each Gaussian to change with viewing "
    "angle — capturing specular highlights, reflections, and lighting variation. "
    "splatfacto-big uses degree-3 SH (45 coefficients per colour channel), giving "
    "high-fidelity view-dependent appearance at the cost of a larger model file."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. STARTING POINT
# ═══════════════════════════════════════════════════════════════════════════════
heading("3. Starting Point — Version 1 Pipeline", 1)
body(
    "The initial pipeline was a single Python script (~250 lines) that orchestrated "
    "Docker containers for COLMAP and Nerfstudio. It was a proof-of-concept that covered "
    "the end-to-end flow but had significant limitations that prevented production-quality output."
)

heading("3.1  V1 Pipeline Architecture", 2)

add_table(
    headers=["Stage", "Tool", "What it did", "Limitation"],
    rows=[
        ["1. Image staging",     "Python / shutil",       "Copy raw JPG/PNG images to workspace",                          "HEIC files not supported (RealityScan output)"],
        ["2. Unmasked SfM",      "COLMAP (Docker)",        "Feature extract → match → sparse reconstruct → undistort",      "hardcoded to sparse/0 — fails if best model is sparse/1 or sparse/2"],
        ["3. SAM3 masking",      "SAM3 (local GPU)",       "Text-prompted segmentation to mask foreground objects",          "Required SAM3 installed locally; fragile model loading"],
        ["4. Masked SfM",        "COLMAP (Docker)",        "Re-run SfM on undistorted images with masks",                   "Often failed or produced empty models with complex scenes"],
        ["5. Nerfstudio train",  "splatfacto (Docker)",    "30,000 iterations at 4× downscale",                             "No dense init; no MVS; 4× downscale loses fine detail"],
        ["6. Splat export",      "ns-export (Docker)",     "Export gaussian-splat → splat.ply",                             "Torch weights_only bug caused crashes without the patch"],
    ],
    col_widths=[1.2, 1.3, 2.5, 2.5],
    header_color="1F4E79",
)

heading("3.2  Key Problems with V1", 2)
bullet("No HEIC support — RealityScan exports HEIC + XMP; the script only accepted JPG/PNG.")
bullet("Hardcoded sparse/0 — COLMAP numbers models by quality, not sequence. The best model is often sparse/1 or sparse/2, causing registration failures.")
bullet("No MVS — Gaussians were initialised from ~37,000 sparse COLMAP points. Without dense geometry, the model fills empty space with semi-transparent ghost Gaussians.")
bullet("SAM3 masking was optional and fragile — required a local model checkpoint and specific Python environment that conflicted with other dependencies.")
bullet("No idempotency — every run deleted and rebuilt all intermediate files from scratch, making re-runs after partial failures very expensive.")
bullet("Alexnet re-download — Nerfstudio's LPIPS loss metric requires downloading a 233 MB alexnet model. Because Docker was run with --rm, the cache was destroyed after every container exit, causing a 10–20 minute hang at 0% GPU on each training attempt.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. WHAT WE TRIED AND WHY IT DIDN'T WORK
# ═══════════════════════════════════════════════════════════════════════════════
heading("4. What We Tried — and Why It Did Not Work", 1)

add_table(
    headers=["Approach", "What we expected", "What actually happened", "Root cause"],
    rows=[
        [
            "DA3 (Depth Anything v3) monocular depth",
            "Per-image depth maps to initialise denser Gaussians without running full MVS",
            "Severe floating artefacts; geometry was incoherent and non-photorealistic",
            "DA3 outputs depth in camera space (scale-ambiguous, no absolute units). Depths from different images are in incompatible scales and cannot be merged into a coherent 3D point cloud.",
        ],
        [
            "ns-train 2dgs (2D Gaussian Splatting)",
            "Surface-oriented flat-disk Gaussians → cleaner mesh extraction via Poisson reconstruction",
            "Exit code 2 — command not found",
            "2DGS is not included in Nerfstudio 1.1.5 built-in methods. The plugin requires a separate installation or a newer nerfstudio build.",
        ],
        [
            "--pipeline.datamanager.cache-images cpu",
            "Pre-load all training images into CPU RAM to speed up GPU data feeding",
            "Training hung or crashed for large image sets",
            "With 130 images at 1512×2016 px, the flag tried to pre-allocate ~10 GB of CPU RAM causing OOM or deadlock before training started.",
        ],
        [
            "--pipeline.model.camera-optimizer.mode SO3xR3",
            "Fine-tune COLMAP camera poses during training to correct small registration errors",
            "Exit code 1 — parameter not accepted by splatfacto-big",
            "splatfacto-big does not expose camera_optimizer as a configurable sub-module in Nerfstudio 1.1.5. The flag is valid for other methods but not this one.",
        ],
        [
            "Masked SfM (V1 Stage 4)",
            "Re-run SfM with SAM3 masks to suppress feature detection on non-room regions",
            "Often produced empty or degenerate sparse models",
            "Masking too aggressively removed features that COLMAP depended on for image matching. With fewer matching features, the mapper failed to register images.",
        ],
        [
            "ARKit poses from XMP sidecars",
            "Use phone-provided camera poses instead of running COLMAP SfM",
            "Not implemented — discarded this path early",
            "ARKit poses are in a device-local coordinate frame with arbitrary scale. Converting to COLMAP/Nerfstudio format requires non-trivial alignment and scale estimation. Fresh COLMAP SfM from images is more robust.",
        ],
        [
            "Docker --rm with no model cache",
            "Clean container on every run",
            "10–20 minute hang at 0% GPU utilisation before training started",
            "Nerfstudio downloads alexnet-owt-7be5be79.pth (233 MB LPIPS model) on first use. --rm destroys the Docker /root/.cache after each run, forcing re-download every single training attempt.",
        ],
    ],
    col_widths=[1.5, 1.5, 2.0, 2.5],
    header_color="C00000",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. WHAT WORKED
# ═══════════════════════════════════════════════════════════════════════════════
heading("5. What Worked — Current Approach", 1)

add_table(
    headers=["Component", "What it does", "Why it works well"],
    rows=[
        [
            "HEIC → JPEG conversion\n(pillow-heif)",
            "Convert RealityScan HEIC+XMP exports to JPEG before any processing",
            "RealityScan produces HEIC files which COLMAP cannot read. pillow-heif decodes HEIC natively on both Mac and Windows without external tools.",
        ],
        [
            "COLMAP SfM — fresh full-resolution",
            "Feature extract (SIFT) → exhaustive match → sparse reconstruct → undistort images",
            "Runs entirely in Docker (colmap/colmap:latest). No ARKit poses needed. PINHOLE camera model + single_camera flag gives stable calibration for phone footage.",
        ],
        [
            "best_sparse_model() selection",
            "Pick the sparse model subfolder with the largest points3D.bin (not always sparse/0)",
            "Fixes the V1 bug where hardcoding sparse/0 caused silent registration failures when COLMAP produced multiple models.",
        ],
        [
            "COLMAP MVS — PatchMatchStereo + StereoFusion",
            "GPU depth estimation for each image → dense fused point cloud (fused.ply)",
            "Produces 5–20 million points vs. 37,000 from sparse SfM. Dense initialisation dramatically reduces ghost Gaussians and improves training convergence.",
        ],
        [
            "Custom colmap_to_ns.py",
            "Read COLMAP binary cameras.bin / images.bin / points3D.bin → transforms.json",
            "Pure-Python, no COLMAP CLI needed. Handles the OpenCV→OpenGL coordinate flip correctly. Embeds ply_file_path for dense Gaussian initialisation.",
        ],
        [
            "Idempotency checks",
            "Skip COLMAP SfM if colmap/dense/sparse/0/cameras.bin already exists",
            "Allows re-running the pipeline after partial failures without waiting 30+ minutes for COLMAP to repeat work that is already done.",
        ],
        [
            "fused.ply backup/restore",
            "Before undistort_images() runs, back up fused.ply; restore after",
            "COLMAP image_undistorter does rm -rf on the dense folder, destroying the 130 MB MVS output. The backup/restore pattern prevents this data loss.",
        ],
        [
            "TORCH_HOME in Docker env",
            "Pass -e TORCH_HOME=/workspace/torch_cache to every docker run",
            "The alexnet LPIPS model (233 MB) is downloaded once and persisted in the mounted workspace directory. All subsequent training and export runs start immediately.",
        ],
        [
            "Pre-generate downscaled images",
            "PIL resize images to images_2/ before training starts",
            "Nerfstudio's nerfstudio-data dataparser expects pre-generated downscaled folders. Without them it either crashes or re-generates them slowly inside the container on first use.",
        ],
        [
            "splatfacto-big at 30,000 iters, 2× downscale",
            "High-quality Gaussian Splat training on 130 registered images",
            "splatfacto-big uses degree-3 spherical harmonics (vs. degree-1 in standard splatfacto) giving better colour fidelity. 2× downscale balances quality and memory usage for ~1500×2000 px images.",
        ],
        [
            "Post-processing prune script",
            "Remove Gaussians with opacity < 0.15 (sigmoid) and spatial outliers >5σ",
            "Eliminates the 36–44% semi-transparent ghost Gaussians that cause haze in the rendered image. Reduces file size from 1.2 GB to 659 MB with no visible loss of real geometry.",
        ],
    ],
    col_widths=[1.8, 2.2, 3.5],
    header_color="375623",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE ARCHITECTURE — CURRENT
# ═══════════════════════════════════════════════════════════════════════════════
heading("6. Current Pipeline Architecture", 1)
body(
    "The pipeline has been completely rewritten from the V1 single-script proof-of-concept "
    "into a modular, idempotent system of ~1,200 lines across six Python modules. "
    "All heavy computation runs inside Docker containers (no local GPU dependencies), "
    "making the pipeline portable across lab machines."
)

add_table(
    headers=["Stage", "Module", "Input", "Output", "Approx. time"],
    rows=[
        ["1. HEIC → JPEG",        "heic_converter.py",      "HEIC + XMP files",                   "JPEG images",                      "2 min"],
        ["2. COLMAP SfM",          "colmap_pipeline.py",     "JPEG images",                         "Undistorted images + sparse model", "15–25 min"],
        ["3. COLMAP MVS",          "colmap_pipeline.py",     "Undistorted images + sparse model",   "fused.ply (5–20M points)",         "25–40 min"],
        ["4. transforms.json",     "colmap_to_ns.py",        "COLMAP binary model + fused.ply",     "Nerfstudio JSON + init PLY",       "< 1 min"],
        ["5. Downscale images",    "nerfstudio_pipeline.py", "Full-res undistorted images",         "images_2/ folder",                 "2 min"],
        ["6. splatfacto-big train","nerfstudio_pipeline.py", "transforms.json + images + fused.ply","Checkpoint (.ckpt)",               "40–55 min"],
        ["7. Splat export",        "nerfstudio_pipeline.py", "Checkpoint",                          "splat.ply (~1 GB)",                "5 min"],
        ["8. Prune splat",         "prune_splat.py",         "splat.ply",                           "splat_pruned.ply (~660 MB)",       "3 min"],
        ["9. Mesh export",         "nerfstudio_pipeline.py", "Checkpoint",                          "mesh.ply / mesh.obj",              "5–10 min"],
    ],
    col_widths=[1.4, 1.6, 1.8, 1.8, 1.0],
    header_color="1F4E79",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CURRENT STATUS
# ═══════════════════════════════════════════════════════════════════════════════
heading("7. Current Status", 1)

add_table(
    headers=["Component", "Status", "Notes"],
    rows=[
        ["HEIC → JPEG conversion",        "✅ Complete",        "185 images converted"],
        ["COLMAP SfM",                     "✅ Complete",        "130 / 185 images registered (70% — expected for complex room)"],
        ["COLMAP MVS (PatchMatchStereo)",  "🔄 Running now",    "Computing GPU depth maps for 130 images (~25–35 min)"],
        ["fused.ply (dense point cloud)",  "🔄 Pending MVS",    "Will produce 5–20M points for dense Gaussian init"],
        ["transforms.json",               "✅ Complete",        "130 frames, coordinate conversion validated"],
        ["splatfacto-big training (V1)",   "✅ Complete",        "30,000 iters from sparse init — baseline result"],
        ["splat.ply (V1 — sparse init)",   "✅ Available",       "1.2 GB, 5M Gaussians, 36% ghost ratio"],
        ["splat_pruned.ply",               "✅ Available",       "659 MB, 2.8M Gaussians — ghosts removed, ready to view"],
        ["splatfacto-big training (V2)",   "⏳ Queued",          "Will start automatically after MVS finishes (~50 min training)"],
        ["splat.ply (V2 — MVS init)",      "⏳ Pending",         "Expected significantly fewer ghost Gaussians"],
        ["Mesh export",                    "⏳ Pending",         "Will run after V2 training completes"],
        ["2DGS (surface-oriented)",        "❌ Not available",   "Not in Nerfstudio 1.1.5 — requires Docker image update"],
        ["Unity VR integration",           "📋 Planned",         "Import splat + mesh → Meta XR SDK → physics colliders"],
    ],
    col_widths=[2.2, 1.4, 4.0],
    header_color="1F4E79",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. QUALITY ANALYSIS — V1 SPLAT
# ═══════════════════════════════════════════════════════════════════════════════
heading("8. Quality Analysis of Current Splat (V1 — Sparse Init)", 1)
body(
    "After training completed, we ran a statistical analysis of the exported splat.ply "
    "to objectively characterise reconstruction quality without needing a visual renderer."
)

add_table(
    headers=["Metric", "Value", "Interpretation"],
    rows=[
        ["Total Gaussians",               "4,998,420",    "Dense coverage — good"],
        ["Spherical Harmonic degree",      "3 (45 coeff)", "Maximum colour fidelity in Nerfstudio"],
        ["Opacity > 0.9 (solid geometry)","25.8% — 1.29M","Core scene geometry — good"],
        ["Opacity 0.15–0.9 (uncertain)",  "38.0% — 1.90M","Partially transparent — contributes to soft/blurry areas"],
        ["Opacity < 0.15 (ghost/floater)","36.2% — 1.81M","⚠ High — caused by sparse initialisation"],
        ["Median Gaussian scale",         "0.007 units",  "Very fine detail — positive indicator"],
        ["Large Gaussians (scale > 1.0)", "0.0%",         "No massive floater blobs — good"],
        ["Spatial outliers (> 5σ)",       "~0.4%",        "Low — minor fringe noise only"],
        ["Post-prune Gaussians kept",     "2,786,436",    "55.7% of originals — solid geometry retained"],
    ],
    col_widths=[2.5, 1.8, 3.2],
    header_color="7F6000",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════════════
heading("9. Next Steps", 1)

add_table(
    headers=["Priority", "Task", "Expected improvement", "Effort"],
    rows=[
        [
            "1 — Immediate",
            "Complete MVS + re-train with dense init (in progress)",
            "Ghost Gaussian ratio should drop from 36% to <10%. Sharper surfaces, better geometry for mesh extraction.",
            "Automated — running now",
        ],
        [
            "2 — Short term",
            "Enable 2D Gaussian Splatting (2DGS)",
            "Surface-aligned disk Gaussians produce much cleaner Poisson mesh extraction — critical for physics colliders.",
            "Update Dockerfile to install gsplat 2DGS extension or use a newer nerfstudio version",
        ],
        [
            "3 — Short term",
            "Mesh extraction and cleanup",
            "Physics-ready triangulated mesh for VR object interaction.",
            "ns-export poisson → mesh decimation to 300k faces → Unity import",
        ],
        [
            "4 — Medium term",
            "Unity VR integration — Gaussian Splat background",
            "Photorealistic room visible in Quest 3 headset.",
            "Unity gaussian-splatting package + Meta XR SDK + Quest 3 build",
        ],
        [
            "5 — Medium term",
            "Unity VR integration — mesh colliders for objects",
            "Users can grab and interact with room objects in VR.",
            "Mesh import → MeshCollider → XR Interaction Toolkit → Quest 3 deploy",
        ],
        [
            "6 — Future",
            "Per-object segmentation and individual mesh extraction",
            "Separate physics bodies per object rather than one monolithic room mesh.",
            "SAM3 / Segment Anything 3D → per-object Gaussian cluster → individual mesh",
        ],
    ],
    col_widths=[1.3, 2.0, 2.7, 1.5],
    header_color="1F4E79",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. KEY DESIGN DECISIONS
# ═══════════════════════════════════════════════════════════════════════════════
heading("10. Key Design Decisions and Justifications", 1)

heading("10.1  Open-source vs. RealityCapture", 2)
body(
    "RealityCapture (Epic Games) offers a one-click photogrammetry pipeline producing "
    "high-quality textured meshes. We evaluated this option but chose to build our own "
    "pipeline for the following reasons:"
)
bullet("Cost: RealityCapture is priced per output export beyond the free tier.")
bullet("Control: We need to integrate the reconstruction output directly into a custom VR pipeline and Unity project.")
bullet("Flexibility: Our pipeline can be extended with custom segmentation, per-object mesh extraction, and automated processing of new scans without manual GUI steps.")
bullet("Reproducibility: All steps run in Docker containers and are fully scriptable and version-controlled.")

heading("10.2  COLMAP over ARKit poses", 2)
body(
    "RealityScan captures ARKit camera poses embedded in XMP sidecars alongside each image. "
    "We chose to discard these and run fresh COLMAP SfM instead because:"
)
bullet("ARKit poses are in a device-local coordinate frame with arbitrary scale — aligning them to a metric world coordinate system requires non-trivial calibration.")
bullet("COLMAP's bundle adjustment refines camera intrinsics jointly with poses, correcting lens distortion more accurately than ARKit's on-device estimates.")
bullet("Fresh SfM from images is fully repeatable and does not depend on ARKit being available or the XMP format remaining stable.")

heading("10.3  Gaussian Splatting over NeRF", 2)
body(
    "Neural Radiance Fields (NeRF) produce photorealistic novel views but require seconds "
    "per frame to render — far too slow for VR. Gaussian Splatting renders at 72–120 FPS "
    "on consumer hardware by rasterising pre-computed ellipsoids rather than integrating "
    "a neural network per pixel. For a Quest 3 VR headset with a strict 72 Hz frame budget, "
    "Gaussian Splatting is the only photorealistic option currently feasible."
)

heading("10.4  MVS over monocular depth (DA3)", 2)
body(
    "We initially explored using Depth Anything v3 (DA3) — a state-of-the-art monocular "
    "depth estimator — to generate dense depth maps without running the computationally "
    "expensive COLMAP MVS pipeline. This failed because monocular depth estimators produce "
    "scale-ambiguous, camera-space depth values. Depths from different images are in "
    "incompatible coordinate systems and cannot be directly merged. COLMAP MVS uses the "
    "calibrated camera geometry from SfM to ensure depth maps are in a consistent metric "
    "world space, making fusion possible."
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\VR_Pipeline_Lab_Report.docx"
doc.save(out_path)
print(f"Saved: {out_path}")
