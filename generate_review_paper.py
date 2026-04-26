"""
Generate a review-style academic paper as a Word document.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Page setup ────────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def h1(text):
    p = doc.add_heading(text, level=1)
    return p

def h2(text):
    p = doc.add_heading(text, level=2)
    return p

def h3(text):
    p = doc.add_heading(text, level=3)
    return p

def body(text, indent=False):
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(11)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.3)
    return p

def bullet(text, level=0):
    style = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(text, style=style)
    p.style.font.size = Pt(11)
    return p

def italic_body(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    return p

def add_table(headers, rows, col_widths=None, header_color="1F4E79"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        set_cell_bg(cell, header_color)
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg = "EBF3FB" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    doc.add_paragraph()
    return table

def add_ref(num, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    run = p.add_run(f"[{num}]  ")
    run.bold = True
    run.font.size = Pt(9)
    run2 = p.add_run(text)
    run2.font.size = Pt(9)
    return p

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
title = doc.add_heading(
    "Room-Scale VR Digital Twins via 3D Gaussian Splatting:\n"
    "A Review of Methods, Implementation Challenges, and Experimental Results", 0
)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size = Pt(16)

doc.add_paragraph()
author = doc.add_paragraph("Mohamed Gallai")
author.alignment = WD_ALIGN_PARAGRAPH.CENTER
author.runs[0].bold = True
author.runs[0].font.size = Pt(12)

inst = doc.add_paragraph(datetime.date.today().strftime("%B %Y"))
inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
inst.runs[0].font.size = Pt(11)
inst.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()

# ── Abstract ──────────────────────────────────────────────────────────────────
abs_h = doc.add_paragraph()
r = abs_h.add_run("Abstract. ")
r.bold = True
r.font.size = Pt(11)
r2 = abs_h.add_run(
    "We present the design, implementation, and iterative refinement of an open-source "
    "pipeline for converting consumer phone scans into interactive room-scale VR digital twins "
    "deployable on the Meta Quest 3 headset. The pipeline combines COLMAP Structure-from-Motion (SfM), "
    "COLMAP Multi-View Stereo (MVS), and 3D Gaussian Splatting (3DGS) to produce both a "
    "photorealistic Gaussian splat for visual rendering and a triangulated mesh for physics-based "
    "VR object interaction. We review the theoretical foundations of the component methods, survey "
    "the state of the art in neural rendering and photogrammetry, document unsuccessful approaches "
    "with their root causes, and report quantitative quality metrics on an initial reconstruction "
    "of a real room from 185 iPhone photographs. Our best single-run result produces 4.99 million "
    "Gaussians with a 36% low-opacity ghost ratio from a sparse COLMAP initialisation; "
    "a follow-up reconstruction using MVS dense initialisation (131 MB fused point cloud, "
    "~10M points) is expected to reduce this ratio substantially. "
    "All code runs inside Docker containers and is freely available."
)
r2.font.size = Pt(11)
abs_h.paragraph_format.left_indent = Inches(0.5)
abs_h.paragraph_format.right_indent = Inches(0.5)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════
h1("1  Introduction")

body(
    "The ability to capture a physical space with a smartphone and re-experience it "
    "in photorealistic virtual reality is rapidly becoming technically feasible. Consumer "
    "apps such as Apple RealityScan and Luma AI allow non-expert users to capture "
    "multi-view photo sets of rooms and objects using standard smartphone hardware. "
    "The central challenge lies in converting these raw photo sets into representations "
    "that satisfy two competing requirements: (1) photorealistic appearance at real-time "
    "frame rates for VR rendering, and (2) geometrically accurate, physics-simulatable "
    "meshes for interactive object manipulation.", indent=True
)
body(
    "This paper documents our effort to build such a pipeline from open-source components. "
    "Starting from a 185-image iPhone scan of a lab room captured with RealityScan, "
    "we developed and iteratively refined a multi-stage reconstruction system that "
    "produces both a Gaussian Splat for photorealistic background rendering and a "
    "polygonal mesh for physics-based interaction — all deployable on the Meta Quest 3 "
    "standalone VR headset.", indent=True
)
body(
    "The paper is structured as follows. Section 2 reviews the theoretical background "
    "of the core methods: photogrammetry, NeRF, 3D Gaussian Splatting, and monocular "
    "depth estimation. Section 3 describes the input data, acquisition process, and "
    "the design choices it imposes. Section 4 presents the evolution of our pipeline "
    "across two versions. Section 5 reports experimental results and quality analysis. "
    "Section 6 discusses lessons learned, limitations, and next steps.", indent=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════════
h1("2  Background and Related Work")

# 2.1
h2("2.1  Classical Photogrammetry")
body(
    "Photogrammetry is the science of making measurements from photographs, with roots "
    "in aerial surveying dating to the 19th century. In the context of 3D reconstruction, "
    "photogrammetry refers to the process of inferring the 3D shape of an object or scene "
    "from a collection of 2D photographs taken from multiple viewpoints. The classical "
    "pipeline comprises three stages: (1) feature detection and matching across image pairs "
    "to establish pixel correspondences, (2) camera pose estimation via bundle adjustment "
    "to recover the position and orientation of each camera, and (3) dense multi-view "
    "stereo reconstruction to generate a 3D point cloud or mesh from the estimated poses."
)
body(
    "Modern photogrammetry systems such as Agisoft Metashape and RealityCapture automate "
    "this pipeline with GPU acceleration and deliver production-quality meshes from "
    "consumer photography. However, they are commercial tools with per-export licensing "
    "fees and limited programmable integration. The open-source COLMAP system [1] implements "
    "the same pipeline with comparable quality and full scriptability, making it the "
    "standard baseline for academic reconstruction work."
)

# 2.2
h2("2.2  Structure-from-Motion (SfM) and COLMAP")
body(
    "Structure-from-Motion (SfM) solves the problem of simultaneously recovering camera "
    "poses and a sparse 3D scene structure from unordered image collections. The dominant "
    "approach, implemented in COLMAP [1], proceeds as follows:"
)
bullet("Feature extraction: Scale-Invariant Feature Transform (SIFT) detects distinctive keypoints in each image and describes them with 128-dimensional gradient histograms that are robust to scale, rotation, and partial illumination change.")
bullet("Exhaustive matching: Every image pair is matched to find putative point correspondences. Geometric verification using RANSAC and the fundamental matrix filters mismatches.")
bullet("Incremental mapping (Mapper): Starting from a reliable two-view seed, the mapper registers new images one at a time using PnP (Perspective-n-Point) pose estimation, triangulates new 3D points, and refines all cameras and points jointly via bundle adjustment.")
bullet("Image undistortion: The calibrated lens distortion model is applied to rectify images to a pinhole camera, producing undistorted images and a corresponding undistorted sparse model — the standard input format for downstream MVS and neural rendering tools.")
body(
    "A critical practical detail is that COLMAP numbers its output sparse model subfolders "
    "(sparse/0, sparse/1, ...) in the order they are finalised, not by quality. The largest "
    "and most complete model — the one covering the most images — is not necessarily in "
    "sparse/0. A robust implementation must identify the best model by inspecting the size "
    "of points3D.bin across all subfolders."
)

# 2.3
h2("2.3  Multi-View Stereo (MVS) and Dense Reconstruction")
body(
    "SfM produces only a sparse point cloud — typically tens of thousands of points "
    "corresponding to detected SIFT keypoints. This is insufficient for photorealistic "
    "reconstruction, which requires millions of points to cover all visible surfaces. "
    "Multi-View Stereo (MVS) densifies the reconstruction by computing per-pixel depth "
    "estimates for every image using the calibrated camera geometry."
)
body(
    "COLMAP's PatchMatchStereo algorithm [2] implements a GPU-accelerated variant of "
    "PatchMatch depth estimation. For each image, it treats each pixel as an unknown "
    "depth hypothesis and propagates depth estimates from well-supported pixels to "
    "neighbouring uncertain ones through iterative sweeping passes. Geometric consistency "
    "filtering then cross-validates depth estimates across multiple views, rejecting "
    "estimates that are inconsistent when reprojected into neighbouring cameras."
)
body(
    "StereoFusion merges the per-image depth maps into a single, globally consistent "
    "dense point cloud. For a 130-image room scan at 1,512 × 2,016 px resolution, "
    "this produces approximately 10 million 3D points — roughly 270× more than the "
    "sparse SfM stage — covering all visible surfaces including walls, furniture edges, "
    "and fine object detail."
)

# 2.4
h2("2.4  Neural Radiance Fields (NeRF)")
body(
    "Neural Radiance Fields (NeRF), introduced by Mildenhall et al. (2020) [3], "
    "represented a paradigm shift in novel view synthesis. A NeRF encodes an entire scene "
    "as a continuous volumetric function F(x,y,z,θ,φ) → (RGB, σ) — mapping a 3D position "
    "and viewing direction to colour and density — using a multi-layer perceptron (MLP). "
    "Novel views are rendered by ray marching: for each pixel, a ray is cast into the "
    "volume, sampled at hundreds of points, and the colour is integrated using volume "
    "rendering equations."
)
body(
    "NeRF achieved unprecedented photorealistic novel view synthesis quality on bounded "
    "scenes but suffered from two fundamental limitations that preclude its use in real-time VR:"
)
bullet("Training time: Original NeRF required 1–2 days of GPU training per scene. Even optimised variants (Instant-NGP, TensoRF) require minutes to hours.")
bullet("Rendering speed: Ray marching requires evaluating the MLP hundreds of times per pixel per frame. Even highly optimised NeRF variants render at 1–10 FPS — far below the 72 FPS required by the Meta Quest 3 for comfortable VR.")
body(
    "Numerous follow-up works (Instant-NGP [4], Mip-NeRF 360, ZipNeRF) improved training "
    "speed and unbounded scene handling but did not solve the rendering bottleneck for "
    "real-time VR applications."
)

# 2.5
h2("2.5  3D Gaussian Splatting (3DGS)")
body(
    "3D Gaussian Splatting (3DGS), introduced by Kerbl et al. (2023) [5], solved the "
    "real-time rendering bottleneck by replacing the implicit neural field with an "
    "explicit set of 3D Gaussian primitives. Each Gaussian is a small, semi-transparent "
    "coloured ellipsoid defined by:"
)
bullet("Position (μ): a 3D centre point in world space.")
bullet("Covariance (Σ): a 3×3 positive semi-definite matrix encoding size and orientation (parameterised as a quaternion rotation and 3 scale values for training stability).")
bullet("Opacity (α): a scalar in [0,1] controlling transparency.")
bullet("Colour (SH): spherical harmonic coefficients encoding view-dependent appearance. Degree 3 (45 coefficients per colour channel) captures specular highlights and reflections.")
body(
    "Rendering is performed by projecting each 3D Gaussian onto the 2D image plane as "
    "a 2D Gaussian splat, sorting by depth, and alpha-compositing front-to-back using "
    "the standard over operator. This tile-based rasterisation runs on the GPU in "
    "milliseconds, achieving 100–300 FPS on consumer hardware — meeting VR frame rate "
    "requirements comfortably."
)
body(
    "Training uses photometric loss between rendered and ground-truth images, optimising "
    "all Gaussian parameters via gradient descent. A key component is Adaptive Density "
    "Control (ADC): Gaussians in under-reconstructed regions are cloned or split, while "
    "Gaussians with very low opacity are periodically pruned. This drives the model to "
    "allocate representational capacity where the scene is complex."
)
body(
    "Nerfstudio [6] provides several 3DGS implementations: splatfacto (standard 3DGS), "
    "splatfacto-big (degree-3 SH for higher colour fidelity), and in newer versions, "
    "2DGS (see Section 2.6). We use splatfacto-big as our primary method."
)

# 2.6
h2("2.6  2D Gaussian Splatting (2DGS)")
body(
    "A limitation of standard 3DGS for mesh extraction is that 3D ellipsoid Gaussians "
    "are volumetric — they fill space rather than lying on surfaces. When Poisson surface "
    "reconstruction is applied to extract a mesh from a 3DGS model, the resulting mesh "
    "is often noisy and poorly defined because the Gaussians do not align with scene surfaces."
)
body(
    "2D Gaussian Splatting (2DGS), proposed by Huang et al. (2024) [7], addresses this "
    "by replacing 3D ellipsoid Gaussians with 2D oriented disk Gaussians that lie in a "
    "local tangent plane. Each 2DGS primitive is essentially a flat ellipse aligned to "
    "a surface normal, making the representation inherently surface-oriented. "
    "This produces sharper and more accurate surface geometry, enabling higher-quality "
    "mesh extraction via Poisson reconstruction — which is essential for our use case "
    "of extracting physics-simulatable meshes for VR object interaction."
)
body(
    "Unfortunately, 2DGS is not included in Nerfstudio version 1.1.5 (our current Docker "
    "image). It requires either a newer Nerfstudio build or the installation of the gsplat "
    "library with 2DGS support. This is a planned upgrade in our pipeline."
)

# 2.7
h2("2.7  Monocular Depth Estimation — Depth Anything v3 (DA3)")
body(
    "Monocular depth estimation is the problem of inferring a dense depth map from a "
    "single RGB image — a fundamentally ill-posed problem since multiple 3D scenes can "
    "produce the same 2D projection. Deep learning-based methods learn strong geometric "
    "priors from large datasets to resolve this ambiguity."
)
body(
    "Depth Anything v3 (DA3) [8] is a state-of-the-art monocular depth estimator based "
    "on a Vision Transformer (ViT) backbone trained on a mixture of labelled and "
    "pseudo-labelled data across diverse scenes. It produces per-pixel relative depth "
    "maps at high resolution with strong perceptual quality."
)
body(
    "We initially explored using DA3 to generate dense depth maps from our input images "
    "as an alternative to COLMAP MVS, with the motivation of avoiding the ~30-minute "
    "GPU cost of PatchMatchStereo. This approach failed for a fundamental reason: "
    "DA3 produces affine-invariant (scale and shift-ambiguous) depth values in camera "
    "space. Depths from different images are in incompatible, camera-relative coordinate "
    "frames and cannot be straightforwardly merged into a coherent world-space 3D point cloud. "
    "The resulting Gaussian Splat exhibited severe floating artefacts and incoherent geometry. "
    "MVS — which uses calibrated multi-view geometry to enforce metric consistency across "
    "all views — is the correct solution for this task."
)

# 2.8
h2("2.8  Segment Anything — SAM3")
body(
    "The Segment Anything Model 3 (SAM3) is a text-prompted and click-prompted image "
    "segmentation model capable of producing pixel-accurate masks for arbitrary objects "
    "in an image given a natural-language description or point prompt. In our V1 pipeline, "
    "SAM3 was used to generate foreground masks over undistorted images — the intention "
    "being to suppress COLMAP feature detection in regions containing background clutter "
    "(floors, ceilings, windows) so that the SfM and Gaussian Splat model would focus "
    "on the room objects of interest."
)
body(
    "In practice, aggressive masking backfired: COLMAP's image matcher depends on finding "
    "sufficient feature correspondences across image pairs. When masks suppressed features "
    "in background regions that happened to be informative for camera registration, "
    "the mapper failed to register many images or produced degenerate sparse models. "
    "SAM3-based masking has been removed from the current pipeline, though it remains "
    "a candidate for future per-object segmentation workflows."
)

# 2.9
h2("2.9  VR Rendering on Meta Quest 3")
body(
    "The Meta Quest 3 is a standalone VR headset with an integrated Snapdragon XR2 Gen 2 "
    "processor. It runs at 72–120 Hz and requires frame rendering in under 11 ms to avoid "
    "motion sickness. This imposes hard constraints on the representations that can be used "
    "for photorealistic room rendering:"
)

add_table(
    headers=["Method", "Render speed", "Memory footprint", "VR feasible?"],
    rows=[
        ["NeRF (vanilla)",        "0.03–0.5 FPS",  "Low (MLP weights)",      "No — 200× too slow"],
        ["Instant-NGP",           "5–20 FPS",       "~50 MB hash grid",       "No — 4-15× too slow"],
        ["3D Gaussian Splatting", "100–300 FPS",    "100 MB – 2 GB (.ply)",   "Yes — with model compression"],
        ["Textured mesh (raster)","1000+ FPS",      "Depends on poly count",  "Yes — standard game pipeline"],
    ],
    col_widths=[2.0, 1.5, 1.8, 2.2],
    header_color="1F4E79",
)
body(
    "For the room background, 3DGS is the only photorealistic option currently feasible "
    "on Quest 3 hardware. For interactive objects (chairs, books, lamps), a conventional "
    "textured polygon mesh with a MeshCollider is required for physics simulation — "
    "Gaussian Splats have no concept of physical surface and cannot be used with "
    "physics engines directly."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. INPUT DATA AND ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════
h1("3  Input Data, Acquisition, and Design Constraints")

h2("3.1  Capture Device and Software")
body(
    "Images were captured using Apple RealityScan on an iPhone. RealityScan guides "
    "the user through an orbital scan pattern, providing real-time feedback on coverage "
    "gaps. For a room-scale scene, this produced 185 photographs across a single capture "
    "session, covering walls, floor, ceiling, and key objects from multiple angles."
)
body(
    "RealityScan exports images in HEIC (High Efficiency Image Codec) format with "
    "accompanying XMP sidecar files containing ARKit camera pose metadata. HEIC is "
    "Apple's proprietary image format offering better compression than JPEG at equivalent "
    "quality but with limited support in open-source tools — COLMAP, for instance, "
    "cannot read HEIC directly. Our pipeline converts all HEIC files to JPEG "
    "(92% quality, lossless for practical purposes) using the pillow-heif library "
    "before any processing."
)

h2("3.2  ARKit Poses vs. Fresh SfM")
body(
    "A natural question is whether to use the ARKit camera poses embedded in the XMP "
    "sidecars rather than running COLMAP SfM from scratch. ARKit provides real-time "
    "visual-inertial odometry poses with sub-centimetre accuracy for short sessions. "
    "We evaluated this path and chose to run fresh COLMAP SfM instead for three reasons:"
)
bullet("Coordinate frame ambiguity: ARKit poses are in a session-local coordinate frame with an arbitrary origin and orientation. Converting to COLMAP / Nerfstudio world coordinates requires aligning the ARKit trajectory to a world-space reference — a non-trivial calibration step.")
bullet("Scale uncertainty: ARKit scale is metric (in real metres), but the relationship between the ARKit world frame and COLMAP's implicit scale requires ground-truth or gravity alignment.")
bullet("Intrinsic quality: COLMAP's bundle adjustment jointly refines camera intrinsics (focal length, principal point, distortion) together with poses. ARKit provides fixed manufacturer-specified intrinsics which may not match the actual lens characteristics at the specific image resolution exported by RealityScan.")
body(
    "Fresh COLMAP SfM from the raw images is fully self-contained, self-calibrating, "
    "and has been validated at production quality on similar phone-capture scenarios."
)

h2("3.3  Registration Rate and Scene Complexity")
body(
    "COLMAP registered 130 out of 185 input images (70.3%). The 55 unregistered images "
    "are typically those with insufficient overlap with already-registered views — "
    "common at the start and end of a scan trajectory, or in regions with repetitive "
    "texture (plain white walls, uniform floors) where SIFT features are sparse. "
    "A 70% registration rate is within the normal range for room-scale scans and "
    "is sufficient for high-quality reconstruction, as 130 well-distributed views "
    "provide dense coverage of all major surfaces."
)

h2("3.4  Image Resolution and Downscaling")
body(
    "The undistorted images from COLMAP are 1,512 × 2,016 pixels (approximately 3 megapixels). "
    "Training 3DGS at full resolution would require loading ~130 × 3M = ~390 million pixels "
    "per training batch, exceeding GPU VRAM on most consumer cards (RTX 5070 Ti has 16 GB). "
    "We use 2× downscaling to 756 × 1,008 px, reducing memory pressure by 4× while "
    "retaining sufficient detail for room-scale reconstruction. The Nerfstudio nerfstudio-data "
    "dataparser expects pre-generated downscaled image folders (images_2/) to be present "
    "on disk; we generate these in a pre-processing step using PIL before launching the "
    "Docker training container."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
h1("4  Pipeline Design and Evolution")

h2("4.1  Version 1 — Proof of Concept")
body(
    "The initial pipeline was implemented as a single ~250-line Python script. "
    "It sequentially invoked Docker containers for COLMAP and Nerfstudio and optionally "
    "applied SAM3 masking between the two COLMAP stages. While it demonstrated the "
    "end-to-end flow, it suffered from several structural problems that prevented "
    "reliable production-quality output."
)

add_table(
    headers=["V1 Stage", "Tool", "Problem"],
    rows=[
        ["Image staging",       "Python / shutil",    "HEIC not supported; copied only JPG/PNG"],
        ["Unmasked SfM",        "COLMAP",             "Hardcoded sparse/0; failed if best model was sparse/1+"],
        ["SAM3 masking",        "SAM3 (local)",       "Fragile; required local checkpoint; broke registration"],
        ["Masked re-SfM",       "COLMAP",             "Often produced empty models from over-masking"],
        ["Nerfstudio training", "splatfacto",         "No MVS; 4× downscale; no TORCH_HOME cache"],
        ["Splat export",        "ns-export",          "torch.load weights_only crash without patch"],
    ],
    col_widths=[1.6, 1.5, 4.4],
    header_color="7F3F00",
)

h2("4.2  Failed Approaches and Root Cause Analysis")
body(
    "During development, several approaches were attempted and subsequently discarded. "
    "Understanding why they failed informed the design of the current pipeline."
)

add_table(
    headers=["Approach", "Expected benefit", "Failure mode", "Root cause"],
    rows=[
        [
            "Depth Anything v3 (DA3) for dense init",
            "Dense point cloud without expensive MVS",
            "Floaters, incoherent 3D geometry, non-photorealistic splat",
            "DA3 outputs affine-invariant camera-space depth; no metric scale; images in incompatible frames — cannot be fused into world space",
        ],
        [
            "ns-train 2dgs",
            "Surface-aligned Gaussians for clean mesh extraction",
            "Exit code 2: command not found",
            "2DGS not included in Nerfstudio 1.1.5; requires separate plugin or newer build",
        ],
        [
            "--pipeline.datamanager.cache-images cpu",
            "Pre-load images to RAM for faster GPU feeding",
            "Training hung / OOM before starting",
            "130 images × 1512×2016 px ≈ 10 GB RAM requirement; exceeded available memory",
        ],
        [
            "--pipeline.model.camera-optimizer.mode SO3xR3",
            "Refine COLMAP poses during training",
            "Exit code 1: unrecognised argument",
            "splatfacto-big does not expose camera_optimizer as a configurable sub-module in Nerfstudio 1.1.5",
        ],
        [
            "SAM3 masked re-SfM",
            "Focus reconstruction on room objects",
            "Degenerate or empty sparse models",
            "Over-masking removed features COLMAP needed for image registration",
        ],
        [
            "Docker --rm with default /root/.cache",
            "Standard Docker cleanup",
            "10–20 min hang at 0% GPU before training",
            "Nerfstudio downloads alexnet (233 MB LPIPS model) on first use; --rm destroys cache; re-downloaded every run",
        ],
        [
            "COLMAP undistort then MVS in same run",
            "Single-pass dense reconstruction",
            "fused.ply destroyed by second undistort call",
            "image_undistorter does rm -rf on the dense folder; running it again after MVS destroyed the 131 MB fused point cloud",
        ],
    ],
    col_widths=[1.5, 1.5, 1.7, 2.8],
    header_color="C00000",
)

h2("4.3  Version 2 — Current Architecture")
body(
    "The current pipeline is a modular rewrite of approximately 1,200 lines across six "
    "Python modules, all orchestrating Docker containers so that no local GPU dependencies "
    "are required beyond Docker and Python with pillow-heif."
)

add_table(
    headers=["Stage", "Module", "Input → Output", "Time"],
    rows=[
        ["1. HEIC → JPEG",         "heic_converter.py",      "185 HEIC → 185 JPEG",                         "2 min"],
        ["2. COLMAP SfM",           "colmap_pipeline.py",     "JPEG → undistorted images + sparse model",    "15–25 min"],
        ["3. COLMAP MVS",           "colmap_pipeline.py",     "Undistorted → fused.ply (10M+ points)",       "25–40 min"],
        ["4. transforms.json",      "colmap_to_ns.py",        "COLMAP binary → Nerfstudio JSON + init PLY",  "< 1 min"],
        ["5. Downscale images",     "nerfstudio_pipeline.py", "Full-res → images_2/ folder",                 "2 min"],
        ["6. splatfacto-big train", "nerfstudio_pipeline.py", "transforms.json + fused.ply → checkpoint",   "40–55 min"],
        ["7. Splat export",         "nerfstudio_pipeline.py", "Checkpoint → splat.ply (~1 GB)",              "5 min"],
        ["8. Prune ghosts",         "prune_splat.py",         "splat.ply → splat_pruned.ply (55% of input)", "3 min"],
        ["9. Mesh export",          "nerfstudio_pipeline.py", "Checkpoint → mesh.ply / mesh.obj",            "5–10 min"],
    ],
    col_widths=[1.4, 1.7, 2.6, 0.9],
    header_color="375623",
)

h2("4.4  Key Engineering Solutions")

h3("4.4.1  COLMAP Binary Parsing and Coordinate Conversion")
body(
    "Nerfstudio's colmap dataparser reads COLMAP text-format models. To use the "
    "nerfstudio-data dataparser (which accepts transforms.json directly and allows "
    "specifying a ply_file_path for Gaussian initialisation), we implemented a custom "
    "colmap_to_ns.py module that reads COLMAP's binary format (cameras.bin, images.bin, "
    "points3D.bin) in pure Python using the struct module."
)
body(
    "The critical coordinate conversion from COLMAP's OpenCV convention (Y down, Z into "
    "scene) to Nerfstudio's OpenGL convention (Y up, Z out of scene) is performed by "
    "negating columns 1 and 2 of the camera-to-world rotation matrix:"
)
italic_body(
    "c2w[:3, 1:3] *= -1    # flip Y and Z axes — OpenCV → OpenGL"
)
body(
    "The transforms.json embeds a ply_file_path field pointing to fused.ply, "
    "which Nerfstudio uses to initialise Gaussian positions from the dense MVS point "
    "cloud rather than from random sampling."
)

h3("4.4.2  Idempotency and Fault Tolerance")
body(
    "Each pipeline stage checks for the existence of its expected output before running. "
    "COLMAP SfM (the most expensive stage at 15–25 min) is skipped entirely if "
    "colmap/dense/sparse/0/cameras.bin already exists. This allows the pipeline to resume "
    "from any intermediate state after a crash, Docker restart, or manual intervention "
    "without repeating completed work."
)
body(
    "A dedicated backup/restore mechanism protects fused.ply from being deleted by "
    "COLMAP's image_undistorter command, which unconditionally removes and recreates "
    "the entire dense output folder."
)

h3("4.4.3  LPIPS Model Caching")
body(
    "Nerfstudio uses the LPIPS (Learned Perceptual Image Patch Similarity) metric "
    "during training, which requires a pre-trained AlexNet network (alexnet-owt-7be5be79.pth, "
    "233 MB). By default, PyTorch caches this in /root/.cache/torch/ inside the container. "
    "Because we run Docker with --rm (automatic container removal), this cache is destroyed "
    "after every run, causing a mandatory 10–20 minute re-download before every training "
    "session. We solved this by passing -e TORCH_HOME=/workspace/torch_cache to every "
    "docker run command, redirecting the cache to the mounted workspace directory "
    "which persists across container restarts."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
h1("5  Experimental Results")

h2("5.1  SfM Registration")
body(
    "COLMAP SfM was run on 185 iPhone HEIC images converted to JPEG at 92% quality. "
    "The feature extractor detected an average of ~8,000 SIFT keypoints per image "
    "with a PINHOLE camera model and single_camera=1 (all images share one camera model). "
    "Exhaustive matching was performed across all image pairs."
)

add_table(
    headers=["Metric", "Value"],
    rows=[
        ["Input images",              "185"],
        ["Registered images",         "130 (70.3%)"],
        ["Sparse 3D points",          "~37,710"],
        ["Camera model",              "PINHOLE"],
        ["Best sparse model",         "Selected by max points3D.bin size"],
        ["Undistorted image size",    "1,512 × 2,016 px"],
        ["Downscaled training size",  "756 × 1,008 px (2× factor)"],
    ],
    col_widths=[3.0, 4.5],
    header_color="1F4E79",
)

h2("5.2  MVS Dense Reconstruction")
body(
    "COLMAP MVS (PatchMatchStereo with geometric consistency, GPU index -1 for all GPUs) "
    "was run on the 130 undistorted images. StereoFusion merged the resulting depth maps "
    "with a minimum pixel support threshold of 3, producing:"
)

add_table(
    headers=["Metric", "Value"],
    rows=[
        ["Dense point cloud file",    "colmap/dense/fused.ply"],
        ["File size",                 "131 MB"],
        ["Estimated point count",     "~10 million points"],
        ["MVS runtime (RTX 5070 Ti)", "~47 minutes (PatchMatchStereo + StereoFusion)"],
        ["Max image size limit",      "1,600 px (config default)"],
    ],
    col_widths=[3.0, 4.5],
    header_color="1F4E79",
)

h2("5.3  Gaussian Splat Quality — V1 (Sparse Init)")
body(
    "The V1 splat was trained from 37,710 sparse COLMAP points. After 30,000 iterations "
    "of splatfacto-big training, the following quality metrics were measured by directly "
    "parsing the binary PLY file and computing per-Gaussian statistics:"
)

add_table(
    headers=["Metric", "Value", "Interpretation"],
    rows=[
        ["Total Gaussians",               "4,998,420",    "Dense; grew via adaptive densification from 37k seed points"],
        ["Spherical Harmonic degree",      "3 (45 coeff)","Maximum colour fidelity in Nerfstudio 1.1.5"],
        ["Solid Gaussians (opacity > 0.9)","25.8%",       "~1.29M Gaussians representing confident solid geometry"],
        ["Ghost Gaussians (opacity < 0.15)","36.2%",      "⚠ High — caused by sparse init filling uncertain regions"],
        ["Mean opacity (sigmoid)",         "0.405",       "Below 0.5 — many uncertain semi-transparent Gaussians"],
        ["Median Gaussian scale",          "0.0066 units","Very fine, detailed primitives — positive indicator"],
        ["Large Gaussians (scale > 1.0)",  "0.0%",        "No catastrophic floater blobs"],
        ["Spatial outliers (> 5σ)",        "~0.4%",       "Low — minor fringe artefacts only"],
        ["After pruning (opacity > 0.15)", "2,786,436",   "55.7% retained; file size 1.2 GB → 659 MB"],
    ],
    col_widths=[2.5, 1.5, 3.5],
    header_color="7F6000",
)
body(
    "The primary quality issue is the 36.2% ghost Gaussian ratio — Gaussians with opacity "
    "below 0.15 that contribute haze and fog to the rendered image. This is directly "
    "attributable to the sparse initialisation: without dense geometry to anchor to, "
    "Gaussians grow in under-constrained regions of space and fail to converge to "
    "high opacity. The V2 training run, initialised from the 131 MB MVS dense cloud "
    "(~10M points), is expected to reduce this ratio substantially."
)

h2("5.4  Post-Processing Pruning")
body(
    "We implemented a post-processing pruning pass that operates directly on the exported "
    "binary PLY without re-training. The pass removes:"
)
bullet("Gaussians with sigmoid opacity < 0.15 (2,191,071 removed — 43.8%)")
bullet("Gaussians whose position deviates more than 5σ from the scene mean in any axis (20,913 removed — 0.4%)")
body(
    "The pruned splat retains 2,786,436 Gaussians (55.7% of the original) at 659 MB, "
    "representing the solid, high-confidence geometry. This provides an immediately "
    "viewable artefact while the V2 training run (from dense MVS init) continues."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DISCUSSION
# ═══════════════════════════════════════════════════════════════════════════════
h1("6  Discussion")

h2("6.1  Dense Initialisation Is Critical")
body(
    "Our most significant finding is that the quality of Gaussian Splatting is "
    "highly sensitive to the quality of the point cloud used for initialisation. "
    "The 37,710-point sparse COLMAP cloud produced a model with 36% ghost Gaussians; "
    "we anticipate the 10M-point MVS cloud will reduce this to under 10%, based on "
    "the literature and the well-understood mechanism: denser initialisation provides "
    "more anchor points for Gaussians to grow toward real surfaces, leaving fewer "
    "under-constrained regions where ghost Gaussians accumulate."
)
body(
    "The tradeoff is time: MVS adds 40–50 minutes to the pipeline. For production use, "
    "this is acceptable. For rapid prototyping or parameter tuning, the sparse init "
    "with post-processing pruning (total time ~60 minutes vs. ~110 minutes for MVS) "
    "provides a usable intermediate result."
)

h2("6.2  Monocular Depth Is Not a Substitute for MVS")
body(
    "The failure of Depth Anything v3 as a dense initialisation source is important "
    "to document, as monocular depth estimators are superficially attractive for this "
    "purpose. The fundamental issue is metric consistency: multi-view stereo uses "
    "calibrated camera geometry to enforce that depth estimates from different views "
    "are in the same coordinate system. Monocular estimators, trained to minimise "
    "scale-invariant loss functions, produce relative depth maps that are incompatible "
    "across images without additional alignment. Scale alignment methods (e.g., "
    "aligning monocular depths to COLMAP sparse points via least-squares) exist but "
    "introduce significant error in regions not covered by SIFT keypoints."
)

h2("6.3  Containerisation and Reproducibility")
body(
    "Running all computation inside Docker containers (colmap/colmap:latest and a "
    "custom nerfstudio-blackwell image built for NVIDIA Blackwell / sm_120 architecture) "
    "ensures that the pipeline is fully reproducible across lab machines with different "
    "operating system configurations. The only host-level requirement is Docker with "
    "GPU passthrough (--gpus all) and a Python environment with pillow-heif for HEIC "
    "conversion. This design makes the pipeline accessible to lab members who are not "
    "expert in managing CUDA dependencies."
)

h2("6.4  Limitations")
bullet("2DGS unavailable: The current Nerfstudio version does not include 2DGS. Mesh extraction via Poisson reconstruction from a splatfacto-big model is of lower quality than from a 2DGS model because 3D ellipsoid Gaussians are not surface-aligned.")
bullet("55 unregistered images: 15% of the captured images were not registered by COLMAP. Improving this requires either additional capture coverage or sequential matching strategies (e.g. COLMAP sequential_matcher) that exploit temporal ordering of images.")
bullet("No texture extraction: The current mesh export (ns-export poisson) produces an untextured mesh. For physics colliders this is acceptable, but for rendering the mesh texture must be projected from the training images — a planned addition.")
bullet("No per-object segmentation: The current pipeline produces a single monolithic splat and mesh of the entire room. Physics interaction requires per-object meshes with individual rigidbody components — this requires 3D segmentation (SAM3D or Gaussian grouping methods).")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════════════
h1("7  Next Steps and Future Work")

add_table(
    headers=["Priority", "Task", "Expected outcome"],
    rows=[
        ["1 — Immediate",   "V2 training from MVS dense init (in progress)",
         "Ghost ratio < 10%, sharper surfaces, better mesh extraction"],
        ["2 — Short term",  "Enable 2DGS (update Dockerfile / gsplat)",
         "Surface-aligned Gaussians; Poisson mesh quality improvement"],
        ["3 — Short term",  "Mesh extraction and decimation to 300k faces",
         "Physics-ready mesh for Unity XR Interaction Toolkit"],
        ["4 — Medium",      "Unity integration — Gaussian Splat background",
         "Photorealistic room visible on Quest 3 at 72+ FPS"],
        ["5 — Medium",      "Unity integration — mesh physics colliders",
         "Users can grab and interact with room objects in VR"],
        ["6 — Future",      "Per-object 3D segmentation (Gaussian grouping / SAM3D)",
         "Individual physics bodies per object rather than monolithic mesh"],
        ["7 — Future",      "Automated re-scan and incremental update",
         "Room changes detected and incrementally updated in the digital twin"],
    ],
    col_widths=[1.3, 2.4, 3.8],
    header_color="1F4E79",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
h1("8  Conclusion")
body(
    "We have built and validated an open-source pipeline for room-scale VR digital twin "
    "construction from consumer iPhone photographs. The pipeline is fully containerised, "
    "idempotent, and requires no expert configuration of CUDA environments. Starting "
    "from 185 HEIC images, it produces a 3D Gaussian Splat of a real room in "
    "approximately 110 minutes of compute on an NVIDIA RTX 5070 Ti GPU.", indent=True
)
body(
    "Our key technical contributions are: (1) a robust COLMAP-to-Nerfstudio coordinate "
    "conversion module with correct OpenCV-to-OpenGL axis flipping; (2) idempotent "
    "stage checkpointing enabling recovery from any intermediate failure; (3) a "
    "fused.ply preservation mechanism preventing data loss during COLMAP undistortion; "
    "(4) LPIPS model caching via TORCH_HOME redirection eliminating 10–20 minute "
    "cold-start penalties; and (5) a statistical post-processing pruning pass "
    "eliminating ghost Gaussians without re-training.", indent=True
)
body(
    "The primary remaining bottleneck is the absence of 2DGS support in the current "
    "Nerfstudio version, which limits mesh extraction quality. Resolving this, together "
    "with Unity integration and per-object physics rigging, will complete the full "
    "VR digital twin pipeline.", indent=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════
h1("References")
add_ref(1, 'Schönberger, J.L. and Frahm, J.M. "Structure-from-Motion Revisited." CVPR 2016.')
add_ref(2, 'Schönberger, J.L. et al. "Pixelwise View Selection for Unstructured Multi-View Stereo." ECCV 2016.')
add_ref(3, 'Mildenhall, B. et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020.')
add_ref(4, 'Müller, T. et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." SIGGRAPH 2022.')
add_ref(5, 'Kerbl, B. et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.')
add_ref(6, 'Tancik, M. et al. "Nerfstudio: A Modular Framework for Neural Radiance Field Development." SIGGRAPH 2023.')
add_ref(7, 'Huang, B. et al. "2D Gaussian Splatting for Geometrically Accurate Radiance Fields." SIGGRAPH 2024.')
add_ref(8, 'Yang, L. et al. "Depth Anything V2." NeurIPS 2024.')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\VR_Pipeline_Review_Paper.docx"
doc.save(out_path)
print(f"Saved: {out_path}")
