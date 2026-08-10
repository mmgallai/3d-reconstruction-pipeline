"""Build TO_EDIT_SHOOTOUT.docx — 40 rows (8 scenes × 5 models) with
original + edited images side-by-side and per-row notes on
mask/prompt inputs actually used.

Layout:
  Cover page
  Per-scene section header (with original + union-mask preview thumbnails)
  Big table with 5 rows per scene = 40 rows total:
    | Model | Original | Edited | Notes (mask/prompt/wallclock) |
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
IMG_DIR = PROJECT / "output/to_edit/images"
MASK_DIR = PROJECT / "output/to_edit/masks"
OUT_DIR = PROJECT / "output/to_edit/outputs"
PREV_DIR = PROJECT / "output/to_edit/previews"
DOCX_OUT = PROJECT / "TO_EDIT_SHOOTOUT.docx"

# Scene metadata (from instrutions.txt + our SAM3 prompts)
SCENES = [
    {
        "stem": "data2_a",
        "label": "Desk scene 2A (bottle + blue box + lobster)",
        "objects_removed": "white water bottle, blue Arducam box, red plastic lobster toy",
        "res": "1920×1080 (Femto ToF)",
    },
    {
        "stem": "data2_b",
        "label": "Desk scene 2B (bottle + blue box + lobster, second angle)",
        "objects_removed": "white water bottle, blue Arducam box, red plastic lobster toy",
        "res": "1920×1080 (Femto ToF)",
    },
    {
        "stem": "data4_a",
        "label": "Desk scene 4A (tape + plant + cardboard box)",
        "objects_removed": "yellow tape measure, small green artificial plant, tan cardboard box",
        "res": "1920×1080 (Femto ToF)",
    },
    {
        "stem": "data4_b",
        "label": "Desk scene 4B (tape + potted plant — box not visible)",
        "objects_removed": "yellow tape measure, small green artificial plant in black pot",
        "res": "1920×1080 (Femto ToF)",
    },
    {
        "stem": "data5_a",
        "label": "Living room 5A (blue armchair + cow-print teddy)",
        "objects_removed": "blue single-seat armchair, cow-print teddy bear pillow",
        "res": "3114×2074 (Fujifilm)",
    },
    {
        "stem": "data5_b",
        "label": "Living room 5B (blue armchair + cow-print teddy, second angle)",
        "objects_removed": "blue single-seat armchair, cow-print teddy bear pillow",
        "res": "3114×2074 (Fujifilm)",
    },
    {
        "stem": "data6_a",
        "label": "Living room 6A (tan slippers on rug)",
        "objects_removed": "pair of tan suede house slippers",
        "res": "3114×2074 (Fujifilm)",
    },
    {
        "stem": "data6_b",
        "label": "Living room 6B (tan slippers on rug, second angle)",
        "objects_removed": "pair of tan suede house slippers",
        "res": "3114×2074 (Fujifilm)",
    },
]

MODEL_INFO = [
    {
        "key":       "lama",
        "display":   "LaMa (IOPaint)",
        "uses_mask": True,
        "uses_prompt": False,
        "notes":     "Deterministic Fourier-CNN inpainter. Runs at native resolution. Mask is the only input; no text prompt.",
    },
    {
        "key":       "flux_fill",
        "display":   "FLUX.1 Fill dev (nf4)",
        "uses_mask": True,
        "uses_prompt": True,
        "notes":     "Masked diffusion inpainter. Needs BOTH a mask and a POSITIVE fill-description prompt (what should be there, not what to remove). guidance 30, 28 steps, 1024 long-edge.",
    },
    {
        "key":       "flux_kontext",
        "display":   "FLUX.1 Kontext dev (no-mask, nf4)",
        "uses_mask": False,
        "uses_prompt": True,
        "notes":     "Instruction-based editor. NO mask — prompt-only. guidance 4.0, 28 steps, 1024 long-edge.",
    },
    {
        "key":       "qwen_edit",
        "display":   "Qwen-Image-Edit-2511 (nf4 both + cpu_offload)",
        "uses_mask": False,
        "uses_prompt": True,
        "notes":     "20B DiT + Qwen2.5-VL-7B text encoder. NO mask — natural-language instruction. true_cfg 4.0, 28 steps, 1024 long-edge.",
    },
    {
        "key":       "fired_edit",
        "display":   "FireRed-Image-Edit-1.1 (nf4 both + cpu_offload)",
        "uses_mask": False,
        "uses_prompt": True,
        "notes":     "Same architecture as Qwen (QwenImageEditPlusPipeline). NO mask — instruction. true_cfg 4.0, 28 steps, 1024 long-edge.",
    },
]

# Per-image prompts must mirror _to_edit_shootout.sh
INSTRUCT = {
    "data2_a": "Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were.",
    "data2_b": "Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were.",
    "data4_a": "Remove the yellow tape measure, the small green artificial plant, and the tan cardboard box from the desk. Show only the empty desk surface where they were.",
    "data4_b": "Remove the yellow tape measure and the green artificial plant in the black pot from the desk. Show only the empty desk surface where they were.",
    "data5_a": "Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them.",
    "data5_b": "Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them.",
    "data6_a": "Remove the pair of tan slippers from the rug. Show the empty rug where they were.",
    "data6_b": "Remove the pair of tan slippers from the rug. Show the empty rug where they were.",
}
FILL = {
    "data2_a": "clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows",
    "data2_b": "clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows",
    "data4_a": "clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows",
    "data4_b": "clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows",
    "data5_a": "empty polished pine hardwood floor, grey linen curtains, warm indoor tungsten lighting, photorealistic living room, no chair, no pillow, no shadows",
    "data5_b": "empty polished pine hardwood floor, grey linen curtains, warm indoor tungsten lighting, photorealistic living room, no chair, no pillow, no shadows",
    "data6_a": "clean empty grey rug with dark grey square accent, warm indoor lighting, photorealistic living room floor, no slippers, no shoes",
    "data6_b": "clean empty grey rug with dark grey square accent, warm indoor lighting, photorealistic living room floor, no slippers, no shoes",
}


def _add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h


def _add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def _cell_image(cell, path: Path, width_in: float):
    """Insert an image into a cell (or show 'MISSING' if not found)."""
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    if path.exists():
        try:
            run.add_picture(str(path), width=Inches(width_in))
        except Exception as e:
            run.text = f"[image load error: {e}]"
            run.font.size = Pt(8)
    else:
        run.text = "[not generated]"
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        run.font.size = Pt(9)


def build_doc():
    doc = Document()

    # Page margins: narrow so images breathe
    for section in doc.sections:
        section.left_margin   = Inches(0.5)
        section.right_margin  = Inches(0.5)
        section.top_margin    = Inches(0.6)
        section.bottom_margin = Inches(0.6)

    # --- Cover page ---
    title = doc.add_heading("Image-Editing Model Shootout — to_edit dataset", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    _add_para(doc, f"Generated: {date.today().isoformat()}", italic=True, size=10)
    _add_para(doc, "5 models × 8 scenes = 40 edits. Each row: model, original image, "
                    "edited output, and exact mask/prompt inputs used.",
              size=11)
    doc.add_paragraph()
    _add_heading(doc, "Models tested", level=2)
    tbl = doc.add_table(rows=1 + len(MODEL_INFO), cols=4)
    tbl.style = "Light List Accent 1"
    hdr = tbl.rows[0].cells
    hdr[0].text = "Model"
    hdr[1].text = "Mask?"
    hdr[2].text = "Prompt?"
    hdr[3].text = "Notes"
    for i, m in enumerate(MODEL_INFO, start=1):
        row = tbl.rows[i].cells
        row[0].text = m["display"]
        row[1].text = "Yes" if m["uses_mask"] else "No"
        row[2].text = "Yes" if m["uses_prompt"] else "No"
        row[3].text = m["notes"]
    doc.add_paragraph()

    _add_heading(doc, "Scenes", level=2)
    for s in SCENES:
        p = doc.add_paragraph()
        p.add_run(f"• {s['label']}").bold = True
        p.add_run(f"  ({s['res']})  → remove: {s['objects_removed']}")
    doc.add_page_break()

    # --- Per-scene sections ---
    for s in SCENES:
        stem = s["stem"]
        _add_heading(doc, s["label"], level=1)
        _add_para(doc, f"Resolution: {s['res']}", italic=True, size=10)
        _add_para(doc, f"Objects to remove: {s['objects_removed']}", size=10)
        _add_para(doc, f"Instruction prompt (Kontext/Qwen/FireRed): \"{INSTRUCT[stem]}\"", size=9)
        _add_para(doc, f"Fill prompt (FLUX Fill only): \"{FILL[stem]}\"", size=9)

        # 5-row table: Model | Original | Edited | Notes
        tbl = doc.add_table(rows=1 + len(MODEL_INFO), cols=4)
        tbl.style = "Table Grid"
        # Column widths
        widths = (Inches(1.6), Inches(3.0), Inches(3.0), Inches(3.4))
        for row in tbl.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = w
        # Header
        h = tbl.rows[0].cells
        h[0].text = "Model"
        h[1].text = "Original"
        h[2].text = "Edited"
        h[3].text = "Mask / Prompt used"
        for cell in h:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

        img_original = IMG_DIR / f"{stem}.jpg"
        for i, m in enumerate(MODEL_INFO, start=1):
            row = tbl.rows[i].cells
            row[0].text = m["display"]
            row[0].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _cell_image(row[1], img_original, width_in=2.9)
            edited = OUT_DIR / m["key"] / f"{stem}.jpg"
            _cell_image(row[2], edited, width_in=2.9)

            # Notes cell
            row[3].text = ""
            np = row[3].paragraphs[0]
            np.alignment = WD_ALIGN_PARAGRAPH.LEFT
            note_bits = []
            if m["uses_mask"]:
                note_bits.append(f"Mask: SAM3 union of target objects (see mask preview at output/to_edit/masks/{stem}__union.png)")
            else:
                note_bits.append("Mask: none (instruction-only)")
            if m["uses_prompt"]:
                if m["key"] == "flux_fill":
                    note_bits.append(f"Prompt (positive fill description): \"{FILL[stem]}\"")
                else:
                    note_bits.append(f"Prompt (instruction): \"{INSTRUCT[stem]}\"")
            else:
                note_bits.append("Prompt: none")
            for b in note_bits:
                r = np.add_run(b + "\n")
                r.font.size = Pt(8)

        doc.add_page_break()

    doc.save(str(DOCX_OUT))
    print(f"[docx] saved {DOCX_OUT}  ({DOCX_OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build_doc()
