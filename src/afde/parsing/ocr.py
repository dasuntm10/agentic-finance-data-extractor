"""OCR fallback for scanned pages (B&E FOODS) and font-encoded pages (AUSNET).

PaddleOCR is the primary engine; Tesseract is a secondary. Both are optional —
if neither is installed the pipeline logs a warning and degrades gracefully.
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)


def _rasterise(pdf_path: Path, page_number: int, dpi: int = 300) -> bytes | None:
    try:
        import fitz  # noqa: PLC0415

        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=dpi)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception as e:
        log.error("Rasterise failed for %s p.%d: %s", pdf_path, page_number, e)
        return None


def ocr_page(pdf_path: str | Path, page_number: int, dpi: int = 300) -> tuple[str, float]:
    """Return (text, average_confidence_0_to_1). Tries PaddleOCR then Tesseract."""
    img_bytes = _rasterise(Path(pdf_path), page_number, dpi=dpi)
    if img_bytes is None:
        return "", 0.0

    # --- Try PaddleOCR ---
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        ocr = _paddle_singleton()
        img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        result = ocr.ocr(img, cls=False)
        lines: list[str] = []
        confs: list[float] = []
        for block in result or []:
            for line in block or []:
                _box, (text, conf) = line
                lines.append(text)
                confs.append(float(conf))
        text = "\n".join(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf
    except Exception as e:
        log.debug("PaddleOCR unavailable or failed: %s", e)

    # --- Tesseract fallback ---
    try:
        import pytesseract  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        img = Image.open(BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data.get("conf", []) if c not in ("-1", -1, "")]
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, avg_conf
    except Exception as e:
        log.warning("OCR fallback unavailable: %s", e)
        return "", 0.0


_PADDLE = None


def _paddle_singleton():
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR  # noqa: PLC0415

        _PADDLE = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _PADDLE
