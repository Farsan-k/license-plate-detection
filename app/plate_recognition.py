import re
from fast_plate_ocr import LicensePlateRecognizer


ocr_model = LicensePlateRecognizer("cct-xs-v1-global-model")


PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{4}[A-Z]{2}$"),
    re.compile(r"^[0-9]{4}[A-Z]{3}$")
]


def clean_ocr_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate(text):
    text = clean_ocr_text(text)
    return any(pattern.fullmatch(text) for pattern in PLATE_PATTERNS)


def preprocess_plate(plate_crop):
    """Light preprocessing only — fast-plate-ocr expects a fairly
    clean, color (3-channel) crop close to the original plate, and
    handles its own internal resizing, so we avoid the heavy
    multi-threshold grayscale pipeline that was mainly compensating
    for EasyOCR's weaknesses."""

    h, w = plate_crop.shape[:2]

    crop_x1 = int(w * 0.02)
    crop_x2 = int(w * 0.98)
    crop_y1 = int(h * 0.05)
    crop_y2 = int(h * 0.95)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    plate_crop = plate_crop[crop_y1:crop_y2, crop_x1:crop_x2]

    if plate_crop.size == 0:
        return None

    return plate_crop


def recognize_plate(plate_crop):

    if plate_crop is None or plate_crop.size == 0:
        return "", 0.0

    try:
        processed = preprocess_plate(plate_crop)

        if processed is None:
            return "", 0.0

        result = ocr_model.run(processed, return_confidence=True)

        if not result or not result[0].plate:
            return "", 0.0

        prediction = result[0]

        text = clean_ocr_text(prediction.plate)

        if prediction.char_probs is not None and len(prediction.char_probs) > 0:
            confidence = float(prediction.char_probs.mean())
        else:
            confidence = 1.0

        print(f"OCR RAW: {text} | conf={confidence:.3f}")

        if not is_valid_plate(text):
            print(f"OCR REJECTED (format mismatch): {text}")
            return "", 0.0

        print(f"FINAL OCR: {text}")

        return text, confidence

    except Exception as e:
        print("OCR ERROR:", e)
        return "", 0.0