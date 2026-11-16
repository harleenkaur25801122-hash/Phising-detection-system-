"""
qr_decoder.py
-------------
Decodes a QR code image and extracts embedded content (typically a URL).
Uses OpenCV's built-in QRCodeDetector, so no extra dependency (pyzbar)
is required.
"""

import cv2
import numpy as np


def decode_qr_from_path(image_path: str) -> dict:
    """Decode a QR code from an image file on disk."""
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": "Could not read image file", "data": None}
    return decode_qr_from_array(img)


def decode_qr_from_bytes(image_bytes: bytes) -> dict:
    """Decode a QR code from raw image bytes (e.g. uploaded via an API)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "error": "Could not decode image bytes", "data": None}
    return decode_qr_from_array(img)


def decode_qr_from_array(img: np.ndarray) -> dict:
    """Core decode routine given an OpenCV image (numpy array)."""
    detector = cv2.QRCodeDetector()

    # Try single QR detect first
    data, points, _ = detector.detectAndDecode(img)
    if data:
        return {"success": True, "data": data, "num_codes_found": 1}

    # Fall back to multi-detect (in case there are multiple QR codes,
    # or the single-detect failed on a busy image)
    try:
        ok, decoded_info, pts, _ = detector.detectAndDecodeMulti(img)
        if ok and decoded_info:
            non_empty = [d for d in decoded_info if d]
            if non_empty:
                return {
                    "success": True,
                    "data": non_empty[0],
                    "all_data": non_empty,
                    "num_codes_found": len(non_empty),
                }
    except Exception:
        pass

    return {"success": False, "error": "No QR code detected in image", "data": None}


def is_url(data: str) -> bool:
    """Quick check whether decoded QR content looks like a URL."""
    if not data:
        return False
    data = data.strip().lower()
    return data.startswith("http://") or data.startswith("https://") or (
        "." in data and " " not in data and len(data) < 300
    )


if __name__ == "__main__":
    # Generate a quick synthetic test image containing a QR-like pattern
    # is non-trivial without the `qrcode` package (unavailable offline here).
    # In a real environment: pip install qrcode, then:
    #
    #   import qrcode
    #   img = qrcode.make("https://example.com")
    #   img.save("test_qr.png")
    #   print(decode_qr_from_path("test_qr.png"))
    print("qr_decoder module loaded. Use decode_qr_from_path(path) on a real QR image.")
