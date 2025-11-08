"""
Simple OCR + regex-based invoice extractor using pytesseract and OpenCV.
This is a pragmatic extractor intended for Phase-1: reasonably-structured invoices
(like the Superdoll example). It returns a dict with header fields and a list of items.

If pytesseract or OpenCV are not installed, falls back to regex-based extraction on plain text.
"""
from PIL import Image
import io
import re
import logging
from decimal import Decimal

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

logger = logging.getLogger(__name__)

# Check if dependencies are available
OCR_AVAILABLE = pytesseract is not None and cv2 is not None


def _image_from_bytes(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert('RGB')


def preprocess_image_pil(img_pil):
    """Convert PIL image -> OpenCV -> simple preprocessing -> back to PIL"""
    if cv2 is None or np is None:
        return img_pil
    arr = np.array(img_pil)
    # Convert to gray
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Resize if too small
    h, w = gray.shape[:2]
    if w < 1000:
        scale = 1000.0 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # Denoise and threshold
    blur = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert back to PIL
    return Image.fromarray(th)


def ocr_image(img_pil):
    if pytesseract is None:
        raise RuntimeError('pytesseract is not available')
    # Simple config: treat as single column text but allow some detection
    config = '--psm 6'
    text = pytesseract.image_to_string(img_pil, config=config)
    return text


def extract_header_fields(text):
    # Helper to find first match group
    def find(pattern):
        m = re.search(pattern, text, re.I)
        return m.group(1).strip() if m else None

    invoice_no = find(r'(?:PI|PI\.?|Invoice|Invoice No|PI No)[\s:\-]*([A-Z0-9\-\/]+)')
    code_no = find(r'(?:Code No|Code)[\s:\-]*([A-Z0-9\-]+)')
    date_str = find(r'\bDate\b[\s:\-]*([0-3]?\d[\-/][01]?\d[\-/]\d{2,4})')
    customer_name = find(r'(?:Customer Name|Customer|Bill To|Buyer)[:\s\-]*([^\n\r\:]{3,120})')
    address = find(r'(?:Address|Addr\.|Add)[:\s\-]*([^\n\r]{5,200})')

    # Totals
    net = find(r'(?:Net Value|Net)[\s:\-]*([0-9\,]+\.?\d{0,2})')
    vat = find(r'(?:VAT|Tax)[\s:\-]*([0-9\,]+\.?\d{0,2})')
    gross = find(r'(?:Gross Value|Gross)[\s:\-]*(?:TSH)?\s*([0-9\,]+\.?\d{0,2})')

    def to_decimal(s):
        try:
            return Decimal(str(s).replace(',', ''))
        except Exception:
            return None

    return {
        'invoice_no': invoice_no,
        'code_no': code_no,
        'date': date_str,
        'customer_name': customer_name,
        'address': address,
        'net_value': to_decimal(net) if net else None,
        'vat': to_decimal(vat) if vat else None,
        'gross_value': to_decimal(gross) if gross else None,
    }


def extract_line_items(text):
    """Very simple heuristic to extract lines that look like: Sr  ItemCode  Description  Qty  Rate  Value
    We will scan lines and pick those containing at least two numbers and one large number-looking value.
    """
    items = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Try to find the table header index by looking for 'Item' and 'Qty' and 'Value'
    header_idx = None
    for idx, line in enumerate(lines[:30]):
        if re.search(r'\b(Item|Description)\b', line, re.I) and re.search(r'\bQty\b', line, re.I):
            header_idx = idx
            break
    # If header found, parse subsequent lines
    start = header_idx + 1 if header_idx is not None else 0
    for line in lines[start:]:
        # stop if footer keywords
        if re.search(r'\b(Net Value|Total|Gross Value|VAT|Payment)\b', line, re.I):
            break
        # Find numbers with decimal or commas
        numbers = re.findall(r'[0-9\,]+\.?\d*', line)
        if len(numbers) >= 2:
            # Heuristic mapping
            # Try to capture item code as first short number
            item_code = None
            qty = None
            rate = None
            value = None
            # If line starts with serial and item code
            parts = re.split(r'\s{2,}|\t', line)
            # fallback: split by spaces
            if len(parts) < 3:
                parts = line.split()
            # Find last numeric token as value
            numeric_tokens = re.findall(r'([0-9\,]+\.?\d*)', line)
            if numeric_tokens:
                value = numeric_tokens[-1]
            # qty is likely a small integer near end
            if len(numeric_tokens) >= 2:
                qty = numeric_tokens[-2]
            # try to find item code as first token with 3-6 digits
            m = re.search(r'\b(\d{3,6})\b', line)
            if m:
                item_code = m.group(1)
            # description: remove numeric tokens from line
            desc = re.sub(r'[0-9\,]+\.?\d*', '', line).strip()
            # Clean values
            def clean_num(s):
                try:
                    return Decimal(s.replace(',', ''))
                except Exception:
                    return None
            items.append({
                'item_code': item_code,
                'description': desc[:255],
                'qty': Decimal(qty.replace(',', '')) if qty and re.match(r'^[0-9\,]+\.?\d*$', qty) else None,
                'rate': clean_num(rate) if rate else None,
                'value': clean_num(value) if value else None,
            })
    return items


def extract_from_bytes(file_bytes):
    """Main entry: take raw bytes, preprocess, OCR, parse and return result dict."""
    try:
        img = _image_from_bytes(file_bytes)
    except Exception as e:
        logger.warning(f"Failed to open uploaded file as image: {e}")
        return {'error': 'invalid_image', 'message': 'Could not open file as image'}

    proc = preprocess_image_pil(img)

    try:
        text = ocr_image(proc)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {'error': 'ocr_failed', 'message': str(e)}

    header = extract_header_fields(text)
    items = extract_line_items(text)

    result = {
        'header': header,
        'items': items,
        'raw_text': text
    }
    return result
