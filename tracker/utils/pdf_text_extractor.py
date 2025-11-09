"""
PDF and image text extraction without OCR.
Uses PyMuPDF (fitz) and PyPDF2 for PDF text extraction.
Falls back to pattern matching for invoice data extraction.
"""

import io
import logging
import re
from decimal import Decimal
from datetime import datetime

try:
    import fitz
except ImportError:
    fitz = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes) -> str:
    """Extract text from PDF file using PyMuPDF or PyPDF2.
    
    Args:
        file_bytes: Raw bytes of PDF file
        
    Returns:
        Extracted text string
        
    Raises:
        RuntimeError: If no PDF extraction library is available
    """
    text = ""
    
    # Try PyMuPDF first (fitz) - best for text extraction
    if fitz is not None:
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf_doc:
                text += page.get_text()
            pdf_doc.close()
            logger.info(f"Extracted {len(text)} characters from PDF using PyMuPDF")
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            text = ""
    
    # Fallback to PyPDF2
    if PyPDF2 is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text()
            logger.info(f"Extracted {len(text)} characters from PDF using PyPDF2")
            return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            text = ""
    
    if not text:
        raise RuntimeError('No PDF text extraction library available. Install PyMuPDF or PyPDF2.')
    
    return text


def extract_text_from_image(file_bytes) -> str:
    """Extract text from image file.
    Since OCR is not available, this returns empty string.
    Images should be uploaded as PDFs or entered manually.
    
    Args:
        file_bytes: Raw bytes of image file
        
    Returns:
        Empty string (manual entry required for images)
    """
    logger.info("Image file detected. OCR not available. Manual entry required.")
    return ""


def parse_invoice_data(text: str) -> dict:
    """Parse invoice data from extracted text using pattern matching.

    This method uses regex patterns to extract invoice fields from raw text.
    It's designed to work with professional invoice formats, especially:
    - Pro forma invoices with Code No, Customer Name, Address, Tel, Reference
    - Traditional invoices with Invoice Number, Date, Customer, etc.

    Args:
        text: Raw extracted text from PDF/image

    Returns:
        dict with extracted invoice data
    """
    if not text or not text.strip():
        return {
            'invoice_no': None,
            'code_no': None,
            'date': None,
            'customer_name': None,
            'address': None,
            'phone': None,
            'email': None,
            'reference': None,
            'subtotal': None,
            'tax': None,
            'total': None,
            'items': []
        }

    normalized_text = text.strip()
    lines = normalized_text.split('\n')

    # Clean and normalize lines
    cleaned_lines = []
    for line in lines:
        cleaned = line.strip()
        # Merge lines that are continuations (very short or just whitespace)
        if cleaned and len(cleaned) > 2:
            cleaned_lines.append(cleaned)

    # Helper to find field value - try multiple strategies including searching ahead
    def extract_field_value(label_patterns, text_to_search=None, max_distance=10):
        """Extract value after a label using flexible pattern matching and distance-based search.

        This handles cases where PDF extraction scrambles text ordering.
        It looks for the label, then finds the most likely value nearby in the text.
        """
        search_text = text_to_search or normalized_text
        patterns = label_patterns if isinstance(label_patterns, list) else [label_patterns]

        for pattern in patterns:
            # Strategy 1: Look for "Label: Value" or "Label = Value"
            m = re.search(rf'{pattern}\s*[:=]\s*([^\n:{{]+)', search_text, re.I | re.MULTILINE)
            if m and m.group(1).strip():
                value = m.group(1).strip()
                # Clean up trailing labels
                value = re.sub(r'\s+(?:Tel|Fax|Del|Ref|Date|Kind|Attended|Type|Payment|Delivery|Reference|PI|Cust|Qty|Rate|Value)\b.*$', '', value, flags=re.I).strip()
                if value:
                    return value

            # Strategy 2: "Label Value" (space separated, often in scrambled PDFs)
            m = re.search(rf'{pattern}\s+(?![:=])([A-Z][^\n:{{]*?)(?=\n[A-Z]|\s{2,}[A-Z]|\n$|$)', search_text, re.I | re.MULTILINE)
            if m and m.group(1).strip():
                value = m.group(1).strip()
                # Remove any trailing keywords
                value = re.sub(r'\s+(?:Tel|Fax|Del|Ref|Date|Kind|Attended|Type|Payment|Delivery|Reference|PI|Cust|Qty|Rate|Value|SR|NO)\b.*$', '', value, flags=re.I).strip()
                if value and len(value) > 2:
                    return value

            # Strategy 3: Find label, then look for value on next non-empty line
            lines = search_text.split('\n')
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.I):
                    # Check if value is on same line (after label)
                    m = re.search(rf'{pattern}\s*[:=]?\s*(.+)$', line, re.I)
                    if m:
                        value = m.group(1).strip()
                        if value and value.upper() not in (':', '=', '') and not re.match(r'^(?:Tel|Fax|Del|Ref|Date)\b', value, re.I):
                            return value

                    # Look for value on next 2-3 lines (handles scrambled layouts)
                    for j in range(1, min(4, len(lines) - i)):
                        next_line = lines[i + j].strip()
                        if next_line and not re.match(r'^[A-Z]+[a-zA-Z\s]*\s*[:=]', next_line):
                            # This looks like a value line
                            if len(next_line) > 2 and not re.match(r'^(?:Tel|Fax|Del|Ref|Date|SR|NO|Code|Customer|Address)\b', next_line, re.I):
                                return next_line
                        elif re.match(r'^[A-Z]+[a-zA-Z\s]*\s*[:=]', next_line):
                            # Hit another label, stop searching
                            break

        return None

    # Extract Code No (specific pattern for Superdoll invoices)
    code_no = extract_field_value([
        r'Code\s*No',
        r'Code\s*#',
        r'Code(?:\s|:)'
    ])

    # Extract customer name
    customer_name = extract_field_value([
        r'Customer\s*Name',
        r'Bill\s*To',
        r'Buyer\s*Name'
    ])

    # Extract address (look for lines after "Address" label)
    address = None
    for i, line in enumerate(cleaned_lines):
        if re.search(r'^Address\s*[:=]?', line, re.I):
            # Get this line value and next lines if they're not labels
            addr_parts = []
            m = re.search(r'^Address\s*[:=]?\s*(.+)$', line, re.I)
            if m:
                addr_parts.append(m.group(1).strip())
            # Collect next 2-3 lines as address continuation
            for j in range(1, 4):
                if i + j < len(cleaned_lines):
                    next_line = cleaned_lines[i + j]
                    # Stop if it's a new label
                    if re.match(r'^[A-Z]+[a-zA-Z\s]*\s*[:=]', next_line) or re.match(r'^(?:Tel|Fax|Del|Kind|Attended|Reference)', next_line, re.I):
                        break
                    # Stop if it's an obviously different section (all caps, ends with colon)
                    if next_line.isupper() and ':' in next_line:
                        break
                    addr_parts.append(next_line)
            address = ' '.join(addr_parts)
            if address:
                break

    # Extract phone/tel
    phone = extract_field_value(r'(?:Tel|Telephone)')
    if phone:
        # Remove "Fax" part if followed by fax number
        phone = re.sub(r'\s+Fax\s+.*$', '', phone, flags=re.I).strip()
        # Validate
        if phone and not re.search(r'\d{5,}', phone):
            phone = None

    # Extract email
    email = None
    email_match = re.search(r'([\w\.-]+@[\w\.-]+\.\w+)', normalized_text)
    if email_match:
        email = email_match.group(1)

    # Extract reference
    reference = extract_field_value(r'(?:Reference|Ref\.?|For|FOR)')

    # Extract PI No. / Invoice Number
    invoice_no = extract_field_value([
        r'PI\s*(?:No|Number|#)',
        r'Invoice\s*(?:No|Number)'
    ])

    # Extract Date (multiple formats)
    date_str = None
    # Look for date patterns
    date_patterns = [
        r'Date\s*[:=]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'Invoice\s*Date\s*[:=]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',  # Fallback: any date pattern
    ]
    for pattern in date_patterns:
        m = re.search(pattern, normalized_text, re.I)
        if m:
            date_str = m.group(1)
            break

    # Parse monetary values helper
    def to_decimal(s):
        try:
            if s:
                # Remove currency symbols and extra characters, keep only numbers, dot, comma
                cleaned = re.sub(r'[^\d\.\,\-]', '', str(s)).strip()
                if cleaned and cleaned not in ('.', ',', '-'):
                    return Decimal(cleaned.replace(',', ''))
        except Exception:
            pass
        return None

    # Extract monetary amounts using flexible patterns
    def find_amount(label_patterns):
        """Find monetary amount after label patterns"""
        patterns = (label_patterns if isinstance(label_patterns, list) else [label_patterns])
        for pattern in patterns:
            # Try with colon
            m = re.search(rf'{pattern}\s*:\s*(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
            if m:
                return m.group(1)
            # Try with equals
            m = re.search(rf'{pattern}\s*=\s*(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
            if m:
                return m.group(1)
            # Try with just space
            m = re.search(rf'{pattern}\s+(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
            if m:
                return m.group(1)
        return None

    # Extract Net Value / Subtotal
    subtotal = to_decimal(find_amount([
        r'Net\s*Value',
        r'Net\s*Amount',
        r'Subtotal',
        r'Net\s*:'
    ]))

    # Extract VAT / Tax
    tax = to_decimal(find_amount([
        r'VAT',
        r'Tax',
        r'GST',
        r'Sales\s*Tax'
    ]))

    # Extract Gross Value / Total
    total = to_decimal(find_amount([
        r'Gross\s*Value',
        r'Total\s*Amount',
        r'Grand\s*Total',
        r'Total\s*(?::|\s)'
    ]))

    # Extract line items with improved detection
    items = []
    item_section_started = False
    item_header_idx = -1

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Detect item section header more flexibly
        # Look for lines containing multiple item-related keywords
        keyword_count = sum([
            1 if re.search(r'\b(?:Sr|S\.N|Serial)\b', line_stripped, re.I) else 0,
            1 if re.search(r'\b(?:Item|Code)\b', line_stripped, re.I) else 0,
            1 if re.search(r'\b(?:Description|Desc)\b', line_stripped, re.I) else 0,
            1 if re.search(r'\b(?:Qty|Quantity)\b', line_stripped, re.I) else 0,
            1 if re.search(r'\b(?:Rate|Price|Unit)\b', line_stripped, re.I) else 0,
            1 if re.search(r'\b(?:Value|Amount)\b', line_stripped, re.I) else 0,
        ])

        if keyword_count >= 3:
            item_section_started = True
            item_header_idx = idx
            continue

        # Stop at totals/summary section
        if item_section_started and idx > item_header_idx + 1:
            if re.search(r'(?:Net\s*Value|Gross\s*Value|Total|Payment|Delivery|Remarks|NOTE)', line_stripped, re.I):
                break

        # Parse item lines
        if item_section_started and idx > item_header_idx:
            # Skip empty or obvious continuation lines
            if not line_stripped or len(line_stripped) < 5:
                continue

            # Check if this looks like a data line (contains numbers)
            numbers = re.findall(r'[0-9\,]+\.?\d*', line_stripped)
            if len(numbers) < 1:
                continue

            # Skip if line starts with only whitespace (continuation)
            if line and line[0] in (' ', '\t'):
                if items and len(line_stripped) > 2:
                    # Append to last item description
                    items[-1]['description'] = (items[-1]['description'] + ' ' + line_stripped)[:255]
                continue

            # Try to identify serial number at start (if present, it's usually the first number and < 100)
            start_idx = 0
            try:
                if len(numbers) > 0:
                    first_num_val = float(numbers[0].replace(',', '').split('.')[0])
                    if 1 <= first_num_val < 100:
                        start_idx = 1
            except Exception:
                pass

            remaining_numbers = numbers[start_idx:]
            if len(remaining_numbers) < 1:
                continue

            # Extract description by removing numbers from line
            desc = re.sub(r'\s*[0-9\,]+\.?\d*\s*', ' | ', line_stripped)
            desc_parts = [p.strip() for p in desc.split('|') if p.strip() and len(p.strip()) > 2]
            desc = ' '.join(desc_parts)
            desc = ' '.join(desc.split())  # Normalize whitespace

            if not desc or len(desc) < 3:
                continue

            # Extract value (last number) and qty (second-to-last)
            value = remaining_numbers[-1] if remaining_numbers else None
            qty = 1

            if len(remaining_numbers) >= 2:
                try:
                    qty_candidate = float(remaining_numbers[-2].replace(',', ''))
                    # Check if it looks like a reasonable quantity (not a price)
                    if 0.1 < qty_candidate < 10000:
                        # If it has decimals, it's probably a price, not qty
                        if '.' not in remaining_numbers[-2] or qty_candidate.is_integer():
                            qty = int(qty_candidate) if qty_candidate.is_integer() else qty_candidate
                except Exception:
                    pass

            items.append({
                'description': desc[:255],
                'qty': qty if isinstance(qty, int) else int(qty) if isinstance(qty, float) and qty.is_integer() else qty,
                'value': to_decimal(value)
            })

    return {
        'invoice_no': invoice_no,
        'code_no': code_no,
        'date': date_str,
        'customer_name': customer_name,
        'phone': phone,
        'email': email,
        'address': address,
        'reference': reference,
        'subtotal': subtotal,
        'tax': tax,
        'total': total,
        'items': items
    }


def extract_from_bytes(file_bytes, filename: str = '') -> dict:
    """Main entry point: extract text from file and parse invoice data.
    
    Supports:
    - PDF files: Uses PyMuPDF/PyPDF2 for text extraction
    - Image files: Requires manual entry (OCR not available)
    
    Args:
        file_bytes: Raw bytes of uploaded file
        filename: Original filename (to detect file type)
        
    Returns:
        dict with keys: success, header, items, raw_text, ocr_available, error, message
    """
    if not file_bytes:
        return {
            'success': False,
            'error': 'empty_file',
            'message': 'File is empty',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }
    
    # Detect file type
    is_pdf = filename.lower().endswith('.pdf') or file_bytes[:4] == b'%PDF'
    is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp'))
    
    text = ""
    extraction_error = None
    
    # Try to extract text
    if is_pdf:
        try:
            text = extract_text_from_pdf(file_bytes)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            extraction_error = str(e)
            return {
                'success': False,
                'error': 'pdf_extraction_failed',
                'message': f'Failed to extract text from PDF: {str(e)}. Please enter invoice details manually.',
                'ocr_available': False,
                'header': {},
                'items': [],
                'raw_text': ''
            }
    elif is_image:
        return {
            'success': False,
            'error': 'image_file_not_supported',
            'message': 'Image files require manual entry (OCR not available). Please save as PDF or enter details manually.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }
    else:
        return {
            'success': False,
            'error': 'unsupported_file_type',
            'message': 'Please upload a PDF file (images are not supported without OCR).',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }
    
    # Parse extracted text
    if text:
        try:
            parsed = parse_invoice_data(text)
            # Prepare header with all extracted fields
            header = {
                'invoice_no': parsed.get('invoice_no'),
                'code_no': parsed.get('code_no'),
                'date': parsed.get('date'),
                'customer_name': parsed.get('customer_name'),
                'phone': parsed.get('phone'),
                'email': parsed.get('email'),
                'address': parsed.get('address'),
                'reference': parsed.get('reference'),
                'subtotal': parsed.get('subtotal'),
                'tax': parsed.get('tax'),
                'total': parsed.get('total'),
            }
            return {
                'success': True,
                'header': header,
                'items': parsed.get('items', []),
                'raw_text': text,
                'ocr_available': False,  # Using text extraction, not OCR
                'message': 'Invoice data extracted successfully from PDF'
            }
        except Exception as e:
            logger.warning(f"Failed to parse invoice data: {e}")
            return {
                'success': False,
                'error': 'parsing_failed',
                'message': 'Could not extract structured data from PDF. Please enter invoice details manually.',
                'ocr_available': False,
                'header': {},
                'items': [],
                'raw_text': text
            }
    
    # If no text was extracted
    return {
        'success': False,
        'error': 'no_text_extracted',
        'message': 'No text found in PDF. Please enter invoice details manually.',
        'ocr_available': False,
        'header': {},
        'items': [],
        'raw_text': ''
    }
