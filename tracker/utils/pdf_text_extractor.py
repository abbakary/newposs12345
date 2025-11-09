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

    # Helper to validate if text looks like a customer name vs address
    def is_likely_customer_name(text):
        """Check if text looks like a company/person name vs an address."""
        if not text:
            return False
        # Customer names are usually shorter, no commas or street keywords
        address_keywords = ['street', 'avenue', 'road', 'box', 'p.o', 'po', 'floor', 'apt', 'suite', 'district', 'region', 'country']
        is_short = len(text) < 80
        has_no_address_keywords = not any(kw in text.lower() for kw in address_keywords)
        is_capitalized = text[0].isupper() if text else False
        return is_short and has_no_address_keywords and is_capitalized

    def is_likely_address(text):
        """Check if text looks like an address."""
        if not text:
            return False
        # Addresses often contain locations, street info, numbers, or are multi-word with specific patterns
        address_indicators = ['street', 'avenue', 'road', 'box', 'p.o', 'po', 'floor', 'apt', 'suite',
                             'district', 'region', 'city', 'country', 'zip', 'postal', 'dar', 'dar-es', 'tanzania', 'nairobi', 'kenya']
        has_indicators = any(ind in text.lower() for ind in address_indicators)
        has_numbers = bool(re.search(r'\d+', text))
        is_longer = len(text) > 15
        return has_indicators or (has_numbers and is_longer)

    # Extract customer name
    customer_name = extract_field_value([
        r'Customer\s*Name',
        r'Bill\s*To',
        r'Buyer\s*Name',
        r'Client\s*Name'
    ])

    # Validate customer name - if it looks like an address, clear it
    if customer_name and is_likely_address(customer_name) and not is_likely_customer_name(customer_name):
        customer_name = None

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

    # Smart fix: If customer_name is empty but address looks like a name, swap them
    if not customer_name and address and is_likely_customer_name(address):
        customer_name = address
        address = None

    # Also check reverse: if customer_name looks like address and address is empty, swap
    if customer_name and is_likely_address(customer_name) and not is_likely_customer_name(customer_name):
        if not address:
            address = customer_name
            customer_name = None

    # Extract phone/tel
    phone = extract_field_value(r'(?:Tel|Telephone|Phone)')
    if phone:
        # Remove "Fax" part if followed by fax number
        phone = re.sub(r'\s+Fax\s+.*$', '', phone, flags=re.I).strip()
        # Validate - phone should have some digits
        if phone and not re.search(r'\d{5,}', phone):
            phone = None
        # Clean up - remove common non-digit prefixes and ensure we have a phone
        if phone:
            phone = re.sub(r'^(?:Tel|Phone|Telephone)\s*[:=]?\s*', '', phone, flags=re.I).strip()

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

    # Extract monetary amounts using flexible patterns (handles scrambled PDFs)
    def find_amount(label_patterns):
        """Find monetary amount after label patterns - works with scrambled PDF text"""
        patterns = (label_patterns if isinstance(label_patterns, list) else [label_patterns])
        for pattern in patterns:
            # Try with colon separator: "Label: Amount"
            m = re.search(rf'{pattern}\s*:\s*(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
            if m:
                return m.group(1)

            # Try with equals: "Label = Amount"
            m = re.search(rf'{pattern}\s*=\s*(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
            if m:
                return m.group(1)

            # Try with space and optional currency on same line
            m = re.search(rf'{pattern}\s+(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
            if m:
                return m.group(1)

            # Try finding amount on next line (for scrambled PDFs)
            lines = normalized_text.split('\n')
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.I):
                    # Check for amount on same line
                    m = re.search(rf'{pattern}\s*[:=]?\s*([0-9\,\.]+)', line, re.I)
                    if m:
                        return m.group(1)

                    # Check next 2 lines for amount
                    for j in range(1, 3):
                        if i + j < len(lines):
                            next_line = lines[i + j].strip()
                            # Look for amount pattern
                            if re.match(r'^(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', next_line, re.I):
                                m = re.match(r'^(?:TSH|TZS|UGX)?\s*([0-9\,\.]+)', next_line, re.I)
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

    # Extract line items with improved detection for scrambled PDFs
    items = []
    item_section_started = False
    item_header_idx = -1
    current_item = {}

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            # Empty line might signal end of current item
            if current_item and ('description' in current_item or 'value' in current_item):
                # Finalize current item if we have enough info
                if current_item.get('description') and (current_item.get('value') or current_item.get('qty')):
                    items.append(current_item)
                    current_item = {}
            continue

        # Detect item section header
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
            if re.search(r'(?:Net\s*Value|Gross\s*Value|Payment|Delivery|Remarks|NOTE)', line_stripped, re.I):
                # Finalize any pending item
                if current_item and ('description' in current_item or 'value' in current_item):
                    items.append(current_item)
                break

        # Parse item lines (after header starts)
        if item_section_started and idx > item_header_idx:
            # Look for numeric patterns - could be sr no, code, qty, rate, or value
            numbers = re.findall(r'[0-9\,]+\.?\d*', line_stripped)

            # Extract text (non-numeric part)
            text_only = re.sub(r'[0-9\,]+\.?\d*', '|', line_stripped)
            text_parts = [p.strip() for p in text_only.split('|') if p.strip()]

            # Case 1: Line looks like "Sr Code Description Qty Rate Value" (table row)
            if len(numbers) >= 2 and text_parts:
                # Try to identify what the numbers represent
                # Usually: Sr#, ItemCode, Qty, Rate, Value
                # Extract numeric values more carefully

                # Description is text parts joined
                desc = ' '.join(text_parts)
                if desc and len(desc) > 2:
                    # Extract numbers - usually qty and value are smaller/larger respectively
                    value = numbers[-1] if len(numbers) >= 1 else None
                    qty = 1

                    # If 2+ numbers, second-to-last might be qty
                    if len(numbers) >= 2:
                        try:
                            qty_candidate = float(numbers[-2].replace(',', ''))
                            if 0.1 < qty_candidate < 10000:
                                if '.' not in numbers[-2] or qty_candidate.is_integer():
                                    qty = int(qty_candidate) if qty_candidate.is_integer() else qty_candidate
                        except Exception:
                            pass

                    current_item = {
                        'description': desc[:255],
                        'qty': qty if isinstance(qty, int) else int(qty) if isinstance(qty, float) and qty.is_integer() else 1,
                        'value': to_decimal(value)
                    }
                    items.append(current_item)
                    current_item = {}

            # Case 2: Line is purely descriptive text (likely description for current item)
            elif text_parts and not numbers:
                # This is likely a description line
                desc_text = ' '.join(text_parts)
                if desc_text and len(desc_text) > 2:
                    if 'description' not in current_item:
                        current_item['description'] = desc_text[:255]
                    else:
                        # Append to existing description
                        current_item['description'] = (current_item['description'] + ' ' + desc_text)[:255]

            # Case 3: Line with just numbers (could be qty, rate, or value)
            elif numbers and not text_parts:
                # Try to figure out what this number represents
                # Usually small numbers are qty, large numbers are values/rates
                try:
                    num_val = float(numbers[0].replace(',', ''))
                    if 0.1 < num_val < 1000 and '.' not in numbers[0]:
                        # Looks like a quantity
                        current_item['qty'] = int(num_val)
                    else:
                        # Looks like a monetary amount
                        if 'value' not in current_item:
                            current_item['value'] = to_decimal(numbers[0])
                except Exception:
                    pass

    # Finalize any pending item
    if current_item and ('description' in current_item or 'value' in current_item):
        items.append(current_item)

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
