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

    # Helper function to extract value after a label
    def extract_field(label_pattern, text_to_search=None, multiline=False):
        """Extract value after a label pattern"""
        search_text = text_to_search or normalized_text

        if multiline:
            # For multiline fields like address, capture until next label line
            pattern = rf'{label_pattern}\s*[:=\s]\s*((?:[^\n]+(?:\n(?![A-Za-z]+\s*[:=])[^\n]*)?)*)'
            m = re.search(pattern, search_text, re.I | re.MULTILINE)
            if m:
                result = m.group(1).strip()
                # Clean up by removing trailing label lines
                result = result.split('\n')[0:3]  # Take up to 3 lines
                result = ' '.join([line.strip() for line in result if line.strip()])
                return result if result else None
        else:
            # Single line field
            pattern = rf'{label_pattern}\s*[:=\s]\s*([^\n]+?)(?:\n|$)'
            m = re.search(pattern, search_text, re.I | re.MULTILINE)
            if m:
                result = m.group(1).strip()
                # Clean up trailing noise like labels
                result = re.sub(r'\s+(Tel|Fax|Del\.|Ref|Date|PI|Cust|Kind|Attended|Type|Payment|Delivery|Remarks|NOTE)\s*.*$', '', result, flags=re.I)
                result = ' '.join(result.split())  # Normalize whitespace
                return result if result else None
        return None

    # Extract Code No
    code_no = extract_field(r'Code\s*(?:No|Number|#)')

    # Extract customer name
    customer_name = extract_field(r'Customer\s*Name')

    # Extract address (multiline)
    address = extract_field(r'Address', multiline=True)

    # Extract phone/tel
    phone = extract_field(r'(?:Tel|Telephone|Phone)')
    # Clean up phone number - remove "Fax" part if present
    if phone:
        phone = re.sub(r'\s*Fax.*$', '', phone, flags=re.I).strip()
        # Ensure it has at least 7 digits
        if not re.search(r'\d{7,}', phone):
            phone = None

    # Extract email (from header or fields)
    email = None
    email_match = re.search(r'([^\s\n]+@[^\s\n]+)', normalized_text)
    if email_match:
        email = email_match.group(1)

    # Extract reference
    reference = extract_field(r'Reference')

    # Extract PI No. / Invoice Number
    invoice_no = extract_field(r'(?:PI\s*(?:No|Number)|Invoice\s*(?:No|Number))')

    # Extract Date
    date_str = None
    # Look for date in DD/MM/YYYY or DD-MM-YYYY format
    date_match = re.search(r'Date\s*[:=\s]\s*([0-3]?\d[/\-][01]?\d[/\-]\d{2,4})', normalized_text, re.I | re.MULTILINE)
    if date_match:
        date_str = date_match.group(1)

    # Parse monetary values helper
    def to_decimal(s):
        try:
            if s:
                # Remove currency symbols and clean
                cleaned = re.sub(r'[^\d\.\,\-]', '', str(s)).strip()
                if cleaned:
                    return Decimal(cleaned.replace(',', ''))
        except Exception:
            pass
        return None

    # Extract Net Value / Subtotal
    subtotal = None
    net_match = re.search(r'Net\s*(?:Value|Amount)\s*[:=\s]\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
    if net_match:
        subtotal = to_decimal(net_match.group(1))

    # Extract VAT / Tax
    tax = None
    vat_match = re.search(r'VAT\s*[:=\s]\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
    if vat_match:
        tax = to_decimal(vat_match.group(1))

    # Extract Gross Value / Total
    total = None
    gross_match = re.search(r'Gross\s*Value\s*[:=\s]*(?:TSH)?\s*([0-9\,\.]+)', normalized_text, re.I | re.MULTILINE)
    if gross_match:
        total = to_decimal(gross_match.group(1))

    # Extract line items
    items = []
    item_section_started = False
    item_header_idx = -1
    skip_next = 0

    for idx, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue

        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Look for the table header with "Sr" "Item Code" "Description" etc
        if re.search(r'\bSr\b.*\b(?:Item\s*Code|Code)\b.*\b(?:Description|Desc)\b', line_stripped, re.I) or \
           re.search(r'\b(?:Description|Desc)\b.*\b(?:Qty|Quantity)\b.*\b(?:Rate|Price|Value|Amount)\b', line_stripped, re.I):
            item_section_started = True
            item_header_idx = idx
            continue

        # Stop when we hit the totals section
        if item_section_started and re.search(r'(?:Net\s*Value|Gross\s*Value|Total|Payment|Delivery|Remarks|NOTE)', line_stripped, re.I):
            break

        # Parse data lines (after header, skip empty lines)
        if item_section_started and idx > item_header_idx:
            # Look for lines with serial number or item code at start
            # Format: Sr No. | Item Code | Description | Qty | Rate | Value

            # Skip if line looks like a continuation of description (starts with many spaces or lowercase)
            if line and line[0] in (' ', '\t') and not re.match(r'^\s*\d+\s+', line):
                # This is a continuation line, check if we should append to last item
                if items and len(line_stripped) > 2:
                    # Append to last item's description
                    items[-1]['description'] = items[-1]['description'] + ' ' + line_stripped
                continue

            # Extract numbers from the line
            numbers = re.findall(r'[0-9\,]+\.?\d*', line_stripped)

            # Should have at least 2 numbers (qty and value)
            if len(numbers) >= 1 and len(line_stripped) > 5:
                # Remove serial number if present (first number if < 100)
                start_idx = 0
                try:
                    if numbers and int(numbers[0].replace(',', '').split('.')[0]) < 100:
                        start_idx = 1
                except Exception:
                    pass

                # Get remaining numbers after removing serial
                remaining_numbers = numbers[start_idx:] if start_idx < len(numbers) else numbers

                if len(remaining_numbers) >= 1:
                    # Extract description by removing numbers from line
                    desc = re.sub(r'[0-9\,]+\.?\d*', '|', line_stripped)
                    desc_parts = [p.strip() for p in desc.split('|') if p.strip()]
                    desc = ' '.join(desc_parts) if desc_parts else ''

                    # Clean description
                    desc = ' '.join(desc.split())

                    if desc and len(desc) > 2:
                        # Last number is typically the value/amount
                        value = remaining_numbers[-1] if remaining_numbers else None

                        # Qty is typically a small integer, look for it
                        qty = 1
                        if len(remaining_numbers) >= 2:
                            try:
                                # Check second to last as qty
                                qty_candidate = int(float(remaining_numbers[-2].replace(',', '')))
                                if 0 < qty_candidate < 1000:
                                    qty = qty_candidate
                            except Exception:
                                pass

                        items.append({
                            'description': desc[:255],
                            'qty': qty,
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
