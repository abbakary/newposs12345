"""
Invoice document processing utility for extracting data from PDF and image files.
Handles OCR using pytesseract and PDF processing using PyMuPDF.
"""

import os
import logging
from decimal import Decimal
from pathlib import Path
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class InvoiceDataExtractor:
    """Extract invoice data from PDF and image files using OCR."""

    # Patterns for matching common invoice fields
    KEYWORDS_CUSTOMER_NAME = [
        "customer name", "customer:", "bill to", "sold to", "customer ref", "code no"
    ]
    KEYWORDS_CUSTOMER_ADDRESS = [
        "address", "location", "customer address", "bill to", "p.o.box", "dar", "dar-es-salaam"
    ]
    KEYWORDS_CUSTOMER_PHONE = [
        "tel", "phone", "telephone", "mobile", "contact"
    ]
    KEYWORDS_ITEMS = [
        "item", "description", "product", "service", "qty", "quantity"
    ]
    KEYWORDS_TOTALS = [
        "total", "subtotal", "gross", "net", "vat", "tax", "amount"
    ]
    KEYWORDS_REFERENCE = [
        "reference", "ref", "po", "order", "invoice no", "pi no", "pi-"
    ]

    def __init__(self, file_path=None, file_obj=None):
        """
        Initialize with either a file path or a file-like object.
        
        Args:
            file_path: Path to PDF or image file
            file_obj: File-like object (InMemoryUploadedFile from Django)
        """
        self.file_path = file_path
        self.file_obj = file_obj
        self.file_extension = None
        self._validate_file()

    def _validate_file(self):
        """Validate that file exists and is supported format.

        This method is defensive: it ensures uploaded file-like objects have
        a usable name and that the detected extension is supported. Raises
        a clear ValueError when input is invalid so callers can handle it.
        """
        if self.file_path:
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            self.file_extension = path.suffix.lower()
        elif self.file_obj:
            # Uploaded file objects from Django should have a .name attribute.
            name = getattr(self.file_obj, 'name', None)
            if not name:
                raise ValueError("Uploaded file is missing a name attribute")
            self.file_extension = Path(name).suffix.lower()
        else:
            raise ValueError("Either file_path or file_obj must be provided")

        supported = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        if self.file_extension not in supported:
            raise ValueError(f"Unsupported file format: {self.file_extension}")

    def extract_text(self) -> str:
        """Extract all text from the document using OCR."""
        try:
            if self.file_extension == '.pdf':
                return self._extract_text_from_pdf()
            else:
                return self._extract_text_from_image()
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    def _extract_text_from_pdf(self) -> str:
        """Extract text from PDF using PyMuPDF.

        Be defensive about reading the uploaded file-like object because some
        file wrappers may behave differently (e.g. TemporaryUploadedFile).
        """
        try:
            if self.file_obj:
                # Ensure we're at the start of the file-like stream
                try:
                    self.file_obj.seek(0)
                except Exception:
                    pass

                pdf_bytes = self.file_obj.read()
                if not pdf_bytes:
                    raise ValueError("Uploaded PDF file appears empty after reading")

                # fitz.open expects bytes-like stream for PDFs
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                doc = fitz.open(self.file_path)

            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                try:
                    text = page.get_text()
                except Exception:
                    text = ""
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"

            try:
                doc.close()
            except Exception:
                pass

            return full_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def _extract_text_from_image(self) -> str:
        """Extract text from image using pytesseract.

        Read the uploaded file safely into a BytesIO to avoid issues with
        file-like wrappers that don't behave like regular file objects.
        """
        try:
            if self.file_obj:
                try:
                    self.file_obj.seek(0)
                except Exception:
                    pass
                raw = self.file_obj.read()
                if not raw:
                    raise ValueError("Uploaded image file appears empty after reading")
                img = Image.open(BytesIO(raw))
            else:
                img = Image.open(self.file_path)

            # Enhance image for better OCR
            img = img.convert('RGB')

            # Use pytesseract to extract text
            text = pytesseract.image_to_string(img, lang='eng')
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

    def extract_invoice_data(self) -> dict:
        """
        Extract structured invoice data from the document.
        
        Returns:
            dict with keys: customer_name, customer_phone, customer_address,
                           reference, items, subtotal, tax_amount, total_amount
        """
        text = self.extract_text()
        if not text:
            return {}

        # Clean and normalize text
        text_lower = text.lower()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        extracted_data = {
            'customer_name': self._extract_customer_name(text_lower, lines),
            'customer_phone': self._extract_customer_phone(text_lower, lines),
            'customer_address': self._extract_customer_address(text_lower, lines),
            'reference': self._extract_reference(text_lower, lines),
            'items': self._extract_items(text, lines),
            'subtotal': self._extract_amount(text_lower, 'subtotal|net value|net'),
            'tax_amount': self._extract_amount(text_lower, 'vat|tax|sales tax'),
            'total_amount': self._extract_amount(text_lower, r'(total|gross value)\s*:', strict=True),
        }

        return {k: v for k, v in extracted_data.items() if v is not None}

    def _extract_customer_name(self, text_lower: str, lines: list) -> str:
        """Extract customer name from text."""
        for keyword in self.KEYWORDS_CUSTOMER_NAME:
            idx = text_lower.find(keyword)
            if idx != -1:
                # Find the line containing this keyword
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        # Try next line or extract from current line
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line and len(next_line) > 3:
                                return next_line
                        # Also check if the name is after the colon
                        if ':' in line:
                            name = line.split(':', 1)[1].strip()
                            if name and len(name) > 3:
                                return name

        # Fallback: look for capitalized names in first 10 lines
        for line in lines[:10]:
            words = line.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if len(w) > 1):
                return line
        
        return None

    def _extract_customer_phone(self, text_lower: str, lines: list) -> str:
        """Extract customer phone number from text."""
        import re

        for keyword in self.KEYWORDS_CUSTOMER_PHONE:
            for line in lines:
                if keyword in line.lower():
                    # Extract phone number pattern
                    phone_pattern = r'(\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9})'
                    match = re.search(phone_pattern, line)
                    if match:
                        return match.group(1).strip()

        return None

    def _extract_customer_address(self, text_lower: str, lines: list) -> str:
        """Extract customer address from text."""
        for keyword in self.KEYWORDS_CUSTOMER_ADDRESS:
            for i, line in enumerate(lines):
                if keyword in line.lower():
                    # Collect next 2-3 lines as address
                    address_lines = []
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not any(kw in next_line.lower() for kw in ['tel', 'phone', 'email']):
                            address_lines.append(next_line)
                    if address_lines:
                        return ' '.join(address_lines)

        return None

    def _extract_reference(self, text_lower: str, lines: list) -> str:
        """Extract reference/PO number from text."""
        import re

        for keyword in self.KEYWORDS_REFERENCE:
            for line in lines:
                if keyword in line.lower():
                    # Extract reference number/code after colon or keyword
                    if ':' in line:
                        ref = line.split(':', 1)[1].strip()
                        if ref and len(ref) < 50:
                            return ref
                    # Extract alphanumeric code after keyword
                    pattern = f"{keyword}[:\\s]+([A-Z0-9-]+)"
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1)

        return None

    def _extract_items(self, text: str, lines: list) -> list:
        """Extract line items from invoice."""
        import re

        items = []
        
        # Look for table-like structure with qty and price columns
        for i, line in enumerate(lines):
            # Pattern: item code, description, quantity, unit, price, total
            item_pattern = r'(\d+)\s+(.+?)\s+(\d+(?:\.\d+)?)\s+(NOS|PCS|UNT|HR|M|KG|L)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s+(\d+(?:,\d{3})*(?:\.\d+)?)'
            match = re.search(item_pattern, line, re.IGNORECASE)
            
            if match:
                items.append({
                    'description': match.group(2).strip(),
                    'quantity': float(match.group(3)),
                    'unit': match.group(4).upper(),
                    'unit_price': self._parse_amount(match.group(5)),
                    'total': self._parse_amount(match.group(6))
                })
        
        # Fallback: if no items found, create a simple item with just description
        if not items:
            for line in lines:
                if line and len(line) > 10 and not any(kw in line.lower() for kw in ['customer', 'date', 'invoice', 'total', 'vat']):
                    # Check if line looks like an item description
                    if re.search(r'[A-Za-z]{5,}', line):
                        items.append({
                            'description': line[:100],
                            'quantity': 1,
                            'unit': 'NOS'
                        })

        return items if items else None

    def _extract_amount(self, text_lower: str, keyword_pattern: str, strict=False) -> Decimal:
        """
        Extract currency amount from text.
        
        Args:
            text_lower: Lowercase text to search in
            keyword_pattern: Regex pattern for the keyword
            strict: If True, only match if keyword appears right before amount
        """
        import re

        # Pattern for amounts: handles 100,000.00 or 100000.00 format
        amount_pattern = r'(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?'
        
        # Look for keyword followed by amount
        combined_pattern = f"{keyword_pattern}[:\\s]+{amount_pattern}"
        match = re.search(combined_pattern, text_lower)
        
        if match:
            amount_str = match.group(1) if ',' not in match.group(1) else match.group(1)
            try:
                # Remove commas and convert to Decimal
                amount_clean = amount_str.replace(',', '')
                return Decimal(amount_clean)
            except:
                pass

        return None

    def _parse_amount(self, amount_str: str) -> Decimal:
        """Parse amount string to Decimal."""
        try:
            clean = amount_str.replace(',', '').strip()
            return Decimal(clean) if clean else None
        except:
            return None


def process_uploaded_invoice_file(uploaded_file) -> dict:
    """
    Process an uploaded invoice file and extract data.

    Args:
        uploaded_file: Django InMemoryUploadedFile or TemporaryUploadedFile

    Returns:
        dict with extracted invoice data
    """
    if not uploaded_file:
        return {
            'success': False,
            'error': 'No uploaded file provided',
            'data': {}
        }

    try:
        # Defensive: ensure uploaded_file has a name attribute for extension detection
        if not getattr(uploaded_file, 'name', None):
            return {'success': False, 'error': 'Uploaded file missing name', 'data': {}}

        extractor = InvoiceDataExtractor(file_obj=uploaded_file)
        data = extractor.extract_invoice_data() or {}

        return {
            'success': True,
            'data': data
        }
    except Exception as e:
        logger.error(f"Error processing invoice file: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'data': {}
        }
