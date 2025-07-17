#!/usr/bin/env python3
"""
PDF Text Extractor with OCR - Web Application
Extracts text from PDF files (including images) and provides a web interface.
Supports both local files and Google Drive links.
"""

import sys
import os
import argparse
import re
import tempfile
import time
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import google.generativeai as genai

# Flask imports
try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Flask library not available: {e}")
    print("Install with: pip install flask")
    FLASK_AVAILABLE = False

# HTTP requests for Google Drive downloads
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: requests library not available: {e}")
    print("Install with: pip install requests")
    REQUESTS_AVAILABLE = False

# Initialize OCR variables
OCR_AVAILABLE = False
OCR_TYPE = None

# OCR imports
try:
    import pytesseract
    from PIL import Image
    import io
    import fitz  # PyMuPDF for image extraction
    
    # Test if tesseract is actually available on the system
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
        OCR_TYPE = "tesseract"
        print("OCR (Tesseract) is available")
    except Exception as e:
        print(f"Warning: Tesseract not found on system: {e}")
        OCR_AVAILABLE = False
        OCR_TYPE = None
        
except ImportError as e:
    print(f"Warning: Tesseract dependencies not available: {e}")
    OCR_AVAILABLE = False
    OCR_TYPE = None

# Try EasyOCR as fallback
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("EasyOCR is available as fallback")
except ImportError as e:
    print(f"Warning: EasyOCR not available: {e}")
    EASYOCR_AVAILABLE = False

# Set final OCR availability
if not OCR_AVAILABLE and EASYOCR_AVAILABLE:
    OCR_AVAILABLE = True
    OCR_TYPE = "easyocr"
    print("Using EasyOCR for text extraction")
elif not OCR_AVAILABLE:
    print("No OCR available. Install tesseract-ocr or easyocr package.")


# genai.configure(api_key="AIzaSyDqeJFw51iaX8Wht89HUO-ELq5Kcu9y8mQ")
genai.configure(api_key="AIzaSyCU8TEdllSZyI_jytijYqfBgXIrCkDj2mk")
model = genai.GenerativeModel("gemini-1.5-flash")
executor = ThreadPoolExecutor(max_workers=2)

# Configuration for AI processing
AI_TIMEOUT = 30  # seconds
MAX_TEXT_LENGTH = 50000  # characters per chunk
AI_RETRY_ATTEMPTS = 3

# Initialize Flask app
app = Flask(__name__)


def chunk_text(text, max_length=MAX_TEXT_LENGTH):
    """
    Split large text into smaller chunks for AI processing.
    
    Args:
        text (str): Text to chunk
        max_length (int): Maximum length per chunk
    
    Returns:
        list: List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by pages first
    pages = text.split("PAGE ")
    
    for i, page in enumerate(pages):
        if i == 0:  # First part (before first PAGE)
            if page.strip():
                current_chunk += page
        else:
            page_text = "PAGE " + page
            
            if len(current_chunk + page_text) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = page_text
            else:
                current_chunk += page_text
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def get_page_count(text):
    """Get the number of pages in the text."""
    pages = text.split("PAGE ")
    return len(pages) - 1 if len(pages) > 1 else 0

def clean_text_page_by_page(text, timeout=AI_TIMEOUT):
    """
    Clean text page by page and save to temporary file.
    
    Args:
        text (str): Text to clean
        timeout (int): Timeout in seconds per page
    
    Returns:
        str: Path to temporary file with cleaned text
    """
    def clean_single_page(page_text, page_num):
        """Clean a single page of text."""
        try:
            rules = """
            You are a document cleanup assistant. Process this single page with these rules:
            
            1. If the page has meaningful, legible text, preserve it exactly
            2. If the page is mostly an image, map, or unreadable OCR, replace with "[ here is image ]"
            3. Keep the page header format: "PAGE X of Y"
            4. Return only the cleaned text for this page, no explanations
            """

            prompt = f"""
            Clean this single page of PDF text according to the rules:
            
            {rules}
            
            Page to clean:
            {page_text}
            """

            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"AI cleaning failed for page {page_num}: {e}")
            return page_text.strip()
    
    try:
        # Create temporary file for cleaned text
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        temp_path = temp_file.name
        temp_file.close()
        
        # Split text into pages
        pages = text.split("PAGE ")
        cleaned_pages = []
        
        # Process header (text before first PAGE)
        if pages[0].strip():
            header_text = pages[0].strip()
            print(f"Processing header...")
            future = executor.submit(lambda: header_text)  # Keep header as-is
            try:
                result = future.result(timeout=timeout)
                cleaned_pages.append(result)
                print(f"✅ Header processed")
            except FutureTimeoutError:
                print(f"⚠️ Header processing timed out, keeping original")
                cleaned_pages.append(header_text)
        
        # Get total page count for progress reporting
        total_pages = len(pages) - 1 if len(pages) > 1 else 0
        print(f"📄 Processing {total_pages} pages with {timeout}s timeout per page")
        
        # Process each page
        for i, page in enumerate(pages[1:], 1):
            page_text = "PAGE " + page
            print(f"🔄 Processing page {i}/{total_pages}...")
            
            future = executor.submit(clean_single_page, page_text, i)
            try:
                result = future.result(timeout=timeout)
                cleaned_pages.append(result)
                print(f"✅ Page {i}/{total_pages} cleaned and added")
            except FutureTimeoutError:
                print(f"⚠️ Page {i}/{total_pages} processing timed out, keeping original")
                cleaned_pages.append(page_text.strip())
        
        # Write all cleaned pages to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(cleaned_pages))
        
        print(f"📄 All pages processed and saved to temporary file")
        
        # Show processing statistics
        successful_pages = sum(1 for page in cleaned_pages if page.strip())
        print(f"📊 Processing complete: {successful_pages}/{total_pages} pages successfully processed")
        
        return temp_path
        
    except Exception as e:
        print(f"AI cleaning failed: {e}")
        # Return original text as temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        temp_path = temp_file.name
        temp_file.write(text)
        temp_file.close()
        return temp_path


def clean_text_with_retry(text, max_retries=AI_RETRY_ATTEMPTS):
    """
    Clean text with retry mechanism for reliability.
    
    Args:
        text (str): Text to clean
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        str: Path to temporary file with cleaned text
    """
    for attempt in range(max_retries):
        try:
            print(f"AI cleaning attempt {attempt + 1}/{max_retries}")
            temp_file_path = clean_text_page_by_page(text)
            return temp_file_path
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("All AI cleaning attempts failed, returning original text as file")
                # Create temporary file with original text
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
                temp_path = temp_file.name
                temp_file.write(text)
                temp_file.close()
                return temp_path


def is_google_drive_link(url):
    """
    Check if the given URL is a Google Drive link.
    
    Args:
        url (str): URL to check
    
    Returns:
        bool: True if it's a Google Drive link, False otherwise
    """
    google_drive_patterns = [
        r'https://drive\.google\.com/file/d/',
        r'https://drive\.google\.com/open\?id=',
        r'https://docs\.google\.com/document/d/'
    ]
    
    for pattern in google_drive_patterns:
        if re.match(pattern, url):
            return True
    return False


def extract_file_id_from_drive_link(url):
    """
    Extract the file ID from a Google Drive link.
    
    Args:
        url (str): Google Drive URL
    
    Returns:
        str: File ID or None if not found
    """
    # Pattern for file/d/ID/view format
    file_pattern = r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)'
    match = re.search(file_pattern, url)
    if match:
        return match.group(1)
    
    # Pattern for open?id=ID format
    open_pattern = r'https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)'
    match = re.search(open_pattern, url)
    if match:
        return match.group(1)
    
    # Pattern for docs.google.com format
    docs_pattern = r'https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)'
    match = re.search(docs_pattern, url)
    if match:
        return match.group(1)
    
    return None


def download_pdf_from_drive(drive_url, verbose=True):
    """
    Download a PDF file from Google Drive.
    
    Args:
        drive_url (str): Google Drive URL
        verbose (bool): Whether to print progress information
    
    Returns:
        str: Path to downloaded temporary file or None if failed
    """
    if not REQUESTS_AVAILABLE:
        print("Error: requests library is required for Google Drive downloads.")
        print("Install with: pip install requests")
        return None
    
    try:
        # Extract file ID from URL
        file_id = extract_file_id_from_drive_link(drive_url)
        if not file_id:
            print(f"Error: Could not extract file ID from URL: {drive_url}")
            return None
        
        # Create direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        
        # Write content to temporary file
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        
        temp_file.close()
        
        return temp_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from Google Drive: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return None


def extract_text_from_image(image_data):
    """
    Extract text from an image using OCR.
    
    Args:
        image_data (bytes): Image data as bytes
    
    Returns:
        str: Extracted text from the image
    """
    if not OCR_AVAILABLE:
        print("OCR not available - skipping image text extraction")
        return ""
        
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        if OCR_TYPE == "tesseract":
            # Use Tesseract OCR
            psm_modes = [6, 8, 3]  # 6=uniform block, 8=word, 3=auto
            best_text = ""
            
            for psm in psm_modes:
                try:
                    custom_config = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(image, config=custom_config)
                    
                    if text.strip() and len(text.strip()) > len(best_text):
                        best_text = text.strip()
                except Exception as e:
                    print(f"Tesseract OCR failed with PSM {psm}: {e}")
                    continue
            
            text = best_text
            
        elif OCR_TYPE == "easyocr":
            # Use EasyOCR
            try:
                reader = easyocr.Reader(['en'])
                results = reader.readtext(image)
                text = ' '.join([result[1] for result in results])
            except Exception as e:
                print(f"EasyOCR failed: {e}")
                text = ""
        else:
            text = ""
        
        # Clean up the extracted text
        if text:
            # Remove excessive whitespace and normalize
            text = ' '.join(text.split())
            return text.strip()
        
        return ""
    except Exception as e:
        print(f"Warning: Could not extract text from image: {e}")
        return ""


def extract_images_from_pdf(pdf_path):
    """
    Extract images from PDF pages.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        list: List of tuples (page_num, image_data)
    """
    images = []
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get images from the page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append((page_num + 1, img_data))
                    
                    pix = None  # Free memory
                except Exception as e:
                    print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        return images
    except Exception as e:
        print(f"Warning: Could not extract images from PDF: {e}")
        return []


def extract_text_from_pdf_web(pdf_source, use_ocr=True):
    """
    Extract text from a PDF file and return it as a string.
    Web version of the extraction function.
    
    Args:
        pdf_source (str): Path to the PDF file or Google Drive URL
        use_ocr (bool): Whether to use OCR for images
    
    Returns:
        tuple: (success: bool, text: str, message: str)
    """
    temp_file_path = None
    
    try:
        # Check if it's a Google Drive link
        if is_google_drive_link(pdf_source):
            # Download the PDF from Google Drive
            temp_file_path = download_pdf_from_drive(pdf_source, verbose=False)
            if not temp_file_path:
                return False, "", "Failed to download PDF from Google Drive."
            
            pdf_path = temp_file_path
            source_info = f"Google Drive: {pdf_source}"
        else:
            # Local file
            pdf_path = pdf_source
            source_info = pdf_source
            
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                return False, "", f"PDF file '{pdf_path}' not found."
        
        # Read PDF
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        # Extract text from all pages
        all_text = ""
        text_extracted = False
        
        for page_num, page in enumerate(reader.pages):
            # Try to extract text normally first
            text = page.extract_text()
            
            # If no text found or very little text, try OCR on the entire page
            if (not text.strip() or len(text.strip()) < 100) and use_ocr and OCR_AVAILABLE:
                
                # Convert the entire page to an image and apply OCR
                try:
                    doc = fitz.open(pdf_path)
                    pdf_page = doc[page_num]
                    
                    # Get page dimensions
                    rect = pdf_page.rect
                    zoom = 2  # Increase resolution for better OCR
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Render page to image
                    pix = pdf_page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Apply OCR to the entire page image
                    ocr_result = extract_text_from_image(img_data)
                    
                    if ocr_result and len(ocr_result) > len(text.strip()):
                        text = ocr_result
                    
                    pix = None  # Free memory
                    doc.close()
                    
                except Exception as e:
                    # Fallback: try to extract individual images from the page
                    try:
                        doc = fitz.open(pdf_path)
                        pdf_page = doc[page_num]
                        image_list = pdf_page.get_images()
                        
                        ocr_text = ""
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                pix = fitz.Pixmap(doc, xref)
                                
                                if pix.n - pix.alpha < 4:  # GRAY or RGB
                                    img_data = pix.tobytes("png")
                                    ocr_result = extract_text_from_image(img_data)
                                    if ocr_result:
                                        ocr_text += f"\n[Image {img_index + 1} OCR Result]:\n{ocr_result}\n"
                                
                                pix = None
                            except Exception as e:
                                pass
                        
                        doc.close()
                        
                        if ocr_text:
                            text = ocr_text
                    except Exception as e:
                        pass
            
            if text.strip():  # Only add non-empty pages
                text_extracted = True
                all_text += f"\n{'='*50}\n"
                all_text += f"PAGE {page_num + 1} of {total_pages}\n"
                all_text += f"{'='*50}\n\n"
                all_text += text
                all_text += "\n\n"

        
        if not text_extracted:
            message = "No text was extracted from any page."
            if not OCR_AVAILABLE and use_ocr:
                message += " Consider installing OCR dependencies: pip install pytesseract pillow PyMuPDF"
            return False, "", message
        
        # Create result text
        result_text = f"PDF Text Extraction Results\n"
        result_text += f"Source: {source_info}\n"
        result_text += f"Total Pages: {total_pages}\n"
        result_text += f"OCR Used: {use_ocr and OCR_AVAILABLE}\n"
        result_text += f"{'='*50}\n"
        result_text += all_text
        
        return True, result_text, f"Successfully extracted text from {total_pages} pages"
        
    except PdfReadError as e:
        return False, "", f"Error reading PDF: {e}"
    except Exception as e:
        return False, "", f"Unexpected error: {e}"
    finally:
        # Clean up temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                pass


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint to verify OCR status."""
    return jsonify({
        'status': 'healthy',
        'ocr_available': OCR_AVAILABLE,
        'ocr_type': OCR_TYPE,
        'easyocr_available': EASYOCR_AVAILABLE,
        'flask_available': FLASK_AVAILABLE,
        'requests_available': REQUESTS_AVAILABLE,
        'ai_config': {
            'timeout_seconds': AI_TIMEOUT,
            'max_text_length': MAX_TEXT_LENGTH,
            'retry_attempts': AI_RETRY_ATTEMPTS
        }
    })


def clean_text(text):
    """Clean the extracted text with robust error handling and timeout management."""
    print(f"Starting AI cleaning for text of length: {len(text)}")
    temp_file_path = clean_text_with_retry(text)
    
    # Read the cleaned text from temporary file
    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            cleaned_text = f.read()
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
        
        return cleaned_text
    except Exception as e:
        print(f"Error reading cleaned text from file: {e}")
        return text.strip()


@app.route('/extract-text')
def extract_text():
    """API endpoint to extract text from PDF."""
    pdf_url = request.args.get('pdf_url')
    use_ai_cleaning = True  # Always enabled
    
    if not pdf_url:
        return jsonify({'error': 'No PDF URL provided'}), 400
    
    print(f"Processing PDF: {pdf_url}")
    print(f"AI cleaning enabled: {use_ai_cleaning}")
    
    # Extract text from PDF
    success, text, message = extract_text_from_pdf_web(pdf_url, use_ocr=True)

    if success:
        if use_ai_cleaning:
            print("Starting AI cleaning process...")
            page_count = get_page_count(text)
            print(f"📊 Text contains {page_count} pages to process")
            cleaned_text = clean_text(text)
        else:
            print("Skipping AI cleaning, returning raw text")
            cleaned_text = text
        
        response_data = {
            'text': cleaned_text, 
            'message': message, 
            'ocr_available': OCR_AVAILABLE,
            'ocr_type': OCR_TYPE,
            'ai_cleaning_used': use_ai_cleaning
        }
        
        if use_ai_cleaning:
            response_data['processing_info'] = {
                'pages_processed': page_count,
                'timeout_per_page': AI_TIMEOUT,
                'method': 'page_by_page'
            }
        
        return jsonify(response_data)
    else:
        return jsonify({
            'error': message, 
            'ocr_available': OCR_AVAILABLE,
            'ocr_type': OCR_TYPE
        }), 400


def main():
    """Main function to run the Flask web application locally."""
    print("Starting PDF Text Extractor Web Application...")
    print(f"OCR Available: {OCR_AVAILABLE}")
    print(f"OCR Type: {OCR_TYPE}")
    print(f"EasyOCR Available: {EASYOCR_AVAILABLE}")
    print(f"Flask Available: {FLASK_AVAILABLE}")
    print(f"Requests Available: {REQUESTS_AVAILABLE}")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1


if __name__ == "__main__":
    main()  