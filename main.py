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
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError

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
        print("OCR (Tesseract) is available")
    except Exception as e:
        print(f"Warning: Tesseract not found on system: {e}")
        print("OCR will be disabled. Install tesseract-ocr package on your system.")
        OCR_AVAILABLE = False
        
except ImportError as e:
    print(f"Warning: OCR dependencies not available: {e}")
    print("Install with: pip install pytesseract pillow PyMuPDF")
    OCR_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)


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
        
        if verbose:
            print(f"Extracted file ID: {file_id}")
        
        # Create direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        if verbose:
            print(f"Downloading from: {download_url}")
        
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
        
        if verbose:
            print(f"Downloaded to temporary file: {temp_path}")
        
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
        
        # Configure OCR for better text extraction
        # Try different PSM modes for better results
        psm_modes = [6, 8, 3]  # 6=uniform block, 8=word, 3=auto
        best_text = ""
        
        for psm in psm_modes:
            try:
                custom_config = f'--oem 3 --psm {psm}'
                text = pytesseract.image_to_string(image, config=custom_config)
                
                if text.strip() and len(text.strip()) > len(best_text):
                    best_text = text.strip()
            except Exception as e:
                print(f"OCR failed with PSM {psm}: {e}")
                continue
        
        text = best_text
        
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
            print(f"Detected Google Drive link: {pdf_source}")
            
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
        print(f"Reading PDF: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        print(f"PDF has {total_pages} pages")
        
        # Extract text from all pages
        all_text = ""
        text_extracted = False
        
        for page_num, page in enumerate(reader.pages):
            print(f"Processing page {page_num + 1}/{total_pages}...")
            
            # Try to extract text normally first
            text = page.extract_text()
            
            # If no text found or very little text, try OCR on the entire page
            if (not text.strip() or len(text.strip()) < 100) and use_ocr and OCR_AVAILABLE:
                print(f"  Limited text found, trying OCR for entire page {page_num + 1}...")
                
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
                        print(f"  OCR extracted text from entire page {page_num + 1}")
                    else:
                        print(f"  OCR did not find additional text on page {page_num + 1}")
                    
                    pix = None  # Free memory
                    doc.close()
                    
                except Exception as e:
                    print(f"    Warning: Could not process page {page_num + 1} with OCR: {e}")
                    
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
                                print(f"    Warning: Could not process image {img_index}: {e}")
                        
                        doc.close()
                        
                        if ocr_text:
                            text = ocr_text
                            print(f"  OCR extracted text from images on page {page_num + 1}")
                    except Exception as e:
                        print(f"    Warning: Could not extract images from page {page_num + 1}: {e}")
            
            if text.strip():  # Only add non-empty pages
                text_extracted = True
                all_text += f"\n{'='*50}\n"
                all_text += f"PAGE {page_num + 1} of {total_pages}\n"
                all_text += f"{'='*50}\n\n"
                all_text += text
                all_text += "\n\n"
            else:
                print(f"  No text found on page {page_num + 1}")
        
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
        
        # Save extracted text to a file for easier viewing
        try:
            output_dir = Path("text files")
            output_dir.mkdir(exist_ok=True)
            
            # Create filename based on source
            if is_google_drive_link(pdf_source):
                file_id = extract_file_id_from_drive_link(pdf_source)
                filename = f"drive_file_{file_id}_extracted_text.txt"
            else:
                source_name = Path(pdf_source).stem
                filename = f"{source_name}_extracted_text.txt"
            
            output_path = output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            print(f"Extracted text saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save extracted text to file: {e}")
        
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
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")


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
        'flask_available': FLASK_AVAILABLE,
        'requests_available': REQUESTS_AVAILABLE
    })


@app.route('/extract-text')
def extract_text():
    """API endpoint to extract text from PDF."""
    pdf_url = request.args.get('pdf_url')
    
    if not pdf_url:
        return jsonify({'error': 'No PDF URL provided'}), 400
    
    # Log OCR availability
    print(f"OCR Available: {OCR_AVAILABLE}")
    
    # Extract text from PDF
    success, text, message = extract_text_from_pdf_web(pdf_url, use_ocr=True)
    
    if success:
        return jsonify({'text': text, 'message': message, 'ocr_available': OCR_AVAILABLE})
    else:
        return jsonify({'error': message, 'ocr_available': OCR_AVAILABLE}), 400


def main():
    """Main function to run the Flask web application locally."""
    print("Starting PDF Text Extractor Web Application...")
    print(f"OCR Available: {OCR_AVAILABLE}")
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