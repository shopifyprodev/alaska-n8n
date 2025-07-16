#!/usr/bin/env python3
"""
Test script to verify OCR functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import OCR_AVAILABLE, extract_text_from_image
from PIL import Image
import io

def test_ocr():
    """Test OCR functionality"""
    print("Testing OCR functionality...")
    print(f"OCR Available: {OCR_AVAILABLE}")
    
    if not OCR_AVAILABLE:
        print("OCR is not available. Please install tesseract-ocr package.")
        return False
    
    # Create a simple test image with text
    try:
        # Create a simple image with text
        img = Image.new('RGB', (200, 50), color='white')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Test OCR
        result = extract_text_from_image(img_data)
        print(f"OCR Test Result: {result}")
        
        return True
    except Exception as e:
        print(f"OCR Test Failed: {e}")
        return False

if __name__ == "__main__":
    test_ocr() 