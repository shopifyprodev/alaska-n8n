#!/usr/bin/env python3
"""
Test script to verify page-by-page processing improvements.
"""

import requests
import time
import json

def test_page_processing():
    """Test the page-by-page processing with a sample PDF."""
    print("🧪 Testing Page-by-Page Processing")
    print("=" * 50)
    
    # Test URL (replace with your actual test URL)
    test_url = "https://drive.google.com/file/d/1uHzmH9iyoIL96UY1E6ZPO_DLUKWj-P4J/view?usp=sharing"
    
    print(f"📄 Testing with URL: {test_url}")
    print("🔄 Starting extraction with AI cleaning...")
    
    start_time = time.time()
    
    try:
        response = requests.get(f'http://localhost:5000/extract-text?pdf_url={test_url}&use_ai_cleaning=true')
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            processing_time = end_time - start_time
            
            print(f"✅ Extraction successful in {processing_time:.2f}s")
            print(f"📊 Text length: {len(data['text'])} characters")
            print(f"🤖 AI cleaning used: {data.get('ai_cleaning_used', 'N/A')}")
            
            if data.get('processing_info'):
                info = data['processing_info']
                print(f"📄 Pages processed: {info.get('pages_processed', 'N/A')}")
                print(f"⏱️ Timeout per page: {info.get('timeout_per_page', 'N/A')}s")
                print(f"🔧 Method: {info.get('method', 'N/A')}")
            
            # Show first 500 characters of result
            preview = data['text'][:500] + "..." if len(data['text']) > 500 else data['text']
            print(f"\n📝 Preview:\n{preview}")
            
            return True
        else:
            print(f"❌ Extraction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return False

def test_without_ai_cleaning():
    """Test without AI cleaning for comparison."""
    print("\n🧪 Testing without AI cleaning")
    print("=" * 30)
    
    test_url = "https://drive.google.com/file/d/1uHzmH9iyoIL96UY1E6ZPO_DLUKWj-P4J/view?usp=sharing"
    
    start_time = time.time()
    
    try:
        response = requests.get(f'http://localhost:5000/extract-text?pdf_url={test_url}&use_ai_cleaning=false')
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            processing_time = end_time - start_time
            
            print(f"✅ Extraction successful in {processing_time:.2f}s")
            print(f"📊 Text length: {len(data['text'])} characters")
            print(f"🤖 AI cleaning used: {data.get('ai_cleaning_used', 'N/A')}")
            
            return True
        else:
            print(f"❌ Extraction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Page-by-Page Processing Tests")
    print("=" * 60)
    
    # Test with AI cleaning (page-by-page)
    success1 = test_page_processing()
    
    # Test without AI cleaning (for comparison)
    success2 = test_without_ai_cleaning()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests completed successfully!")
        print("📊 Page-by-page processing is working correctly.")
    else:
        print("❌ Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main() 