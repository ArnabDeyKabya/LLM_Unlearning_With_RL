try:
    import PyPDF2
    print("PyPDF2 imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    import sys
    sys.exit(1)

import sys
import os

def extract_pdf_text(pdf_path):
    text = ""
    try:
        if not os.path.exists(pdf_path):
            return f"Error: File not found at {pdf_path}"
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"Found {num_pages} pages in PDF")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text
                text += "\n\n"
        
        return text
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    pdf_path = r"f:\Unlearning\New-method.pdf"
    print(f"Attempting to extract from: {pdf_path}")
    extracted_text = extract_pdf_text(pdf_path)
    print(extracted_text)
