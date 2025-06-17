import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import pickle

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            if text.strip():
                return text
    except Exception:
        pass

    # OCR fallback for scanned PDFs
    images = convert_from_path(pdf_path)
    return "\n".join(pytesseract.image_to_string(img) for img in images)

def process_pdf_folder(folder_path, save_path="extracted_text.pkl"):
    folder_path = Path(folder_path)
    all_texts = []
    for pdf_file in folder_path.glob("*.pdf"):
        print(f"📄 Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        all_texts.append(text.strip())
    with open(save_path, "wb") as f:
        pickle.dump(all_texts, f)
    print(f"✅ Extracted {len(all_texts)} documents and saved to {save_path}")

if __name__ == "__main__":
    process_pdf_folder("data/raw")
