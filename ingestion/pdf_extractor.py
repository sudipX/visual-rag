import pdfplumber
import io
from PIL import Image
from pathlib import Path

def extract_text_by_page(pdf_path : str) -> list:

    # extracting all text from a PDF, returning one entry per page

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_index, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text is None or text.strip() == "":
                print(f"Page {page_index+1} :  No text found.")
                continue
            
            pages.append({
                "page_number" : page_index+1, 
                "text" : text.strip()
            })

    print(f"Extracted text from {len(pages)} pages.")
    return pages

def extract_image_by_page(pdf_path:str, output_dir:str) ->list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)

    pdf_stem = Path(pdf_path).stem

    extracted = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1

            if not page.images:
                continue

            for img_index, img_obj in enumerate(page.images):
                try:
                    raw_bytes = img_obj["stream"].get_data()
                    image = Image.open(io.BytesIO(raw_bytes))
                    image = image.convert("RGB")

                    file_name = f"{pdf_stem}_{page_num}_img{img_index}.png"
                    save_path = output_path/file_name

                    image.save(save_path)

                    extracted.append({
                        "page_number" : page_num,
                        "image_path" : str(save_path),
                        "image_index" : img_index
                    })

                    print(f" Extracted : {file_name}")

                except Exception as e:
                    print(f"Skipped image on page {page_num} : {e}")
                    continue

    print(f"Extracted {len(extracted)} images total.")
    return extracted
                
