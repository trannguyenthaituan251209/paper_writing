import pypdf

pdf_path = r"c:\Users\PC\Desktop\SMS Dataset Writing\29-07-2026 Scopy Paper.pdf"
output_txt_path = r"c:\Users\PC\Desktop\SMS Dataset Writing\extracted_pdf_29_07_2026.txt"

reader = pypdf.PdfReader(pdf_path)
full_text = []

for i, page in enumerate(reader.pages):
    full_text.append(f"--- PAGE {i+1} ---")
    page_text = page.extract_text()
    if page_text:
        full_text.append(page_text)
    full_text.append("\n")

final_content = "\n".join(full_text)

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(final_content)

print(f"Extracted {len(reader.pages)} pages to {output_txt_path}")
