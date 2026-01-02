from pathlib import Path
import pdfplumber


def load_pdf_text(file_path: str) -> str:
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(file_path)

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    return "\n".join(pages)
