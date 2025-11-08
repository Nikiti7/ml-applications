import os
from pathlib import Path
from docx import Document
import PyPDF2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DOCS_SRC = ROOT.parent / "docs"  # ../docs/
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)


def read_txt(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def read_md(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx(path):
    doc = Document(path)
    full = []
    for p in doc.paragraphs:
        full.append(p.text)
    return "\n".join(full)


def read_pdf(path):
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                txt = page.extract_text()
            except Exception:
                txt = ""
            if txt:
                text.append(txt)
    return "\n".join(text)


def process_file(path: Path):
    if path.suffix.lower() in [".txt"]:
        return read_txt(path)
    if path.suffix.lower() in [".md", ".markdown"]:
        return read_md(path)
    if path.suffix.lower() in [".docx"]:
        return read_docx(path)
    if path.suffix.lower() in [".pdf"]:
        return read_pdf(path)
    return None


def main():
    files = list(DOCS_SRC.rglob("*"))
    processed = 0
    for f in tqdm(files):
        if f.is_file() and f.suffix.lower() in [".txt", ".md", ".docx", ".pdf"]:
            text = process_file(f)
            if text and text.strip():
                outname = f.name + ".txt"
                outpath = OUT_DIR / outname
                outpath.write_text(text, encoding="utf-8", errors="ignore")
                processed += 1
    print(f"Processed {processed} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
