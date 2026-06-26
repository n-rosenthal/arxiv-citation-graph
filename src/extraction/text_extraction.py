# ============================================================================
# EXTRAÇÃO DE TEXTO — PDF -> .txt via PyMuPDF (fitz)
# ============================================================================
from pathlib import Path

import fitz


def extract_text_from_pdf(pdf_path: Path, text_path: Path) -> str:
    """Extrai o texto de um PDF e salva em text_path. Retorna o texto extraído.

    Não toca no banco de dados — apenas processa arquivos. A atualização do
    status `text_extracted` e o enfileiramento de `extract_refs` ficam a
    cargo do worker (ex: src.extraction.worker.extract_worker)."""
    doc = fitz.open(pdf_path)
    try:
        text = ""
        for page in doc:
            text += page.get_text()
    finally:
        doc.close()

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)

    return text


def text_already_extracted(text_path: Path, min_size: int = 100) -> bool:
    """Verifica se o texto já foi extraído em uma sessão anterior (idempotência)."""
    return text_path.exists() and text_path.stat().st_size > min_size


def load_text(text_path: Path) -> str:
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()