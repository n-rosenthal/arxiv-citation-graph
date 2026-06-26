# ============================================================================
# EXTRAÇÃO DE REFERÊNCIAS — regex sobre o texto extraído do PDF
# ============================================================================
import re
from typing import Iterable, List, Optional, Set, Tuple

from src.config import REF_PATTERN, BARE_ID_PATTERN

# Padrões compilados a partir de config.py
# Captura IDs arXiv em todos os formatos comuns encontrados em PDFs:
# arxiv:2106.12345, arXiv:2106.12345v2, [2106.12345], https://arxiv.org/abs/2106.12345
_REF_RE = re.compile(REF_PATTERN, re.IGNORECASE)
# Fallback: qualquer sequência que pareça ID arXiv (YYMM.NNNNN) isolada
_BARE_ID_RE = re.compile(BARE_ID_PATTERN)
# Detecta início de seção de referências/bibliografia
_REFERENCES_SECTION_RE = re.compile(
    r'(?:^|\n)\s*(?:references|bibliography|works cited|literature)\s*\n',
    re.IGNORECASE
)


def normalize_line_breaks(text: str) -> str:
    """Remove quebras de linha que separam partes de um mesmo ID arXiv.
    Ex: '2106.\\n12345' -> '2106.12345'. Também normaliza 'arXiv :' -> 'arXiv:'."""
    text = re.sub(r'(\d{4})\.\s*\n\s*(\d{4,5})', r'\1.\2', text)
    text = re.sub(r'(arxiv|arXiv)[\s\n]*:[\s\n]*', 'arXiv:', text)
    return text


def extract_reference_section(text: str) -> Optional[str]:
    """Tenta extrair apenas a seção de referências para reduzir falsos positivos.
    Retorna o texto a partir do início da seção, ou o texto completo se não encontrada."""
    match = _REFERENCES_SECTION_RE.search(text)
    return text[match.start():] if match else text


def extract_reference_ids(text: str, paper_id: str, use_section: bool = True) -> List[str]:
    """Extrai IDs arXiv referenciados no texto, normalizados (sem versão 'vN')
    e excluindo o próprio paper. Pura — não toca no banco.

    Args:
        text: texto extraído do PDF.
        paper_id: ID do paper atual (ex: 2106.12345v1) — excluído do resultado.
        use_section: se True (padrão), tenta restringir a busca à seção de
            referências, reduzindo falsos positivos no corpo do artigo.
    """
    if use_section:
        text = extract_reference_section(text)
    text = normalize_line_breaks(text)

    found = set(_REF_RE.findall(text))
    found |= set(_BARE_ID_RE.findall(text))

    clean_paper_id = paper_id.split('v')[0]
    found.discard(clean_paper_id)

    # Normaliza: remove versão (v1, v2...) e descarta vazios
    references = list({m.split('v')[0] for m in found if m})
    return references


def split_known_unknown(references: Iterable[str], existing_ids: Set[str]) -> Tuple[List[str], List[str]]:
    """Separa referências entre as que já existem no banco (valid_refs, vão para o
    grafo de citações) e as que ainda não existem (new_refs, precisam ser buscadas
    no arXiv). Pura — recebe existing_ids já carregado pelo chamador."""
    valid_refs = [ref for ref in references if ref in existing_ids]
    new_refs = [ref for ref in references if ref not in existing_ids]
    return valid_refs, new_refs