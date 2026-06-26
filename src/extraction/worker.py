# ============================================================================
# WORKER DE EXTRAÇÃO — orquestra text_extraction + reference_extraction
# com banco de dados e fila de processamento
# ============================================================================
import time

from src.db import Paper, Citation
from src.extraction.text_extraction import (
    extract_text_from_pdf, text_already_extracted, load_text
)
from src.extraction.reference_extraction import (
    extract_reference_ids, split_known_unknown
)


def extract_worker(collector):
    """Consome tarefas 'extract_text' e 'extract_refs' da fila do collector.

    Recebe um ContinuousCollector (ou objeto com mesma interface: SessionLocal,
    pdf_dir, text_dir, get_next_task, add_to_queue, complete_task,
    fetch_papers_by_ids)."""
    print("📄 Worker de extração iniciado")
    while collector.running:
        try:
            task_info = collector.get_next_task('extract_text')
            if not task_info:
                task_info = collector.get_next_task('extract_refs')
            if not task_info:
                time.sleep(5)
                continue

            task_id = task_info['id']
            paper_id = task_info['paper_id']
            task_type = task_info['task_type']
            print(f"\n📄 Processando: {paper_id} ({task_type})")

            session = collector.SessionLocal()
            try:
                paper = session.get(Paper, paper_id)
                if not paper or not paper.pdf_downloaded:
                    collector.complete_task(task_id, False, "PDF não disponível")
                    continue

                pdf_path = collector.pdf_dir / f"{paper_id}.pdf"

                if task_type == 'extract_text':
                    _handle_extract_text(collector, session, paper, pdf_path, task_id)
                elif task_type == 'extract_refs':
                    _handle_extract_refs(collector, session, paper, task_id)

            except Exception as e:
                session.rollback()
                print(f"   ❌ Erro na extração: {e}")
                collector.complete_task(task_id, False, str(e))
            finally:
                session.close()
        except Exception as e:
            print(f"   ❌ Erro inesperado no worker de extração: {e}")
            time.sleep(5)


def _handle_extract_text(collector, session, paper, pdf_path, task_id):
    text_path = collector.text_dir / f"{paper.id}.txt"

    if text_already_extracted(text_path):
        # Texto já extraído em sessão anterior
        if not paper.text_extracted:
            paper.text_extracted = True
            session.commit()
        print(f"   ♻️  Texto já existe: {paper.id}")
    else:
        text = extract_text_from_pdf(pdf_path, text_path)
        paper.text_extracted = True
        session.commit()
        print(f"   ✓ Texto extraído ({len(text)} caracteres)")

    collector.add_to_queue(paper.id, 'extract_refs', priority=3)
    collector.complete_task(task_id, True)


def _handle_extract_refs(collector, session, paper, task_id):
    text_path = collector.text_dir / f"{paper.id}.txt"
    if not text_path.exists():
        raise Exception("Texto não encontrado")

    text = load_text(text_path)

    # 1) Extrai referências do texto (restrito à seção de referências quando possível)
    text_references = extract_reference_ids(text, paper.id, use_section=True)

    # 2) Enriquece com Semantic Scholar (se habilitado)
    ss_references = collector.fetch_citations_from_semantic_scholar(paper.id)

    # 3) Une as duas fontes — sem duplicatas, mantendo rastreabilidade de origem
    text_ref_set = set(text_references)
    ss_ref_set = set(ss_references)
    all_references = list(text_ref_set | ss_ref_set)

    existing_ids = {row[0] for row in session.query(Paper.id).all()}
    valid_refs, new_refs = split_known_unknown(all_references, existing_ids)

    paper.num_references = len(all_references)
    for ref_id in valid_refs:
        in_text = ref_id in text_ref_set
        in_ss = ref_id in ss_ref_set
        # Confiança: 0.95 se encontrado em ambas as fontes, 0.9 só no texto, 0.7 só no S2
        if in_text and in_ss:
            confidence = 0.95
        elif in_text:
            confidence = 0.9
        else:
            confidence = 0.7
        citation = Citation(
            source_id=paper.id, target_id=ref_id,
            found_in_text=in_text, confidence=confidence,
        )
        session.add(citation)

    paper.references_extracted = True
    paper.num_citations_in_graph = len(valid_refs)
    session.commit()

    print(
        f"   ✓ Referências: {len(all_references)} total "
        f"(texto: {len(text_references)}, S2: {len(ss_references)}), "
        f"{len(valid_refs)} no grafo"
    )

    if new_refs:
        print(f"   🔍 {len(new_refs)} novas referências identificadas")
        collector.fetch_papers_by_ids(new_refs)

    collector.complete_task(task_id, True)