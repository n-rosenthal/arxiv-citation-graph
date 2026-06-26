#!/usr/bin/env python3
# ============================================================================
# CLI — entrypoint único do projeto arXiv Citation Graph
# Uso: python cli.py {collect|process-local|train|evaluate|dashboard|api|all}
# ============================================================================
import argparse
import time
from datetime import datetime

from src.config import DB_URL, DATA_DIR
from src.collection.collector import ContinuousCollector
from src.models.train import train_models
from src.evaluation.compare_models import run_comparison, print_comparison_table

def cmd_cleanup_pdfs(args):
    """
    Remove PDFs já processados para economizar espaço.

    Critérios:
        - pdf_downloaded=True
        - text_extracted=True
        - references_extracted=True
        - TXT correspondente existe

    Também remove PDFs órfãos (arquivo existe mas paper não está no banco).

    Exemplos:

        python cli.py cleanup-pdfs --dry-run
        python cli.py cleanup-pdfs
    """

    from pathlib import Path

    from src.db import Paper, get_session_factory

    pdf_dir = Path(args.data_dir) / "pdfs"
    text_dir = Path(args.data_dir) / "texts"

    SessionLocal = get_session_factory(args.db)

    session = SessionLocal()

    try:

        papers = session.query(Paper).all()

        paper_ids = {
            p.id
            for p in papers
        }

        eligible = []
        orphan_pdfs = []

        bytes_recoverable = 0

        # --------------------------------------------------
        # PDFs ligados a papers
        # --------------------------------------------------

        for paper in papers:

            if not (
                paper.pdf_downloaded
                and paper.text_extracted
                and paper.references_extracted
            ):
                continue

            pdf_path = pdf_dir / f"{paper.id}.pdf"

            if not pdf_path.exists():
                continue

            txt_path = text_dir / f"{paper.id}.txt"

            if not txt_path.exists():
                continue

            size = pdf_path.stat().st_size

            eligible.append(
                (
                    paper,
                    pdf_path,
                    size,
                )
            )

            bytes_recoverable += size

        # --------------------------------------------------
        # PDFs órfãos
        # --------------------------------------------------

        for pdf_file in pdf_dir.glob("*.pdf"):

            paper_id = pdf_file.stem

            if paper_id not in paper_ids:

                size = pdf_file.stat().st_size

                orphan_pdfs.append(
                    (
                        pdf_file,
                        size,
                    )
                )

                bytes_recoverable += size

        # --------------------------------------------------
        # RELATÓRIO
        # --------------------------------------------------

        gb = bytes_recoverable / (1024 ** 3)

        print("\n🧹 Auditoria de PDFs")
        print("=" * 60)

        print(f"PDFs elegíveis      : {len(eligible):,}")
        print(f"PDFs órfãos         : {len(orphan_pdfs):,}")
        print(f"Espaço recuperável  : {gb:.2f} GB")

        if args.dry_run:
            print("\n🔎 Modo auditoria (--dry-run)")
            print("Nenhum arquivo removido.")
            return

        # --------------------------------------------------
        # REMOÇÃO
        # --------------------------------------------------

        removed = 0

        for paper, pdf_path, _ in eligible:

            try:

                pdf_path.unlink()

                paper.pdf_downloaded = False
                paper.pdf_size = 0

                removed += 1

            except Exception as e:

                print(
                    f"❌ Erro removendo "
                    f"{pdf_path.name}: {e}"
                )

        for pdf_path, _ in orphan_pdfs:

            try:

                pdf_path.unlink()

                removed += 1

            except Exception as e:

                print(
                    f"❌ Erro removendo "
                    f"{pdf_path.name}: {e}"
                )

        session.commit()

        print("\n✅ Limpeza concluída")
        print(f"Arquivos removidos : {removed:,}")
        print(f"Espaço liberado    : {gb:.2f} GB")

    finally:

        session.close()

def cmd_collect_year(args):
    """Coleta artigos de uma faixa de anos específica, ordenando pelos mais
    antigos primeiro — contorna o limite de 1000 resultados da API do arXiv
    que faz collect --days-back sempre retornar os mesmos papers recentes.

    Exemplo: python cli.py collect-year --from 2018 --to 2022
    Para cobrir histórico amplo: rode por faixa de 2 anos por vez."""
    collector = ContinuousCollector(db_url=args.db, data_dir=args.data_dir)
    collector.start_download_workers(num_download_workers=1)
    total = 0
    try:
        for year in range(args.year_from, args.year_to + 1):
            n = collector.search_by_year_range(year, year)
            total += n
            print(f"   Subtotal até {year}: {total} novos artigos")
            time.sleep(5)  # pausa entre anos para não sobrecarregar a API
        print(f"\n✅ Total coletado: {total} artigos ({args.year_from}–{args.year_to})")
        print("   Aguardando downloads pendentes... (Ctrl+C para parar)")
        while collector.running:
            time.sleep(10)
    except KeyboardInterrupt:
        collector.stop()


def cmd_collect(args):
    """Busca novos artigos no arXiv e baixa PDFs continuamente (Ctrl+C para parar)."""
    collector = ContinuousCollector(db_url=args.db, data_dir=args.data_dir)
    collector.start_download_workers(num_download_workers=1)

    try:
        while True:
            collector.search_new_papers(days_back=args.days_back)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        collector.stop()


def cmd_process_local(args):
    """Processa PDFs já presentes em disco: registra no banco, extrai texto e referências."""
    collector = ContinuousCollector(db_url=args.db, data_dir=args.data_dir,
                                     discover_new_papers=not args.no_discover)
    collector.scan_local_pdfs()

    # Enfileira extração para tudo que já tem PDF mas não tem texto
    from src.db import Paper, get_session_factory
    SessionLocal = get_session_factory(args.db)
    session = SessionLocal()
    try:
        pending = session.query(Paper.id).filter(
            Paper.pdf_downloaded == True, Paper.text_extracted == False
        ).all()
    finally:
        session.close()
    for (paper_id,) in pending:
        collector.add_to_queue(paper_id, 'extract_text', priority=2)

    print(f"\n📄 Processando {len(pending)} papers pendentes de extração...")
    _drain_extraction_queue(collector)
    print("✅ Processamento local concluído")


def _drain_extraction_queue(collector):
    """Roda o extract_worker de forma síncrona (single-thread) até não haver
    mais tarefas pendentes de 'extract_text' ou 'extract_refs'."""
    from src.db import get_session_factory, Paper
    from src.extraction.worker import _handle_extract_text, _handle_extract_refs

    collector.running = True
    while collector.running:
        task = collector.get_next_task('extract_text') or collector.get_next_task('extract_refs')
        if not task:
            collector.running = False
            return

        SessionLocal = collector.SessionLocal
        session = SessionLocal()
        try:
            paper = session.get(Paper, task['paper_id'])
            if not paper or not paper.pdf_downloaded:
                collector.complete_task(task['id'], False, "PDF não disponível")
                continue
            if task['task_type'] == 'extract_text':
                pdf_path = collector.pdf_dir / f"{paper.id}.pdf"
                _handle_extract_text(collector, session, paper, pdf_path, task['id'])
            else:
                _handle_extract_refs(collector, session, paper, task['id'])
        except Exception as e:
            session.rollback()
            print(f"   ❌ Erro: {e}")
            collector.complete_task(task['id'], False, str(e))
        finally:
            session.close()


def cmd_train(args):
    """Treina GNNs (GCN, GraphSAGE, GAT) e baselines clássicos."""
    train_models(db_url=args.db, min_degree=args.min_degree)



def cmd_evaluate(args):
    """Treina e compara todos os modelos, imprimindo tabela de acurácia."""
    comparison = run_comparison(db_url=args.db)
    print_comparison_table(comparison)



def cmd_compare_features(args):
    """Compara modelos com TF-IDF puro vs. TF-IDF+embeddings (mesmo split)."""
    from src.evaluation.compare_models import run_feature_comparison, print_feature_comparison_table
    comparison = run_feature_comparison(db_url=args.db, feature_modes=('tfidf', 'combined'),
                                         min_degree=args.min_degree)
    print_feature_comparison_table(comparison)



def cmd_enrich_ss(args):
    """Enriquece com Semantic Scholar os papers já processados (references_extracted=True).

    Prioriza papers mais antigos (já indexados no S2). Papers de 2026 têm lag
    de 4-8 semanas para aparecer no S2 — use --before-year 2026 para focar
    em papers de 2025 ou anteriores.

    Use --limit para processar um subconjunto por vez (recomendado: 200-500,
    pois o rate limit do S2 é ~100 req/5min no tier gratuito)."""
    from src.db import Paper, Citation, get_session_factory
    from src.config import ENABLE_SEMANTIC_SCHOLAR

    if not ENABLE_SEMANTIC_SCHOLAR:
        print("❌ ENABLE_SEMANTIC_SCHOLAR=False em src/config.py. Ative para usar este comando.")
        return

    collector = ContinuousCollector(db_url=args.db, data_dir=args.data_dir)
    SessionLocal = get_session_factory(args.db)

    session = SessionLocal()
    try:
        query = session.query(Paper.id, Paper.published_date).filter(
            Paper.references_extracted == True
        )
        # Filtra por ano se --before-year foi passado
        if args.before_year:
            from datetime import datetime
            cutoff = datetime(args.before_year, 1, 1)
            query = query.filter(Paper.published_date < cutoff)
        # Mais antigos primeiro — maior chance de estar indexado no S2
        query = query.order_by(Paper.published_date.asc())
        if args.limit:
            query = query.limit(args.limit)
        paper_ids = [row[0] for row in query.all()]

        existing_edges = {
            (row[0], row[1])
            for row in session.query(Citation.source_id, Citation.target_id).all()
        }
        all_ids = {row[0] for row in session.query(Paper.id).all()}
    finally:
        session.close()

    print(f"\n🔬 Enriquecendo {len(paper_ids)} papers via Semantic Scholar...")
    if args.before_year:
        print(f"   (filtrando papers publicados antes de {args.before_year})")
    new_edges_total = 0
    errors = 0
    skipped_not_indexed = 0

    for i, paper_id in enumerate(paper_ids, 1):
        print(f"   [{i}/{len(paper_ids)}] {paper_id}", end=" ", flush=True)
        try:
            ss_refs = collector.fetch_citations_from_semantic_scholar(paper_id)
        except Exception as e:
            print(f"❌ {e}")
            errors += 1
            continue

        if not ss_refs:
            skipped_not_indexed += 1
            print(f"— não indexado ou sem refs no S2")
            continue

        new_citations = [
            ref for ref in ss_refs
            if ref in all_ids and (paper_id, ref) not in existing_edges
        ]

        if not new_citations:
            print(f"— sem novas arestas (S2: {len(ss_refs)} refs, nenhuma nova no banco)")
            continue

        session = SessionLocal()
        try:
            for ref_id in new_citations:
                session.add(Citation(
                    source_id=paper_id,
                    target_id=ref_id,
                    found_in_text=False,
                    confidence=0.7,
                ))
                existing_edges.add((paper_id, ref_id))
            session.commit()
            new_edges_total += len(new_citations)
            print(f"✓ +{len(new_citations)} arestas (S2: {len(ss_refs)} refs)")
        except Exception as e:
            session.rollback()
            print(f"❌ Erro ao salvar: {e}")
            errors += 1
        finally:
            session.close()

    print(f"\n✅ Enriquecimento concluído:")
    print(f"   Novas arestas adicionadas : {new_edges_total}")
    print(f"   Não indexados no S2       : {skipped_not_indexed}")
    print(f"   Erros                     : {errors}")
    print(f"   Papers processados        : {len(paper_ids)}")


def cmd_download_pending(args):
    """Baixa PDFs dos papers já registrados no banco mas ainda sem PDF
    (pdf_downloaded=False). Após cada download bem-sucedido, enfileira
    extração de texto e referências automaticamente.

    Use --limit para controlar quantos baixar por execução.
    Use --no-discover para não buscar novos papers via referências durante
    a extração (foco em densificar o grafo com o que já está no banco)."""
    from src.db import Paper, get_session_factory
    from src.extraction.worker import _handle_extract_text, _handle_extract_refs

    collector = ContinuousCollector(
        db_url=args.db,
        data_dir=args.data_dir,
        discover_new_papers=not args.no_discover,
    )
    SessionLocal = get_session_factory(args.db)

    # 1) Papéis pendentes de download, priorizando os mais antigos
    #    (mais chance de estar no Semantic Scholar e de ter PDF disponível)
    session = SessionLocal()
    try:
        query = (
            session.query(Paper.id)
            .filter(Paper.pdf_downloaded == False)
            .order_by(Paper.published_date.asc())
        )
        if args.limit:
            query = query.limit(args.limit)
        pending_ids = [row[0] for row in query.all()]
    finally:
        session.close()

    if not pending_ids:
        print("✅ Nenhum paper pendente de download.")
        return

    print(f"\n📥 {len(pending_ids)} papers pendentes de download"
          + (f" (discover={'sim' if not args.no_discover else 'não'})" ))

    # 2) Enfileira todos como tarefas 'download'
    for paper_id in pending_ids:
        collector.add_to_queue(paper_id, 'download', priority=3)

    # 3) Drena a fila de download de forma síncrona (sem threading)
    downloaded = 0
    failed = 0
    collector.running = True

    while collector.running:
        task = collector.get_next_task('download')
        if not task:
            break

        task_id = task['id']
        paper_id = task['paper_id']
        print(f"   📥 [{downloaded+failed+1}/{len(pending_ids)}] {paper_id}", end=" ", flush=True)

        session = SessionLocal()
        try:
            paper = session.get(Paper, paper_id)
            if not paper:
                collector.complete_task(task_id, False, "Paper não existe")
                print("— não encontrado no banco")
                failed += 1
                continue

            pdf_path = collector.pdf_dir / f"{paper_id}.pdf"

            # PDF já existe em disco
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                if not paper.pdf_downloaded:
                    paper.pdf_downloaded = True
                    paper.pdf_size = pdf_path.stat().st_size
                    paper.updated_at = datetime.utcnow()
                    session.commit()
                print(f"♻️  já existe ({pdf_path.stat().st_size // 1024} KB)")
                collector.add_to_queue(paper_id, 'extract_text', priority=2)
                collector.complete_task(task_id, True)
                downloaded += 1
                continue

            # Baixa via URL pública
            import urllib.request
            clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; arxiv-collector/1.0)'}
            req = urllib.request.Request(pdf_url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                with open(pdf_path, 'wb') as f:
                    f.write(response.read())

            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                paper.pdf_downloaded = True
                paper.pdf_size = pdf_path.stat().st_size
                paper.updated_at = datetime.utcnow()
                session.commit()
                print(f"✓ ({pdf_path.stat().st_size // 1024} KB)")
                collector.add_to_queue(paper_id, 'extract_text', priority=2)
                collector.complete_task(task_id, True)
                downloaded += 1
            else:
                raise Exception("PDF inválido após download")

        except Exception as e:
            session.rollback()
            print(f"❌ {e}")
            collector.complete_task(task_id, False, str(e))
            failed += 1
        finally:
            session.close()

    print(f"\n✅ Downloads concluídos: {downloaded} OK, {failed} erros")

    # 4) Agora extrai texto e referências de tudo que foi baixado
    if downloaded > 0:
        print(f"\n📄 Extraindo texto e referências de {downloaded} PDFs baixados...")
        _drain_extraction_queue(collector)


def cmd_dashboard(args):
    """Sobe o dashboard Streamlit."""
    from src.dashboard.app import create_dashboard
    create_dashboard()


def cmd_api(args):
    """Sobe a API FastAPI."""
    import uvicorn
    from src.api.server import app
    from src.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)


def cmd_all(args):
    """Pipeline completo: process-local -> train -> evaluate."""
    cmd_process_local(args)
    cmd_train(args)
    cmd_evaluate(args)
    print("\n✅ Pipeline completo. Para dashboard/API, rode separadamente:")
    print("   python cli.py dashboard")
    print("   python cli.py api")


def main():
    parser = argparse.ArgumentParser(description="arXiv Citation Graph - CLI")
    parser.add_argument("--db", default=DB_URL, help="URL do banco de dados")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Diretório de dados")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_collect = subparsers.add_parser("collect", help=cmd_collect.__doc__)
    p_collect.add_argument("--days-back", type=int, default=7, help="Janela de busca em dias")
    p_collect.add_argument("--interval", type=int, default=300, help="Intervalo entre buscas (s)")
    p_collect.set_defaults(func=cmd_collect)

    p_collect_year = subparsers.add_parser("collect-year", help=cmd_collect_year.__doc__)
    p_collect_year.add_argument("--from", dest="year_from", type=int, required=True,
                                 help="Ano inicial da busca (ex: 2018)")
    p_collect_year.add_argument("--to", dest="year_to", type=int, required=True,
                                 help="Ano final da busca (ex: 2022)")
    p_collect_year.set_defaults(func=cmd_collect_year)

    p_process = subparsers.add_parser("process-local", help=cmd_process_local.__doc__)
    p_process.add_argument("--no-discover", action="store_true",
                            help="Não buscar/baixar novos papers via referências; "
                                 "apenas registrar citações entre papers já existentes "
                                 "(aumenta a densidade do grafo sem inflar o número de nós)")
    p_process.set_defaults(func=cmd_process_local)

    p_train = subparsers.add_parser("train", help=cmd_train.__doc__)
    p_train.add_argument("--min-degree", type=int, default=1,
                          help="Grau mínimo (in+out) para incluir um nó no treino. "
                               "1=remove apenas isolados, 2+=foca em nós com vizinhança real.")
    p_train.set_defaults(func=cmd_train)


    p_eval = subparsers.add_parser("evaluate", help=cmd_evaluate.__doc__)
    p_eval.add_argument("--min-degree", type=int, default=1)
    p_eval.set_defaults(func=cmd_evaluate)
 
    p_compare = subparsers.add_parser("compare-features", help=cmd_compare_features.__doc__)
    p_compare.add_argument("--min-degree", type=int, default=1,
                            help="Grau mínimo (in+out) para incluir um nó. Padrão=1.")
    p_compare.set_defaults(func=cmd_compare_features)


    p_enrich = subparsers.add_parser("enrich-ss", help=cmd_enrich_ss.__doc__)
    p_enrich.add_argument(
        "--limit", type=int, default=None,
        help="Número máximo de papers a processar (padrão: todos)."
    )
    p_enrich.add_argument(
        "--before-year", type=int, default=2026,
        help="Processa apenas papers publicados antes deste ano (padrão: 2026). "
             "Papers de 2026 ainda não estão indexados no S2."
    )
    p_enrich.set_defaults(func=cmd_enrich_ss)

    p_dl = subparsers.add_parser("download-pending", help=cmd_download_pending.__doc__)
    p_dl.add_argument(
        "--limit", type=int, default=None,
        help="Número máximo de papers a baixar (padrão: todos os pendentes). "
             "Recomendado: 200-500 por execução."
    )
    p_dl.add_argument(
        "--no-discover", action="store_true",
        help="Não buscar/baixar novos papers via referências durante a extração."
    )
    p_dl.set_defaults(func=cmd_download_pending)

    p_cleanup = subparsers.add_parser(
        "cleanup-pdfs",
        help=cmd_cleanup_pdfs.__doc__,
    )

    p_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas audita os PDFs; não remove arquivos.",
    )

    p_cleanup.set_defaults(
        func=cmd_cleanup_pdfs
    )

    p_dash = subparsers.add_parser("dashboard", help=cmd_dashboard.__doc__)
    p_dash.set_defaults(func=cmd_dashboard)

    p_api = subparsers.add_parser("api", help=cmd_api.__doc__)
    p_api.set_defaults(func=cmd_api)

    p_all = subparsers.add_parser("all", help=cmd_all.__doc__)
    p_all.add_argument("--days-back", type=int, default=7)
    p_all.add_argument("--interval", type=int, default=300)
    p_all.add_argument("--no-discover", action="store_true",
                        help="Não buscar/baixar novos papers via referências durante process-local")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main();
