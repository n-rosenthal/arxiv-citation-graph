# ============================================================================
# COLLECTOR — busca, download de PDFs, fila de processamento, scan local
# ============================================================================
import re
import time
import random
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set

import arxiv
import requests

from src.config import (
    DB_URL, DATA_DIR, PDF_DIR, TEXT_DIR, GRAPH_DIR,
    DEFAULT_CATEGORIES, EXCLUDED_CATEGORIES, ARXIV_BATCH_SIZE,
    ARXIV_ID_PATTERN, REF_PATTERN, BARE_ID_PATTERN,
    ENABLE_SEMANTIC_SCHOLAR, SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_RATE_LIMIT, SEMANTIC_SCHOLAR_TIMEOUT,
    SEMANTIC_SCHOLAR_FIELDS,
)
from src.db import Paper, Citation, ProcessingQueue, get_session_factory
from src.collection.arxiv_client import ArxivRateLimitedClient


class ContinuousCollector:
    def __init__(self, db_url: str = DB_URL, data_dir: str = None, discover_new_papers: bool = True):
        self.SessionLocal = get_session_factory(db_url)

        # Diretórios
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.pdf_dir = self.data_dir / 'pdfs' if data_dir else PDF_DIR
        self.text_dir = self.data_dir / 'texts' if data_dir else TEXT_DIR
        self.graph_dir = self.data_dir / 'graphs' if data_dir else GRAPH_DIR
        for d in [self.pdf_dir, self.text_dir, self.graph_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Filas em memória (opcionais, mas mantidas)
        self.download_queue = queue.Queue()
        self.extract_queue = queue.Queue()
        self.train_queue = queue.Queue()

        self.running = False
        self.workers = []

        # Se False, extract_refs registra citações apenas entre papers já
        # presentes no banco e NÃO busca/baixa novos papers referenciados.
        # Útil para "fechar" o grafo (aumentar densidade) sem inflar
        # indefinidamente o número de nós.
        self.discover_new_papers = discover_new_papers

        # Padrões regex (vindos de config.py)
        self.arxiv_id_pattern = re.compile(ARXIV_ID_PATTERN)
        self.ref_pattern = re.compile(REF_PATTERN, re.IGNORECASE)
        self.bare_id_pattern = re.compile(BARE_ID_PATTERN)

        # Cliente centralizado com rate limiting — única instância para todo o processo
        self._arxiv = ArxivRateLimitedClient()

        print("✅ Sistema de Coleta Contínua Inicializado")

    def search_by_year_range(self, year_start: int, year_end: int,
                               categories: List[str] = None) -> int:
        """Busca artigos dentro de uma faixa de anos específica, ordenando
        pelos MAIS ANTIGOS primeiro — contorna o limite de 1000 resultados
        da API quando combinado com múltiplas chamadas por ano.

        Útil para coletar histórico de categorias com poucos artigos
        (stat.ML, math.CT, etc.) que nunca aparecem na busca por recência."""
        cat_query = ' OR '.join([f'cat:{cat}' for cat in (categories or DEFAULT_CATEGORIES)])
        date_from = f"{year_start}0101"
        date_to = f"{year_end}1231"
        query = f"({cat_query}) AND submittedDate:[{date_from} TO {date_to}]"
        print(f"\n🔍 Buscando {year_start}–{year_end}: {query[:80]}...")

        client = arxiv.Client(page_size=100, delay_seconds=5.0, num_retries=15);
        search = arxiv.Search(
            query=query,
            max_results=1000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending,   
        )
        results = list(client.results(search))

        if EXCLUDED_CATEGORIES:
            before = len(results)
            results = [
                r for r in results
                if not any(
                    cat.startswith(prefix)
                    for prefix in EXCLUDED_CATEGORIES
                    for cat in (r.categories or [])
                )
            ]
            if before != len(results):
                print(f"   ↪️ Após filtro: {len(results)}/{before} restantes")

        session = self.SessionLocal()
        try:
            existing_ids: Set[str] = {row[0] for row in session.query(Paper.id).all()}
        finally:
            session.close()

        new_papers = []
        for result in results:
            paper_id = result.entry_id.split('/abs/')[-1]
            if paper_id not in existing_ids:
                new_papers.append(Paper(
                    id=paper_id,
                    title=result.title,
                    authors=', '.join([a.name for a in result.authors]),
                    abstract=result.summary,
                    categories=', '.join(result.categories),
                    published_date=result.published,
                    updated_date=result.updated,
                    pdf_downloaded=False,
                    text_extracted=False,
                    references_extracted=False
                ))

        new_paper_ids = []
        if new_papers:
            session = self.SessionLocal()
            try:
                session.add_all(new_papers)
                session.commit()
                new_paper_ids = [p.id for p in new_papers]
                print(f"   ✓ {len(new_paper_ids)} novos artigos ({year_start}–{year_end})")
            except Exception as e:
                session.rollback()
                print(f"   ❌ Erro ao salvar: {e}")
            finally:
                session.close()
        else:
            print(f"   ✓ Nenhum artigo novo para {year_start}–{year_end}")

        for paper_id in new_paper_ids:
            self.add_to_queue(paper_id, 'download', priority=2)

        return len(new_paper_ids)

    # ------------------ Busca de novos artigos ------------------
    def search_new_papers(self, categories: List[str] = None, days_back: int = 7):
        cat_query = ' OR '.join([f'cat:{cat}' for cat in (categories or DEFAULT_CATEGORIES)])
        since_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y%m%d')
        query = f"({cat_query}) AND submittedDate:[{since_date} TO 99991231]"
        print(f"\n🔍 Buscando artigos desde {since_date}")
        print(f"   Query: {query}")

        # 1) Busca no arxiv SEM nenhuma sessao aberta — pode demorar minutos
        client = arxiv.Client(page_size=100, delay_seconds=5.0)
        search = arxiv.Search(query=query, max_results=1000, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = list(client.results(search))  # coleta tudo antes de tocar no banco

        # Aplica filtro de exclusão de categorias (ex: EXCLUDED_CATEGORIES = ["cs."])
        if EXCLUDED_CATEGORIES:
            before = len(results)
            results = [
                r for r in results
                if not any(
                    cat.startswith(prefix)
                    for prefix in EXCLUDED_CATEGORIES
                    for cat in (r.categories or [])
                )
            ]
            print(f"   ↪️ Após filtro de exclusão: {len(results)}/{before} artigos restantes")

        # 2) Carrega IDs ja existentes de uma vez so (leitura rapida, sem autoflush)
        session = self.SessionLocal()
        try:
            existing_ids: Set[str] = {row[0] for row in session.query(Paper.id).all()}
        finally:
            session.close()

        # 3) Filtra apenas os novos e monta objetos Paper em memoria
        new_papers = []
        for result in results:
            paper_id = result.entry_id.split('/abs/')[-1]
            if paper_id not in existing_ids:
                new_papers.append(Paper(
                    id=paper_id,
                    title=result.title,
                    authors=', '.join([a.name for a in result.authors]),
                    abstract=result.summary,
                    categories=', '.join(result.categories),
                    published_date=result.published,
                    updated_date=result.updated,
                    pdf_downloaded=False,
                    text_extracted=False,
                    references_extracted=False
                ))

        # 4) Insere todos de uma vez em uma unica transacao curta
        new_paper_ids = []
        if new_papers:
            session = self.SessionLocal()
            try:
                session.add_all(new_papers)
                session.commit()
                new_paper_ids = [p.id for p in new_papers]
                print(f"   ✓ Encontrados {len(new_paper_ids)} novos artigos")
            except Exception as e:
                session.rollback()
                print(f"   ❌ Erro ao salvar artigos: {e}")
            finally:
                session.close()
        else:
            print(f"   ✓ Nenhum artigo novo encontrado")

        # 5) Enfileira DEPOIS de fechar a sessao
        for paper_id in new_paper_ids:
            self.add_to_queue(paper_id, 'download', priority=1)

        return len(new_paper_ids)

    # ------------------ Fila de processamento ------------------
    def add_to_queue(self, paper_id: str, task_type: str, priority: int = 5):
        for attempt in range(5):
            session = self.SessionLocal()
            try:
                existing = session.query(ProcessingQueue).filter_by(
                    paper_id=paper_id, task_type=task_type, status='pending'
                ).first()
                if not existing:
                    task = ProcessingQueue(paper_id=paper_id, task_type=task_type, priority=priority, status='pending')
                    session.add(task)
                    session.commit()
                session.close()
                return  # sucesso
            except Exception as e:
                session.rollback()
                session.close()
                if attempt < 4:
                    time.sleep(0.3 * (attempt + 1) + random.random() * 0.5)
                else:
                    print(f"   ❌ Erro ao adicionar tarefa {paper_id}/{task_type}: {e}")

    def get_next_task(self, task_type: str):
        """Pega a proxima tarefa pendente com retry para lidar com lock do SQLite."""
        for attempt in range(5):
            session = self.SessionLocal()
            try:
                task = session.query(ProcessingQueue).filter_by(
                    task_type=task_type, status='pending'
                ).order_by(ProcessingQueue.priority, ProcessingQueue.created_at).first()
                if task:
                    task.status = 'processing'
                    task.started_at = datetime.utcnow()
                    session.commit()
                    # Le todos os atributos ANTES de fechar (evita DetachedInstanceError)
                    task_id = task.id
                    paper_id = task.paper_id
                    task_type_val = task.task_type
                    session.close()
                    return {'id': task_id, 'paper_id': paper_id, 'task_type': task_type_val}
                else:
                    session.close()
                    return None
            except Exception as e:
                session.rollback()
                session.close()
                if attempt < 4:
                    time.sleep(0.5 * (attempt + 1) + random.random())
                else:
                    print(f"   ❌ Erro em get_next_task apos {attempt+1} tentativas: {e}")
                    return None

    def complete_task(self, task_id: int, success: bool, error_msg: str = None):
        session = self.SessionLocal()
        try:
            task = session.get(ProcessingQueue, task_id)
            if task:
                task.status = 'completed' if success else 'failed'
                task.completed_at = datetime.utcnow()
                if error_msg:
                    task.error_message = error_msg
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"   ❌ Erro ao completar tarefa {task_id}: {e}")
        finally:
            session.close()

    # ------------------ Worker de download ------------------
    def download_worker(self):
        """Baixa PDFs e, ao concluir, enfileira 'extract_text' (consumido por src.extraction)."""
        print("📥 Worker de download iniciado")
        while self.running:
            try:
                task_info = self.get_next_task('download')
                if not task_info:
                    time.sleep(5)
                    continue
                task_id = task_info['id']
                paper_id = task_info['paper_id']
                print(f"\n📥 Baixando: {paper_id}")

                session = self.SessionLocal()
                try:
                    paper = session.get(Paper, paper_id)
                    if not paper:
                        self.complete_task(task_id, True, "Paper nao existe no banco")
                        continue
                    pdf_path = self.pdf_dir / f"{paper_id}.pdf"
                    # Se o PDF já existe em disco, apenas atualiza o banco e enfileira extração
                    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                        if not paper.pdf_downloaded:
                            paper.pdf_downloaded = True
                            paper.pdf_size = pdf_path.stat().st_size
                            paper.updated_at = datetime.utcnow()
                            session.commit()
                        print(f"   ♻️  PDF já existe em disco: {paper_id} ({pdf_path.stat().st_size // 1024} KB)")
                        self.add_to_queue(paper_id, 'extract_text', priority=2)
                        self.complete_task(task_id, True)
                        continue
                    if paper.pdf_downloaded:
                        # Banco diz que foi baixado mas arquivo sumiu — re-baixa
                        paper.pdf_downloaded = False
                        session.commit()
                    # Baixa direto via URL publica
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
                        print(f"   ✓ PDF baixado ({paper.pdf_size / 1024:.1f} KB)")
                        self.add_to_queue(paper_id, 'extract_text', priority=2)
                        self.complete_task(task_id, True)
                    else:
                        raise Exception("PDF invalido ou nao encontrado apos download")
                except Exception as e:
                    session.rollback()
                    print(f"   ❌ Erro no download: {e}")
                    self.complete_task(task_id, False, str(e))
                finally:
                    session.close()
            except Exception as e:
                print(f"   ❌ Erro inesperado no worker: {e}")
                time.sleep(10)

    # ------------------ Busca de papers referenciados ------------------
    def fetch_papers_by_ids(self, paper_ids: List[str]):
        """Busca metadados de papers referenciados via cliente centralizado com rate limit.

        Se self.discover_new_papers=False, não faz nada (modo de
        'fechamento do grafo': só registra citações entre papers já
        existentes, sem expandir o conjunto de nós)."""
        if not self.discover_new_papers:
            return

        # 1) Descobre quais IDs ainda não existem no banco
        session = self.SessionLocal()
        try:
            existing_ids = {row[0] for row in session.query(Paper.id).all()}
        finally:
            session.close()

        to_fetch = [pid for pid in paper_ids if pid not in existing_ids]
        if not to_fetch:
            return

        # 2) Uma única requisição em lote via fila de rate limiting
        meta_list = []
        for i in range(0, len(to_fetch), ARXIV_BATCH_SIZE):
            meta_list.extend(self._arxiv.fetch_batch(to_fetch[i:i + ARXIV_BATCH_SIZE]))

        if not meta_list:
            return

        # 3) Normaliza IDs (remove versão) e remove duplicatas dentro do próprio lote
        seen = set()
        new_papers = []
        excluded_count = 0
        for m in meta_list:
            clean_id = m["id"].split('v')[0]
            if clean_id in existing_ids or clean_id in seen:
                continue

            # Filtra por categoria excluída (ex: EXCLUDED_CATEGORIES = ["cs."])
            # m["categories"] é uma string "cat1, cat2, ..." — verifica todas
            if EXCLUDED_CATEGORIES:
                cats = [c.strip() for c in (m.get("categories") or "").split(",")]
                if any(cat.startswith(prefix) for prefix in EXCLUDED_CATEGORIES for cat in cats):
                    excluded_count += 1
                    continue

            seen.add(clean_id)
            new_papers.append(Paper(
                id=clean_id, title=m["title"], authors=m["authors"],
                abstract=m["abstract"], categories=m["categories"],
                published_date=m["published_date"], updated_date=m["updated_date"],
                pdf_downloaded=False
            ))

        if excluded_count:
            print(f"   ↪️ {excluded_count} referências excluídas por categoria (EXCLUDED_CATEGORIES)")

        if not new_papers:
            return

        # 4) Insere um a um, ignorando conflitos individuais (outro worker pode
        #    ter inserido o mesmo ID entre o passo 1 e agora)
        inserted_ids = []
        for paper in new_papers:
            session = self.SessionLocal()
            try:
                session.add(paper)
                session.commit()
                inserted_ids.append(paper.id)
            except Exception:
                session.rollback()  # já existe (corrida com outro worker) — ignora
            finally:
                session.close()

        if inserted_ids:
            print(f"   ✓ {len(inserted_ids)} novos artigos adicionados via referências")

        # 5) Enfileira DEPOIS de fechar sessão
        for pid in inserted_ids:
            self.add_to_queue(pid, 'download', priority=4)

    def fetch_metadata_for_stubs(self, batch_size: int = 25):
        """Busca metadados reais para papers que só têm ID (registrados via scan_local_pdfs).
        Usa o cliente centralizado com rate limiting — seguro chamar de qualquer thread."""
        session = self.SessionLocal()
        try:
            stubs = [row[0] for row in session.query(Paper.id).filter(
                Paper.abstract == ""
            ).limit(batch_size).all()]
        finally:
            session.close()

        if not stubs:
            return  # silencioso — chamado em loop pelo orquestrador

        print(f"   🔄 Buscando metadados para {len(stubs)} papers (lote)...")
        updated = self._arxiv.fetch_batch(stubs)

        if updated:
            session = self.SessionLocal()
            try:
                for meta in updated:
                    paper = session.get(Paper, meta['id'])
                    if paper:
                        for k, v in meta.items():
                            if k != 'id':
                                setattr(paper, k, v)
                session.commit()
                print(f"   ✓ Metadados atualizados para {len(updated)} papers")
            except Exception as e:
                session.rollback()
                print(f"   ❌ Erro ao salvar metadados: {e}")
            finally:
                session.close()

    # ------------------ Processamento de dados locais ------------------
    def scan_local_pdfs(self) -> int:
        """Registra no banco todos os PDFs já presentes em disco que ainda não estão cadastrados.
        Usa apenas o ID extraído do nome do arquivo — não faz nenhuma chamada de rede."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        print(f"\n🗂️  Escaneando {len(pdf_files)} PDFs locais em {self.pdf_dir}...")

        # IDs já cadastrados no banco
        session = self.SessionLocal()
        try:
            existing_ids = {row[0] for row in session.query(Paper.id).all()}
        finally:
            session.close()

        new_papers = []
        for pdf_file in pdf_files:
            # Nome do arquivo é o paper_id (ex: 2606.12412v1.pdf)
            paper_id = pdf_file.stem
            if paper_id in existing_ids:
                continue
            new_papers.append(Paper(
                id=paper_id,
                title=f"[Título pendente] {paper_id}",
                authors="",
                abstract="",
                categories="",
                pdf_path=str(pdf_file),
                pdf_downloaded=True,
                pdf_size=pdf_file.stat().st_size,
                text_extracted=False,
                references_extracted=False,
            ))

        if new_papers:
            session = self.SessionLocal()
            try:
                session.add_all(new_papers)
                session.commit()
                print(f"   ✓ {len(new_papers)} novos papers registrados a partir do disco")
            except Exception as e:
                session.rollback()
                print(f"   ❌ Erro ao registrar PDFs locais: {e}")
                new_papers = []
            finally:
                session.close()
        else:
            print(f"   ✓ Todos os PDFs locais já estavam cadastrados")

        return len(new_papers)

    # ------------------ Semantic Scholar ------------------
    def fetch_citations_from_semantic_scholar(self, paper_id: str) -> List[str]:
        """Consulta a API Graph v1 do Semantic Scholar para obter referências de
        um paper arXiv. Retorna lista de IDs arXiv normalizados (sem versão).

        Endpoint: GET /graph/v1/paper/arXiv:{id}/references?fields=externalIds
        Rate limit free tier: ~100 req/5 min (~1 req/3s); usamos 1.0s por padrão
        pois chamadas ocorrem em série com outras operações de I/O."""
        if not ENABLE_SEMANTIC_SCHOLAR:
            return []

        time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)
        clean_id = paper_id.split('v')[0]
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/arXiv:{clean_id}/references"
        params = {"fields": SEMANTIC_SCHOLAR_FIELDS, "limit": 500}

        try:
            resp = requests.get(url, params=params, timeout=SEMANTIC_SCHOLAR_TIMEOUT)
            if resp.status_code == 404:
                return []  # paper não indexado no S2 — silencioso
            if resp.status_code == 400:
                # Geralmente significa que o paper ainda não foi indexado pelo S2
                # (comum para papers muito recentes — lag de semanas a meses)
                return []
            if resp.status_code == 429:
                print(f"   ⏳ Semantic Scholar rate limit — aguardando 60s...")
                time.sleep(60)
                return []
            if resp.status_code != 200:
                print(f"   ⚠️ Semantic Scholar HTTP {resp.status_code} para {paper_id}")
                return []

            data = resp.json()
            references = []
            for item in data.get("data", []):
                ext_ids = item.get("citedPaper", {}).get("externalIds") or {}
                arxiv_id = ext_ids.get("ArXiv")
                if arxiv_id and re.match(r'\d{4}\.\d{4,5}', arxiv_id):
                    references.append(arxiv_id.split('v')[0])
            return references

        except requests.exceptions.Timeout:
            print(f"   ⚠️ Timeout no Semantic Scholar para {paper_id}")
            return []
        except Exception as e:
            print(f"   ⚠️ Erro no Semantic Scholar para {paper_id}: {e}")
            return []

    # ------------------ Controle de workers de download ------------------
    def start_download_workers(self, num_download_workers: int = 2):
        self.running = True
        for _ in range(num_download_workers):
            t = threading.Thread(target=self.download_worker, daemon=True)
            t.start()
            self.workers.append(t)
        print(f"\n🚀 {num_download_workers} workers de download ativos")

    def stop(self):
        self.running = False
        print("\n🛑 Parando coletor...")
        for w in self.workers:
            w.join(timeout=5)
        print("✅ Coletor parado")