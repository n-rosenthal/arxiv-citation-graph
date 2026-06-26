# ============================================================================
# CLIENTE ARXIV COM RATE LIMITING GLOBAL
# ============================================================================
import threading
import queue
import time
from typing import List, Dict, Optional

import arxiv

from src.config import ARXIV_RATE_LIMIT_INTERVAL


class ArxivRateLimitedClient:
    """
    Wrapper singleton em torno do arxiv.Client que serializa TODAS as chamadas
    à API do arXiv em uma única thread dedicada, respeitando o rate limit de
    ~1 req/3s. Qualquer número de workers pode chamar fetch_metadata() e
    fetch_batch() — eles apenas enfileiram a requisição e aguardam o resultado.
    Backoff exponencial automático em caso de HTTP 429.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._req_queue: queue.Queue = queue.Queue()
        self._min_interval = ARXIV_RATE_LIMIT_INTERVAL  # segundos entre requisições
        self._last_req_time = 0.0
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="arxiv-api-worker")
        self._worker_thread.start()
        self._initialized = True

    def _worker(self):
        """Thread única que processa todas as requisições serializadas."""
        while True:
            item = self._req_queue.get()
            if item is None:
                break
            func, result_holder, event = item
            backoff = self._min_interval
            for attempt in range(6):
                # Garante espaçamento mínimo entre requisições
                elapsed = time.time() - self._last_req_time
                if elapsed < backoff:
                    time.sleep(backoff - elapsed)
                try:
                    result_holder["result"] = func()
                    self._last_req_time = time.time()
                    break
                except Exception as e:
                    msg = str(e)
                    if "429" in msg:
                        wait = backoff * (2 ** attempt) + (attempt * 1.5)
                        print(f"   ⏳ arXiv API rate limit — aguardando {wait:.0f}s (tentativa {attempt+1}/6)")
                        time.sleep(wait)
                        self._last_req_time = time.time()
                    else:
                        result_holder["error"] = e
                        break
            else:
                result_holder["error"] = Exception("arXiv API: máximo de tentativas atingido (429 persistente)")
            event.set()

    def _call(self, func):
        """Enfileira func() e bloqueia até o resultado estar disponível."""
        result_holder = {}
        event = threading.Event()
        self._req_queue.put((func, result_holder, event))
        event.wait()
        if "error" in result_holder:
            raise result_holder["error"]
        return result_holder.get("result")

    def fetch_metadata(self, paper_id: str) -> Optional[Dict]:
        """Busca metadados de um único paper. Retorna dict ou None se não encontrado."""
        def _fetch():
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            try:
                result = next(client.results(search))
                return {
                    "id": paper_id,
                    "title": result.title,
                    "authors": ", ".join([a.name for a in result.authors]),
                    "abstract": result.summary,
                    "categories": ", ".join(result.categories),
                    "published_date": result.published,
                    "updated_date": result.updated,
                }
            except StopIteration:
                return None
        try:
            return self._call(_fetch)
        except Exception as e:
            print(f"   ❌ Erro na API arXiv para {paper_id}: {e}")
            return None

    def fetch_batch(self, paper_ids: List[str]) -> List[Dict]:
        """Busca metadados de múltiplos IDs em uma única requisição (mais eficiente)."""
        if not paper_ids:
            return []
        def _fetch():
            client = arxiv.Client()
            search = arxiv.Search(id_list=paper_ids)
            results = []
            for result in client.results(search):
                pid = result.entry_id.split("/abs/")[-1]
                results.append({
                    "id": pid,
                    "title": result.title,
                    "authors": ", ".join([a.name for a in result.authors]),
                    "abstract": result.summary,
                    "categories": ", ".join(result.categories),
                    "published_date": result.published,
                    "updated_date": result.updated,
                })
            return results
        try:
            return self._call(_fetch)
        except Exception as e:
            print(f"   ❌ Erro na API arXiv (batch {len(paper_ids)} ids): {e}")
            return []