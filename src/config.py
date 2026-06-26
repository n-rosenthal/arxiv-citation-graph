# ============================================================================
# CONFIGURAÇÃO CENTRAL DO PROJETO
# ============================================================================
from pathlib import Path

# ------------------ Paths ------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "texts"
GRAPH_DIR = DATA_DIR / "graphs"
MODELS_DIR = BASE_DIR / "models"

for d in (PDF_DIR, TEXT_DIR, GRAPH_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------ Database ------------------
DB_URL = f"sqlite:///{DATA_DIR / 'arxiv_continuous.db'}?check_same_thread=False"

# ------------------ arXiv collection ------------------
# Foco em categorias fora de cs.* para balancear o dataset
DEFAULT_CATEGORIES = [
    #"cs.LG",
    #"cs.AI",
    #"cs.CL",
    #"stat.ML",
    "math.CT",
    "stat.CO",
    "eess.SP",
    "physics.data-an",
]
# Papers cuja categoria primária começa com algum desses prefixos são ignorados
# durante search_new_papers. Útil para evitar acúmulo excessivo de cs.*
EXCLUDED_CATEGORIES: list[str] = []   # ex: ["cs."] para excluir toda cs.*

ARXIV_RATE_LIMIT_INTERVAL = 4  # segundos entre requisições
ARXIV_BATCH_SIZE = 25
DOWNLOAD_WORKERS = 1
EXTRACT_WORKERS = 2

# ------------------ Regex patterns ------------------
ARXIV_ID_PATTERN = r'(\d{4}\.\d{4,5})(?:v\d+)?'
REF_PATTERN = (
    r'(?:arxiv[:\s]+|arxiv\.org/(?:abs|pdf)/|\[)'
    r'(\d{4}\.\d{4,5})(?:v\d+)?'
)
BARE_ID_PATTERN = r'(?<![\d\.])([12]\d{3}\.\d{4,5})(?:v\d+)?(?![\d])'

# ------------------ External APIs ------------------
ENABLE_SEMANTIC_SCHOLAR = False
# Usa a API Graph v1 (o endpoint /v1/paper foi depreciado em 2023)
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0   # segundos entre requisições (free tier: ~100 req/5min)
SEMANTIC_SCHOLAR_TIMEOUT = 10       # segundos de timeout por requisição
# Campos solicitados à API (evita trazer tudo e exceder rate limit)
SEMANTIC_SCHOLAR_FIELDS = "references.externalIds,references.title"

# ------------------ Modeling ------------------
TFIDF_MAX_FEATURES = 500
GNN_HIDDEN_DIM = 128        # aumentado: mais capacidade com residual+JK
GNN_NUM_LAYERS = 3          # 3 camadas → agrega vizinhança de 3 hops
GAT_HIDDEN_DIM = 64         # aumentado: concatena heads internamente
GAT_HEADS = 4               # reduzido de 8→4: evita explosão de dimensão com 3 camadas
DROPOUT = 0.3               # reduzido: residual+BN já regularizam bem
GAT_DROPOUT = 0.3
LEARNING_RATE = 0.005       # reduzido: arquitetura mais complexa converge mais devagar
WEIGHT_DECAY = 5e-4
EPOCHS = 200                # aumentado: mais camadas precisam de mais épocas
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.8  # test = restante

# ------------------ Training trigger ------------------
MIN_NODES_FOR_TRAINING = 50
MIN_EDGES_FOR_TRAINING = 50
RETRAIN_EVERY_N_PAPERS = 50

# ------------------ API/Dashboard ------------------
API_HOST = "0.0.0.0"
API_PORT = 8001
