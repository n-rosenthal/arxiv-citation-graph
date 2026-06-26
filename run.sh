#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run.sh — orquestra todo o pipeline do projeto
# Uso: ./run.sh [comando]
# Comandos: setup | collect | process | train | evaluate |
#           dashboard | api | notebook | all
# ============================================================

VENV=".venv"
DB="sqlite:///data/arxiv_continuous.db?check_same_thread=False"
DATA_DIR="data"

activate() {
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
}

setup() {
    echo "🔧 Criando ambiente virtual e instalando dependências..."
    python3 -m venv "$VENV"
    activate
    pip install --upgrade pip
    pip install -r requirements.txt
    mkdir -p data/pdfs data/texts data/graphs models notebooks
    echo "✅ Setup concluído"
}

collect() {
    DAYS="${2:-365}"
    echo "📥 Coletando artigos dos últimos $DAYS dias (Ctrl+C para parar)..."
    activate
    python cli.py --db "$DB" --data-dir "$DATA_DIR" collect --days-back "$DAYS"
}

collect_year() {
    FROM="${2:-2020}"
    TO="${3:-2023}"
    echo "📅 Coletando artigos de $FROM a $TO (mais antigos primeiro)..."
    activate
    python cli.py --db "$DB" --data-dir "$DATA_DIR" collect-year --from "$FROM" --to "$TO"
}

process() {
    echo "🗃️  Processando corpus local (texto + referências)..."
    activate
    python cli.py --db "$DB" --data-dir "$DATA_DIR" process-local
}

densify() {
    echo "🕸️  Processando corpus local SEM descobrir novos papers (foco em densidade)..."
    activate
    python cli.py --db "$DB" --data-dir "$DATA_DIR" process-local --no-discover
}

enrich_ss() {
    LIMIT="${2:-300}"
    echo "🔬 Enriquecendo citações via Semantic Scholar (limite: $LIMIT papers, antes de 2026)..."
    activate
    python cli.py --db "$DB" enrich-ss --limit "$LIMIT" --before-year 2026
}

download_pending() {
    LIMIT="${2:-500}"
    echo "📥 Baixando PDFs pendentes (limite: $LIMIT papers)..."
    activate
    python cli.py --db "$DB" --data-dir "$DATA_DIR" download-pending --limit "$LIMIT" --no-discover
}

cleanup_pdfs() {

    activate

    if [[ "${2:-}" == "--dry-run" ]]; then

        echo "🔎 Auditoria de PDFs..."
        python cli.py \
            --db "$DB" \
            --data-dir "$DATA_DIR" \
            cleanup-pdfs \
            --dry-run

    else

        echo "🧹 Limpando PDFs processados..."
        python cli.py \
            --db "$DB" \
            --data-dir "$DATA_DIR" \
            cleanup-pdfs

    fi
}

train() {
    MIN_DEG="${2:-2}"
    echo "🎓 Treinando modelos (GNNs + baseline, min_degree=$MIN_DEG)..."
    activate
    python cli.py --db "$DB" train --min-degree "$MIN_DEG"
}
 
evaluate() {
    echo "📊 Comparando modelos e gerando relatório de métricas..."
    activate
    python cli.py --db "$DB" evaluate
}
 
compare_features() {
    MIN_DEG="${2:-2}"
    echo "🧬 Comparando TF-IDF vs. TF-IDF+embeddings (min_degree=$MIN_DEG)..."
    activate
    python cli.py --db "$DB" compare-features --min-degree "$MIN_DEG"
}


dashboard() {
    echo "📈 Subindo dashboard Streamlit..."
    activate
    streamlit run src/dashboard/app.py
}

api() {
    echo "🌐 Subindo API FastAPI..."
    activate
    python cli.py --db "$DB" api
}

notebook() {
    echo "📓 Abrindo Jupyter..."
    activate
    jupyter notebook notebooks/01_eda_and_results.ipynb
}

all() {
    activate
    python cli.py --db "$DB" --data-dir "$DATA_DIR" all
    echo "✅ Pipeline completo. Para dashboard/API/notebook, rode separadamente:"
    echo "   ./run.sh dashboard | ./run.sh api | ./run.sh notebook"
}

case "${1:-}" in
    setup)     setup ;;
    collect)      collect "$@" ;;
    collect-year) collect_year "$@" ;;
    process)   process ;;
    densify)   densify ;;
    enrich-ss)        enrich_ss "$@" ;;
    download-pending) download_pending "$@" ;;
    cleanup-pdfs) cleanup_pdfs "$@" ;;
    train)     train ;;
    evaluate)  evaluate ;;
    compare-features) compare_features ;;
    dashboard) dashboard ;;
    api)       api ;;
    notebook)  notebook ;;
    all)       all ;;
    *)
        echo "Uso: ./run.sh {setup|collect|collect-year|process|densify|download-pending|cleanup-pdfs|enrich-ss|train|evaluate|compare-features|dashboard|api|notebook|all}" 
        exit 1
        ;;
esac
