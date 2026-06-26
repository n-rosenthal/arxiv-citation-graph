# ============================================================================
# DASHBOARD (STREAMLIT)
# ============================================================================
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
import networkx as nx
from matplotlib import pyplot as plt

from src.config import DB_URL
from src.db import get_engine, get_session_factory
from src.graph import build_graph_from_db
from src.graph.metrics import compute_graph_metrics

from src.visualizations import (
    build_nx_graph,
    plot_largest_component,
)

from src.visualizations.altair_plots import (
    altair_growth_chart,
    altair_reference_distribution_simple,
    altair_category_distribution,
    altair_macro_category_distribution,
    altair_degree_distribution,
    altair_category_citation_matrix,
    altair_top_cited,
)
from src.visualizations.model_plots import (
    altair_checkpoint_timeline,
    altair_accuracy_vs_nodes,
    altair_accuracy_vs_edges,
    altair_training_duration,
    altair_job_status,
    checkpoint_regression_summary
)


# ============================================================================
# CACHE
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_graph(db_path: str):
    """
    Carrega o banco e constrói o grafo NetworkX.
    Executado apenas quando o banco muda.
    """
    SessionLocal = get_session_factory(db_path)

    graph_data = build_graph_from_db(
        SessionLocal
    )

    g = build_nx_graph(
        graph_data
    )

    return graph_data, g
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


@st.cache_resource(show_spinner=False)
def compute_largest_component(_g: nx.DiGraph):
    """
    Calcula apenas uma vez o maior componente conectado
    e o layout utilizado para desenhá-lo.
    """

    largest_nodes = max(
        nx.connected_components(
            _g.to_undirected()
        ),
        key=len,
    )

    h = _g.subgraph(
        largest_nodes
    ).copy()

    pos = nx.spring_layout(
        h,
        seed=42,
        k=0.5,
    )

    return h, pos


def render_largest_component(
    g: nx.DiGraph,
) -> tuple[Figure, int]:
    """
    Renderiza o maior componente utilizando o layout
    previamente cacheado.
    """

    h, pos = compute_largest_component(g)

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    node_sizes = [
        50 + 30 * h.in_degree(node)
        for node in h.nodes()
    ]

    nx.draw_networkx_nodes(
        h,
        pos,
        node_size=node_sizes,
        node_color="#4C72B0",
        alpha=0.8,
        ax=ax,
    )

    nx.draw_networkx_edges(
        h,
        pos,
        alpha=0.3,
        arrows=False,
        ax=ax,
    )

    ax.set_title(
        f"Maior componente do grafo de citações ({h.number_of_nodes()} nós)"
    )

    ax.axis("off")

    fig.tight_layout()

    return fig, h.number_of_nodes()

# ============================================================================
# SNAPSHOT METRICS
# ============================================================================

def _snapshot_metrics(engine, total_papers: int, total_citations: int):
    from sqlalchemy import text

    ts = datetime.utcnow().isoformat()

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO dashboard_metrics
                (metric_name, metric_value, timestamp)
                VALUES
                (:n, :v, :t)
            """),
            [
                {
                    "n": "total_papers",
                    "v": float(total_papers),
                    "t": ts,
                },
                {
                    "n": "total_citations",
                    "v": float(total_citations),
                    "t": ts,
                },
            ],
        )


# ============================================================================
# MAIN
# ============================================================================

def create_dashboard():
    st.set_page_config(
        page_title="arXiv Citation Graph Monitor",
        layout="wide",
    )

    st.title("📚 arXiv Citation Graph")

    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (30s)",
        value=True,
    )

    db_path = st.sidebar.text_input(
        "Database Path",
        DB_URL,
    )

    engine = get_engine(db_path)

    try:
        _render_dashboard(
            engine,
            db_path,
        )
    except Exception as e:
        st.error(f"Erro ao carregar dashboard: {e}")
        st.exception(e)

    if auto_refresh:
        time.sleep(30)
        st.rerun()


# ============================================================================
# DASHBOARD
# ============================================================================

def _render_dashboard(engine, db_path: str):

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    papers_df = pd.read_sql(
        "SELECT * FROM papers",
        engine,
    )

    queue_df = pd.read_sql(
        """
        SELECT *
        FROM processing_queue
        WHERE status != 'completed'
        ORDER BY created_at
        """,
        engine,
    )

    citations_count = pd.read_sql(
        "SELECT COUNT(*) AS n FROM citations",
        engine,
    )["n"].iloc[0]

    try:
        _snapshot_metrics(
            engine,
            len(papers_df),
            int(citations_count),
        )
    except Exception as e:
        st.sidebar.warning(
            f"Snapshot falhou: {e}"
        )

    metrics_df = pd.read_sql(
        """
        SELECT *
        FROM dashboard_metrics
        ORDER BY timestamp ASC
        """,
        engine,
    )

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------

    tab_coleta, tab_grafo, tab_modelos = st.tabs(
        [
            "Coleta",
            "Grafo",
            "Modelos",
        ]
    )

    # ==================================================================
    # COLETA
    # ==================================================================

    with tab_coleta:

        st.subheader("Resumo")

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric(
                "Artigos",
                len(papers_df),
            )

        with m2:
            st.metric(
                "PDFs",
                int(
                    papers_df["pdf_downloaded"].sum()
                ),
            )

        with m3:
            st.metric(
                "Com Referências",
                int(
                    papers_df["references_extracted"].sum()
                ),
            )

        with m4:
            st.metric(
                "Citações",
                int(citations_count),
            )

        with m5:
            st.metric(
                "Fila",
                len(
                    queue_df[
                        queue_df["status"] == "pending"
                    ]
                ),
            )

        # --------------------------------------------------------------
        # Crescimento
        # --------------------------------------------------------------

        st.subheader(
            "Crescimento do Banco"
        )

        growth_df = metrics_df[
            metrics_df["metric_name"].isin(
                [
                    "total_papers",
                    "total_citations",
                ]
            )
        ].copy()

        if not growth_df.empty:
            growth_df["timestamp"] = pd.to_datetime(
                growth_df["timestamp"]
            )

            st.altair_chart(
                altair_growth_chart(
                    growth_df
                ),
                width="stretch",
            )

            col2, col3 = st.columns(2);
            with col2:
                st.subheader(
                    "Referências por Paper"
                )
                
                if (papers_df["num_references"] > 0).any():
                    st.altair_chart(
                        altair_reference_distribution_simple(
                            papers_df
                        ),
                        width='stretch',
                    )

        # --------------------------------------------------------------
        # Categorias
        # --------------------------------------------------------------

        with col3:
            st.subheader(
            "Categorias Primárias"
            )

            st.altair_chart(
                altair_category_distribution(
                    papers_df
                ),
                width='stretch',
            )

        # --------------------------------------------------------------
        # Recentes
        # --------------------------------------------------------------

        st.subheader(
            "Artigos Recentes"
        )

        recent = (
            papers_df
            .sort_values(
                "created_at",
                ascending=False,
            )
            .head(20)
        )

        st.dataframe(
            recent[
                [
                    "id",
                    "title",
                    "categories",
                    "pdf_downloaded",
                    "references_extracted",
                    "num_references",
                ]
            ],
            width='stretch',
        )

        # --------------------------------------------------------------
        # Fila
        # --------------------------------------------------------------

        st.subheader(
            "Fila de Processamento"
        )

        if not queue_df.empty:

            st.dataframe(
                queue_df[
                    [
                        "paper_id",
                        "task_type",
                        "status",
                        "priority",
                        "created_at",
                    ]
                ],
                width='stretch',
            )

        else:
            st.info("Fila vazia.")

    # ==================================================================
    # GRAFO
    # ==================================================================

    with tab_grafo:

        with st.spinner(
            "Construindo grafo..."
        ):

            graph_data, g = load_graph(
                db_path
            )

        metrics = compute_graph_metrics(
            g
        )

        st.subheader(
            "Métricas Estruturais"
        )

        g1, g2, g3 = st.columns(3)

        g1.metric(
            "Nós",
            metrics["nodes"],
        )

        g2.metric(
            "Arestas",
            metrics["edges"],
        )

        g3.metric(
            "Maior Componente Conexo",
            f"{metrics['giant_component_ratio']:.1%}",
        )

        if graph_data["num_edges"] > 0:

            st.subheader(
                "Top Papers Mais Citados"
	    )

            st.altair_chart(
                altair_top_cited(
                    g,
		    graph_data,
		    papers_df,
		    top_n=10,
		),
		width="stretch",
	    )

            with st.expander("Visualizar maior componente conectado"):
                if st.button("Renderizar componente"):
                    fig, size = render_largest_component(g);
                    st.write(f"Tamanho: {size} nós");
                    st.pyplot(fig);
                else:
                    st.info("Ainda não há arestas no grafo.");

    # ==================================================================
# MODELOS
# ==================================================================

    with tab_modelos:

        st.subheader("Modelos")

        try:
            checkpoints_df = pd.read_sql(
                """
                SELECT *
                FROM model_checkpoints
                ORDER BY training_date DESC
                """,
                engine,
            )
        except Exception:
            checkpoints_df = pd.DataFrame()

        try:
            jobs_df = pd.read_sql(
                """
                SELECT *
                FROM training_jobs
                ORDER BY id DESC
                """,
                engine,
            )
        except Exception:
            jobs_df = pd.DataFrame()

        # --------------------------------------------------------------
        # RESUMO
        # --------------------------------------------------------------

        if not checkpoints_df.empty:

            latest = checkpoints_df.iloc[0]

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Último Modelo",
                latest["model_name"],
            )

            m2.metric(
                "Accuracy",
                f"{latest['accuracy']:.4f}",
            )

            m3.metric(
                "Loss",
                f"{latest['loss']:.4f}",
            )

            m4.metric(
                "Versão",
                latest["version"],
            )

        else:
            st.info("Nenhum checkpoint encontrado.")

        # --------------------------------------------------------------
        # EVOLUÇÃO DOS CHECKPOINTS
        # --------------------------------------------------------------

        if not checkpoints_df.empty:

            st.subheader("Evolução dos Modelos")
            st.altair_chart(
                altair_checkpoint_timeline(
                    checkpoints_df
                ),
                width="stretch",
            )
            
            summary_df = checkpoint_regression_summary(
                checkpoints_df
            )
            
            if not summary_df.empty:
                
                st.subheader(
                    "Tendência dos Modelos"
                )

                st.dataframe(
                    summary_df,
                    width="stretch",
                    column_config={
                        "current_accuracy":
                        st.column_config.NumberColumn(
                            "Accuracy Atual",
                            format="%.4f",
                        ),
                        "predicted_next_accuracy":
                        st.column_config.NumberColumn(
                            "Próxima Accuracy",
                            format="%.4f",
                        ),
                        "slope":
                        st.column_config.NumberColumn(
                            "Slope",
                            format="%.6f",
                        ),
                        "r2":
                        st.column_config.NumberColumn(
                            "R²",
                            format="%.4f",
                        ),
                    },
                )

        # --------------------------------------------------------------
        # JOBS E TREINAMENTO
        # --------------------------------------------------------------

        if not jobs_df.empty:

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Status dos Jobs"
                )

                st.altair_chart(
                    altair_job_status(
                        jobs_df
                    ),
                    width='stretch',
                )

            with col2:

                st.subheader(
                    "Tempo Médio de Treinamento"
                )

                st.altair_chart(
                    altair_training_duration(
                        jobs_df
                    ),
                    width='stretch',
                )

        # --------------------------------------------------------------
        # ACCURACY VS TAMANHO DO GRAFO
        # --------------------------------------------------------------

        if not checkpoints_df.empty:

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Accuracy × Número de Nós"
                )

                st.altair_chart(
                    altair_accuracy_vs_nodes(
                        checkpoints_df
                    ),
                    width='stretch',
                )

            with col2:

                st.subheader(
                    "Accuracy × Número de Arestas"
                )

                st.altair_chart(
                    altair_accuracy_vs_edges(
                        checkpoints_df
                    ),
                    width='stretch',
                )

        # --------------------------------------------------------------
        # TABELAS (OPCIONAL)
        # --------------------------------------------------------------

        with st.expander(
            "Ver dados brutos dos checkpoints"
        ):

            if not checkpoints_df.empty:

                st.dataframe(
                    checkpoints_df[
                        [
                            "model_name",
                            "version",
                            "accuracy",
                            "loss",
                            "num_nodes",
                            "num_edges",
                            "training_date",
                        ]
                    ],
                    width='stretch',
                )

        with st.expander(
            "Ver jobs de treinamento"
        ):

            if not jobs_df.empty:

                st.dataframe(
                    jobs_df,
                    width='stretch',
                )


if __name__ == "__main__":
    create_dashboard()
