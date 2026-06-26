from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx

from src.features import extract_primary_categories, extract_macro_categories


def plot_pipeline_status(papers_df: pd.DataFrame) -> plt.Figure:
    """Barras com o progresso do pipeline: total, PDF baixado, texto extraído,
    referências extraídas."""
    stages = {
        "Total": len(papers_df),
        "PDF baixado": papers_df["pdf_downloaded"].sum(),
        "Texto extraído": papers_df["text_extracted"].sum(),
        "Referências extraídas": papers_df["references_extracted"].sum(),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages.keys(), stages.values(), color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_title("Progresso do Pipeline de Coleta")
    ax.set_ylabel("Número de papers")
    for bar, val in zip(bars, stages.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val, str(int(val)), ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_category_distribution(papers_df: pd.DataFrame) -> plt.Figure:
    """Barras com a distribuição de categorias primárias."""
    primary_categories = extract_primary_categories(list(papers_df.itertuples(index=False)))
    cat_counts = pd.Series(primary_categories).value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    cat_counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Distribuição de Categorias Primárias")
    ax.set_xlabel("Categoria arXiv")
    ax.set_ylabel("Número de papers")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_macro_category_distribution(papers_df: pd.DataFrame) -> plt.Figure:
    """Barras com a distribuição de macro-áreas (cs, stat, math, eess, ...) —
    o agrupamento usado como label de classificação pelos modelos
    (ver src.features.extract_macro_categories)."""
    macro_categories = extract_macro_categories(list(papers_df.itertuples(index=False)))
    cat_counts = pd.Series(macro_categories).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    cat_counts.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Distribuição de Macro-Áreas (labels de classificação)")
    ax.set_xlabel("Macro-área arXiv")
    ax.set_ylabel("Número de papers")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def plot_abstract_length(papers_df: pd.DataFrame) -> plt.Figure:
    """Histograma do tamanho dos abstracts (em palavras)."""
    abstract_lengths = papers_df["abstract"].fillna("").str.split().apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abstract_lengths, bins=30, color="#55A868", edgecolor="white")
    ax.set_title("Distribuição do Tamanho dos Abstracts")
    ax.set_xlabel("Número de palavras")
    ax.set_ylabel("Número de papers")
    fig.tight_layout()
    return fig


def plot_references_distribution(papers_df: pd.DataFrame) -> plt.Figure:
    """Histogramas lado a lado: referências encontradas no texto vs. presentes no grafo."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    axes[0].hist(papers_df["num_references"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Referências encontradas no texto")
    axes[0].set_xlabel("Número de referências")
    axes[0].set_ylabel("Número de papers")

    axes[1].hist(papers_df["num_citations_in_graph"], bins=20, color="#C44E52", edgecolor="white")
    axes[1].set_title("Referências presentes no grafo")
    axes[1].set_xlabel("Número de citações no grafo")

    fig.tight_layout()
    return fig


def build_nx_graph(graph_data: Dict) -> nx.DiGraph:
    """Constrói um nx.DiGraph a partir do dict retornado por build_graph_from_db."""
    g = nx.DiGraph()
    g.add_nodes_from(range(graph_data["num_nodes"]))
    g.add_edges_from(graph_data["edges"])
    return g


def plot_degree_distribution(g: nx.DiGraph) -> plt.Figure:
    """Histogramas lado a lado de out-degree e in-degree."""
    in_degrees = [d for _, d in g.in_degree()]
    out_degrees = [d for _, d in g.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    axes[0].hist(out_degrees, bins=range(0, max(out_degrees, default=1) + 2), color="#4C72B0", edgecolor="white")
    axes[0].set_title("Out-degree (artigos citados, presentes no grafo)")
    axes[0].set_xlabel("Grau")
    axes[0].set_ylabel("Número de papers")

    axes[1].hist(in_degrees, bins=range(0, max(in_degrees, default=1) + 2), color="#C44E52", edgecolor="white")
    axes[1].set_title("In-degree (vezes citado)")
    axes[1].set_xlabel("Grau")

    fig.tight_layout()
    return fig


def get_top_cited(g: nx.DiGraph, graph_data: Dict, papers_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Retorna um DataFrame com os top_n papers mais citados (in-degree) no grafo."""
    idx_to_id = {idx: pid for pid, idx in graph_data["id_to_idx"].items()}
    top_in = sorted(g.in_degree(), key=lambda kv: kv[1], reverse=True)[:top_n]

    rows = []
    for idx, deg in top_in:
        pid = idx_to_id[idx]
        title = papers_df.loc[papers_df["id"] == pid, "title"].values
        rows.append({"id": pid, "in_degree": deg, "title": title[0][:70] if len(title) else "?"})
    return pd.DataFrame(rows)


def plot_largest_component(g: nx.DiGraph) -> Tuple[plt.Figure, int]:
    """Plota o maior componente fracamente conectado do grafo.
    Retorna (figura, tamanho_do_componente)."""
    undirected = g.to_undirected()
    components = list(nx.connected_components(undirected))
    largest = max(components, key=len) if components else set()

    h = g.subgraph(largest)

    fig, ax = plt.subplots(figsize=(9, 9))
    pos = nx.spring_layout(h, seed=42, k=0.5)
    node_sizes = [50 + 30 * h.in_degree(n) for n in h.nodes()]
    nx.draw_networkx_nodes(h, pos, node_size=node_sizes, node_color="#4C72B0", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(h, pos, alpha=0.3, arrows=True, arrowsize=8, ax=ax)
    ax.set_title(f"Maior componente do grafo de citações ({len(largest)} nós)")
    ax.axis("off")
    fig.tight_layout()
    return fig, len(largest)


def plot_category_citation_matrix(g: nx.DiGraph, graph_data: Dict, papers_df: pd.DataFrame) -> plt.Figure:
    """Heatmap de citações entre macro-áreas (origem -> destino)."""
    idx_to_id = {idx: pid for pid, idx in graph_data["id_to_idx"].items()}
    id_to_cat = dict(zip(papers_df["id"], extract_macro_categories(list(papers_df.itertuples(index=False)))))

    cat_pairs = []
    for src_idx, tgt_idx in graph_data["edges"]:
        src_cat = id_to_cat.get(idx_to_id[src_idx], "?")
        tgt_cat = id_to_cat.get(idx_to_id[tgt_idx], "?")
        cat_pairs.append((src_cat, tgt_cat))

    fig, ax = plt.subplots(figsize=(8, 7))
    if cat_pairs:
        pair_df = pd.DataFrame(cat_pairs, columns=["source_category", "target_category"])
        matrix = pair_df.groupby(["source_category", "target_category"]).size().unstack(fill_value=0)

        im = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel("Categoria do artigo citado")
        ax.set_ylabel("Categoria do artigo que cita")
        ax.set_title("Citações entre categorias")
        fig.colorbar(im, ax=ax, label="Número de citações")
    else:
        ax.text(0.5, 0.5, "Sem arestas suficientes ainda", ha="center")
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_model_comparison(comparison: Dict) -> plt.Figure:
    """Barras comparando test_acc de todos os modelos (GNNs vs baselines)."""
    comp_df = pd.DataFrame([
        {"model": name, "test_acc": res["test_acc"]}
        for name, res in comparison.items()
    ]).sort_values("test_acc", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if m in ("gcn", "graphsage", "gat") else "#8172B2" for m in comp_df["model"]]
    bars = ax.bar(comp_df["model"], comp_df["test_acc"], color=colors)
    ax.set_title("Acurácia no conjunto de teste — GNNs vs. Baselines Clássicos")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for bar, val in zip(bars, comp_df["test_acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1%}", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig
