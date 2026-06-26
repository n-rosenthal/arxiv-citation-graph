from typing import Dict, List, Tuple

import pandas as pd
import altair as alt
import networkx as nx

from src.features import extract_primary_categories, extract_macro_categories

# Paleta consistente com plots.py
C = {
    "blue":    "#4C72B0",
    "green":   "#55A868",
    "red":     "#C44E52",
    "purple":  "#8172B2",
    "orange":  "#DD8452",
    "teal":    "#4C8FA0",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _bar(df, x, y, color=C["blue"], title="", tooltip=None, sort="-y",
         x_title=None, y_title=None, height=320) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{x}:N", sort=sort, title=x_title or x),
            y=alt.Y(f"{y}:Q", title=y_title or y),
            tooltip=tooltip or [x, y],
        )
        .properties(title=title, height=height)
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────

def altair_pipeline_status(papers_df: pd.DataFrame) -> alt.Chart:
    stages = pd.DataFrame([
        {"etapa": "Total",                "count": len(papers_df),                             "cor": C["blue"]},
        {"etapa": "PDF baixado",          "count": int(papers_df["pdf_downloaded"].sum()),      "cor": C["green"]},
        {"etapa": "Texto extraído",       "count": int(papers_df["text_extracted"].sum()),      "cor": C["red"]},
        {"etapa": "Refs. extraídas",      "count": int(papers_df["references_extracted"].sum()), "cor": C["purple"]},
    ])
    order = list(stages["etapa"])
    return (
        alt.Chart(stages)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("etapa:N", sort=order, title=None),
            y=alt.Y("count:Q", title="Número de papers"),
            color=alt.Color("etapa:N", scale=alt.Scale(
                domain=order,
                range=[C["blue"], C["green"], C["red"], C["purple"]]),
                legend=None),
            tooltip=["etapa", "count"],
        )
        .properties(title="Progresso do Pipeline de Coleta", height=320)
    )



# ── Categorias ───────────────────────────────────────────────────────────────

def altair_category_distribution(papers_df: pd.DataFrame) -> alt.Chart:
    cats = extract_primary_categories(list(papers_df.itertuples(index=False)))
    df = pd.Series(cats).value_counts().head(10).reset_index()
    df.columns = ["categoria", "count"]
    return (
        alt.Chart(df)
        .mark_bar(color=C["blue"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("categoria:N", sort="-y", title="Categoria arXiv",
                    axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("count:Q", title="Número de papers"),
            tooltip=["categoria", "count"],
        )
        .properties(title="Top 10 Categorias Primárias", height=320)
    )



def altair_macro_category_distribution(papers_df: pd.DataFrame) -> alt.Chart:
    macro = extract_macro_categories(list(papers_df.itertuples(index=False)))
    df = pd.Series(macro).value_counts().reset_index()
    df.columns = ["macro_area", "count"]
    df["pct"] = (df["count"] / df["count"].sum() * 100).round(1)
 
    # Donut chart
    base = alt.Chart(df).encode(
        theta=alt.Theta("count:Q", stack=True),
        color=alt.Color(
            "macro_area:N",
            scale=alt.Scale(scheme="tableau10"),
            legend=alt.Legend(title="Macro-área", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("macro_area:N", title="Área"),
            alt.Tooltip("count:Q", title="Papers"),
            alt.Tooltip("pct:Q", title="% do total", format=".1f"),
        ],
    )
    arc = base.mark_arc(innerRadius=70, outerRadius=140, stroke="white", strokeWidth=1.5)
    text = base.mark_text(radius=165, size=11).encode(
        text=alt.Text("pct:Q", format=".0f"),
    )
    return (arc + text).properties(
        title="Distribuição de Macro-Áreas (labels de classificação)", height=360
    )



def altair_class_balance_comparison(papers_df_before: pd.DataFrame,
                                     papers_df_after: pd.DataFrame) -> alt.Chart:
    """Antes e depois da subamostragem lado a lado."""
    def _counts(df, label):
        macro = extract_macro_categories(list(df.itertuples(index=False)))
        c = pd.Series(macro).value_counts().reset_index()
        c.columns = ["macro_area", "count"]
        c["dataset"] = label
        return c
 
    combined = pd.concat([
        _counts(papers_df_before, "Original"),
        _counts(papers_df_after, "Balanceado"),
    ])
    return (
        alt.Chart(combined)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("macro_area:N", title="Macro-área"),
            y=alt.Y("count:Q", title="Número de papers"),
            color=alt.Color("dataset:N", scale=alt.Scale(
                domain=["Original", "Balanceado"],
                range=[C["red"], C["green"]])),
            xOffset="dataset:N",
            tooltip=["macro_area", "dataset", "count"],
        )
        .properties(title="Distribuição antes e depois da subamostragem", height=360)
    )



# ── Abstracts / Referências ──────────────────────────────────────────────────

def altair_abstract_length(papers_df: pd.DataFrame) -> alt.Chart:
    df = papers_df.copy()
    df["abstract_words"] = df["abstract"].fillna("").str.split().apply(len)
    return (
        alt.Chart(df)
        .mark_bar(color=C["green"], cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("abstract_words:Q", bin=alt.Bin(maxbins=30), title="Palavras no abstract"),
            y=alt.Y("count():Q", title="Número de papers"),
            tooltip=[alt.Tooltip("count():Q", title="papers")],
        )
        .properties(title="Distribuição do Tamanho dos Abstracts", height=320)
    )



def altair_references_distribution(papers_df: pd.DataFrame) -> alt.Chart:
    df = papers_df[papers_df["num_references"] > 0]
    left = (
        alt.Chart(df)
        .mark_bar(color=C["blue"], cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("num_references:Q", bin=alt.Bin(maxbins=20), title="Refs. no texto"),
            y=alt.Y("count():Q", title="Número de papers"),
            tooltip=[alt.Tooltip("count():Q", title="papers")],
        )
        .properties(title="Referências encontradas no texto", height=300, width=320)
    )
    right = (
        alt.Chart(df)
        .mark_bar(color=C["red"], cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("num_citations_in_graph:Q", bin=alt.Bin(maxbins=20), title="Citações no grafo"),
            y=alt.Y("count():Q", title=None),
            tooltip=[alt.Tooltip("count():Q", title="papers")],
        )
        .properties(title="Referências presentes no grafo", height=300, width=320)
    )
    return left | right



def altair_reference_distribution_simple(papers_df: pd.DataFrame) -> alt.Chart:
    df = papers_df[papers_df["num_references"] > 0]
    counts = df["num_references"].value_counts().head(20).sort_index().reset_index()
    counts.columns = ["num_references", "count"]
    return (
        alt.Chart(counts)
        .mark_bar(color=C["blue"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("num_references:O", title="Número de referências",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Número de papers"),
            tooltip=["num_references", "count"],
        )
        .properties(title="Referências encontradas no texto", height=320)
    )



# ── Grafo ────────────────────────────────────────────────────────────────────

def altair_degree_distribution(g: nx.DiGraph) -> alt.Chart:
    out_df = pd.DataFrame({"degree": [d for _, d in g.out_degree()], "tipo": "out-degree"})
    in_df  = pd.DataFrame({"degree": [d for _, d in g.in_degree()],  "tipo": "in-degree"})
    df = pd.concat([out_df, in_df])
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("degree:Q", bin=alt.Bin(maxbins=25), title="Grau"),
            y=alt.Y("count():Q", title="Número de nós"),
            color=alt.Color("tipo:N", scale=alt.Scale(
                domain=["out-degree", "in-degree"],
                range=[C["blue"], C["red"]])),
            tooltip=["tipo", alt.Tooltip("count():Q", title="nós")],
        )
        .properties(title="Distribuição de Grau (in / out)", height=320)
        .interactive()
    )



def altair_category_citation_matrix(g: nx.DiGraph, graph_data: Dict,
                                     papers_df: pd.DataFrame) -> alt.Chart:
    idx_to_id = {idx: pid for pid, idx in graph_data["id_to_idx"].items()}
    id_to_cat = dict(zip(
        papers_df["id"],
        extract_macro_categories(list(papers_df.itertuples(index=False)))
    ))
    cat_pairs = [
        {"origem": id_to_cat.get(idx_to_id[s], "?"),
         "destino": id_to_cat.get(idx_to_id[t], "?")}
        for s, t in graph_data["edges"]
    ]
    if not cat_pairs:
        return (alt.Chart(pd.DataFrame({"msg": ["Sem arestas"]}))
                .mark_text(size=14).encode(text="msg:N").properties(height=300))
    df = pd.DataFrame(cat_pairs).groupby(["origem", "destino"]).size().reset_index(name="citações")
    return (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("destino:N", title="Categoria citada",
                    axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("origem:N", title="Categoria que cita"),
            color=alt.Color(
                "citações:Q",
                scale=alt.Scale(scheme="viridis"),
                title="Citações",
                legend=alt.Legend(gradientLength=200),
            ),
            tooltip=["origem", "destino", "citações"],
        )
        .properties(title="Citações entre Macro-Áreas", height=350)
    )



def altair_homophily_bar(g: nx.DiGraph, graph_data: Dict,
                          papers_df: pd.DataFrame) -> alt.Chart:
    """Barra dupla: arestas dentro da mesma macro-área vs. entre áreas distintas."""
    idx_to_id = {idx: pid for pid, idx in graph_data["id_to_idx"].items()}
    id_to_cat = dict(zip(
        papers_df["id"],
        extract_macro_categories(list(papers_df.itertuples(index=False)))
    ))
    same = sum(1 for s, t in graph_data["edges"]
               if id_to_cat.get(idx_to_id[s]) == id_to_cat.get(idx_to_id[t]))
    total = len(graph_data["edges"])
    df = pd.DataFrame([
        {"tipo": "Mesma macro-área",    "arestas": same},
        {"tipo": "Entre macro-áreas",   "arestas": total - same},
    ])
    homophily = same / total if total else 0
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("tipo:N", title=None),
            y=alt.Y("arestas:Q", title="Número de arestas"),
            color=alt.Color("tipo:N", scale=alt.Scale(
                domain=["Mesma macro-área", "Entre macro-áreas"],
                range=[C["blue"], C["red"]]), legend=None),
            tooltip=["tipo", "arestas"],
        )
        .properties(title=f"Homofilia do grafo: {homophily:.2f}", height=320)
    )



def altair_top_cited(g: nx.DiGraph, graph_data: Dict,
                     papers_df: pd.DataFrame, top_n: int = 10) -> alt.Chart:
    """Lollipop chart dos papers mais citados (in-degree) — título legível."""
    idx_to_id = {idx: pid for pid, idx in graph_data["id_to_idx"].items()}
    top = sorted(g.in_degree(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows = []
    for idx, deg in top:
        pid = idx_to_id[idx]
        title_vals = papers_df.loc[papers_df["id"] == pid, "title"].values
        full_title = title_vals[0] if len(title_vals) else pid
        label = (full_title[:60] + "…") if len(full_title) > 60 else full_title
        rows.append({"paper": label, "in_degree": deg, "id": pid})
    df = pd.DataFrame(rows)
 
    base = alt.Chart(df).encode(
        y=alt.Y("paper:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=400, labelFontSize=11)),
        x=alt.X("in_degree:Q", title="In-degree (citações recebidas)"),
        tooltip=["id", "paper", "in_degree"],
    )
    stem = base.mark_rule(color="#cccccc", strokeWidth=1.5)
    dot  = base.mark_circle(
        color=C["blue"], size=90,
    ).encode(
        color=alt.Color(
            "in_degree:Q",
            scale=alt.Scale(scheme="blues", domainMin=0),
            legend=None,
        )
    )
    return (stem + dot).properties(
        title=f"Top {top_n} Papers Mais Citados", height=max(320, top_n * 34)
    )


# ── Crescimento ───────────────────────────────────────────────────────────────

def altair_growth_chart(metrics_df: pd.DataFrame) -> alt.Chart:
    df = metrics_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=4)
    df = df[df["timestamp"] >= cutoff]
    return (
        alt.Chart(df)
        .mark_line(
            point=alt.OverlayMarkDef(filled=True, size=40),
            interpolate="monotone",
            strokeWidth=2,
        )
        .encode(
            x=alt.X("timestamp:T",
                    axis=alt.Axis(
                        format="%d/%m %H:%M",
                        labelAngle=-45,
                        labelOverlap=False,
                    )),
            y=alt.Y("metric_value:Q", title="Valor",
                    scale=alt.Scale(type="log"),
                    axis=alt.Axis(format="~s")),
            color=alt.Color("metric_name:N", title="Métrica"),
            tooltip=["metric_name", "metric_value",
                     alt.Tooltip("timestamp:T", format="%d/%m %H:%M")],
        )
        .properties(
            width="container",
            height=400,
        )
        .interactive()
    )


# ── Modelos ───────────────────────────────────────────────────────────────────

def altair_model_comparison(comparison: Dict) -> alt.Chart:
    rows = [{"modelo": name,
             "test_acc": res["test_acc"],
             "tipo": "GNN" if name in ("gcn", "graphsage", "gat") else "Baseline"}
            for name, res in comparison.items()]
    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("modelo:N", sort="-y", title=None),
            y=alt.Y("test_acc:Q", title="Acurácia (teste)",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("tipo:N", scale=alt.Scale(
                domain=["GNN", "Baseline"],
                range=[C["blue"], C["purple"]])),
            tooltip=["modelo", alt.Tooltip("test_acc:Q", format=".2%"), "tipo"],
        )
        .properties(title="Acurácia no Conjunto de Teste — GNNs vs. Baselines", height=380)
    )


def altair_feature_mode_comparison(feature_comparison: Dict) -> alt.Chart:
    """Barras agrupadas por feature_mode (tfidf / combined) para cada modelo."""
    rows = []
    for mode, results in feature_comparison.items():
        for name, res in results.items():
            if name == "baselines":
                for bname, bres in res.items():
                    rows.append({"modelo": bname, "feature_mode": mode,
                                 "test_acc": bres["test_acc"],
                                 "tipo": "Baseline"})
            else:
                rows.append({"modelo": name, "feature_mode": mode,
                             "test_acc": res["test_acc"],
                             "tipo": "GNN"})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("modelo:N", title=None),
            y=alt.Y("test_acc:Q", title="Acurácia (teste)",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("feature_mode:N",
                            scale=alt.Scale(domain=["tfidf", "embeddings", "combined"],
                                            range=[C["blue"], C["green"], C["orange"]]),
                            title="Feature mode"),
            xOffset="feature_mode:N",
            tooltip=["modelo", "feature_mode",
                     alt.Tooltip("test_acc:Q", format=".2%"), "tipo"],
        )
        .properties(title="TF-IDF vs. Embeddings vs. Combinado por Modelo", height=380)
        .interactive()
    )
 
 
def altair_task_comparison(results: Dict) -> alt.Chart:
    """Compara GNN vs baseline em múltiplas tarefas de classificação.
 
    results: {
        "tarefa": {"GNN": float, "Baseline": float, "Ingênuo": float},
        ...
    }"""
    rows = []
    for tarefa, vals in results.items():
        for modelo, acc in vals.items():
            rows.append({"tarefa": tarefa, "modelo": modelo, "acurácia": acc})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("tarefa:N", title=None),
            y=alt.Y("acurácia:Q", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("modelo:N", scale=alt.Scale(
                domain=["GNN", "Baseline", "Ingênuo"],
                range=[C["blue"], C["purple"], C["red"]])),
            xOffset="modelo:N",
            tooltip=["tarefa", "modelo", alt.Tooltip("acurácia:Q", format=".2%")],
        )
        .properties(title="GNN vs. Baseline por Tipo de Tarefa", height=400)
    )


def altair_checkpoints_timeline(checkpoints_df: pd.DataFrame) -> alt.Chart:
    """Evolução da acurácia dos modelos ao longo das versões de treino."""
    return (
        alt.Chart(checkpoints_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("version:O", title="Versão"),
            y=alt.Y("accuracy:Q", title="Acurácia",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_name:N", title="Modelo"),
            tooltip=["model_name", "version",
                     alt.Tooltip("accuracy:Q", format=".2%"),
                     "num_nodes", "num_edges"],
        )
        .properties(title="Evolução da Acurácia por Versão de Treino", height=340)
        .interactive()
    )


