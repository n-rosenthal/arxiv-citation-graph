# ============================================================================
# src/visualizations/model_plots.py
# ============================================================================
#
# Visualizações Altair para monitoramento de treinamento de modelos.
#
# Espera DataFrames originados das tabelas:
#
#   model_checkpoints
#   training_jobs
#
# ============================================================================

import altair as alt
import pandas as pd


# ============================================================================
# CHECKPOINT TIMELINE
# ============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import streamlit as st


def checkpoint_regression_summary(
    checkpoints_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula regressão linear da accuracy ao longo do tempo
    para cada modelo.

    Retorna:
        model_name
        slope
        intercept
        r2
        current_accuracy
        predicted_next_accuracy
        num_checkpoints
    """

    if checkpoints_df.empty:
        return pd.DataFrame()

    df = checkpoints_df.copy()

    df["training_date"] = pd.to_datetime(
        df["training_date"]
    )

    results = []

    for model_name, group in df.groupby(
        "model_name"
    ):

        group = group.sort_values(
            "training_date"
        )

        if len(group) < 2:
            continue

        # Dias desde o primeiro checkpoint
        x = (
            group["training_date"]
            - group["training_date"].min()
        ).dt.total_seconds() / 86400.0

        X = x.values.reshape(-1, 1)

        y = group["accuracy"].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        r2 = float(r2_score(y, y_pred))

        current_accuracy = float(
            group["accuracy"].iloc[-1]
        )

        # previsão para o próximo checkpoint
        if len(group) >= 2:

            avg_step = (
                x.diff()
                .dropna()
                .mean()
            )

            next_x = float(
                x.iloc[-1] + avg_step
            )

        else:

            next_x = float(
                x.iloc[-1] + 1.0
            )

        predicted_next_accuracy = float(
            model.predict(
                [[next_x]]
            )[0]
        )

        results.append(
            {
                "model_name": model_name,
                "num_checkpoints": len(group),
                "current_accuracy": current_accuracy,
                "predicted_next_accuracy": predicted_next_accuracy,
                "slope": slope,
                "r2": r2,
            }
        )

    summary_df = pd.DataFrame(
        results
    )

    if not summary_df.empty:

        summary_df = summary_df.sort_values(
            "current_accuracy",
            ascending=False,
        )

    return summary_df


def altair_checkpoint_timeline(
    checkpoints_df: pd.DataFrame,
):
    """
    Evolução da accuracy ao longo do tempo com
    regressão linear por modelo.
    """

    if checkpoints_df.empty:
        return (
            alt.Chart(
                pd.DataFrame(
                    {"message": ["Sem checkpoints"]}
                )
            )
            .mark_text(size=16)
            .encode(text="message")
        )

    df = checkpoints_df.copy()

    df["training_date"] = pd.to_datetime(
        df["training_date"]
    )

    df = df.sort_values(
        ["model_name", "training_date"]
    )

    selection = alt.selection_point(
        fields=["model_name"],
        bind="legend",
    )

    base = (
        alt.Chart(df)
        .encode(
            x=alt.X(
                "training_date:T",
                title="Data"
            ),
            y=alt.Y(
                "accuracy:Q",
                title="Accuracy"
            ),
            color=alt.Color(
                "model_name:N",
                title="Modelo"
            ),
            tooltip=[
                "model_name",
                "version",
                alt.Tooltip(
                    "accuracy:Q",
                    format=".4f"
                ),
                alt.Tooltip(
                    "loss:Q",
                    format=".4f"
                ),
                alt.Tooltip(
                    "training_date:T",
                    title="Data"
                ),
            ],
        )
        .add_params(selection)
    )

    observations = (
        base
        .mark_line(
            point=True,
            strokeWidth=4,
        )
        .encode(
            opacity=alt.condition(
                selection,
                alt.value(1.0),
                alt.value(0.15),
            )
        )
    )

    regression = (
        alt.Chart(df)
        .transform_filter(selection)
        .transform_regression(
            "training_date",
            "accuracy",
            groupby=["model_name"],
        )
        .mark_line(
            strokeWidth=2,
            strokeDash=[8, 4],
        )
        .encode(
            x="training_date:T",
            y="accuracy:Q",
            color="model_name:N",
        )
    )

    return (
        observations + regression
    ).properties(
        height=400,
        title="Evolução dos Checkpoints com Tendência Linear",
    ).interactive()


# ============================================================================
# ACCURACY VS NODES
# ============================================================================

def altair_accuracy_vs_nodes(
    checkpoints_df: pd.DataFrame,
):
    """
    Accuracy em função do número de nós.
    """

    if checkpoints_df.empty:
        return (
            alt.Chart(
                pd.DataFrame(
                    {"message": ["Sem checkpoints"]}
                )
            )
            .mark_text(size=16)
            .encode(text="message")
        )

    df = checkpoints_df.sort_values(
        ["model_name", "num_nodes"]
    )

    base = alt.Chart(df).encode(
        x=alt.X(
            "num_nodes:Q",
            title="Número de Nós",
        ),
        y=alt.Y(
            "accuracy:Q",
            title="Accuracy",
        ),
        color=alt.Color(
            "model_name:N",
            title="Modelo",
        ),
        detail="model_name:N",
        tooltip=[
            "model_name",
            "version",
            "accuracy",
            "loss",
            "num_nodes",
            "num_edges",
        ],
    )

    lines = base.mark_line(
        strokeWidth=2,
    )

    points = base.mark_circle(
        size=120,
    )

    return (
        lines + points
    ).properties(
        height=400,
        title="Accuracy × Número de Nós",
    ).interactive()


# ============================================================================
# ACCURACY VS EDGES
# ============================================================================

def altair_accuracy_vs_edges(
    checkpoints_df: pd.DataFrame,
):
    """
    Accuracy em função do número de arestas.
    """

    if checkpoints_df.empty:
        return (
            alt.Chart(
                pd.DataFrame(
                    {"message": ["Sem checkpoints"]}
                )
            )
            .mark_text(size=16)
            .encode(text="message")
        )

    df = checkpoints_df.sort_values(
        ["model_name", "num_edges"]
    )

    base = alt.Chart(df).encode(
        x=alt.X(
            "num_edges:Q",
            title="Número de Arestas",
        ),
        y=alt.Y(
            "accuracy:Q",
            title="Accuracy",
        ),
        color=alt.Color(
            "model_name:N",
            title="Modelo",
        ),
        detail="model_name:N",
        tooltip=[
            "model_name",
            "version",
            "accuracy",
            "loss",
            "num_nodes",
            "num_edges",
        ],
    )

    lines = base.mark_line(
        strokeWidth=2,
    )

    points = base.mark_circle(
        size=120,
    )

    return (
        lines + points
    ).properties(
        height=400,
        title="Accuracy × Número de Arestas",
    ).interactive()


# ============================================================================
# TRAINING DURATION
# ============================================================================

def altair_training_duration(
    jobs_df: pd.DataFrame,
):
    """
    Duração média dos treinamentos por modelo.

    Espera:
        started_at
        completed_at
        model_name
    """

    if jobs_df.empty:
        return alt.Chart(
            pd.DataFrame(
                {"message": ["Sem jobs"]}
            )
        ).mark_text(size=16).encode(
            text="message"
        )

    df = jobs_df.copy()

    df = df[
        df["started_at"].notna()
        &
        df["completed_at"].notna()
    ]

    if df.empty:
        return alt.Chart(
            pd.DataFrame(
                {"message": ["Sem jobs concluídos"]}
            )
        ).mark_text(size=16).encode(
            text="message"
        )

    df["started_at"] = pd.to_datetime(
        df["started_at"]
    )

    df["completed_at"] = pd.to_datetime(
        df["completed_at"]
    )

    df["duration_minutes"] = (
        (
            df["completed_at"]
            -
            df["started_at"]
        ).dt.total_seconds()
        / 60.0
    )

    summary = (
        df.groupby(
            "model_name",
            as_index=False
        )["duration_minutes"]
        .mean()
    )

    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "model_name:N",
                title="Modelo"
            ),
            y=alt.Y(
                "duration_minutes:Q",
                title="Duração Média (min)"
            ),
            tooltip=[
                "model_name",
                alt.Tooltip(
                    "duration_minutes:Q",
                    format=".2f"
                ),
            ],
        )
        .properties(
            height=400,
            title="Tempo Médio de Treinamento"
        )
    )


# ============================================================================
# JOB STATUS
# ============================================================================

def altair_job_status(
    jobs_df: pd.DataFrame,
):
    """
    Distribuição dos status dos jobs.

    Espera:
        status
    """

    if jobs_df.empty:
        return alt.Chart(
            pd.DataFrame(
                {"message": ["Sem jobs"]}
            )
        ).mark_text(size=16).encode(
            text="message"
        )

    status_df = (
        jobs_df["status"]
        .value_counts()
        .reset_index()
    )

    status_df.columns = [
        "status",
        "count",
    ]

    return (
        alt.Chart(status_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "status:N",
                title="Status"
            ),
            y=alt.Y(
                "count:Q",
                title="Quantidade"
            ),
            color=alt.Color(
                "status:N",
                legend=None
            ),
            tooltip=[
                "status",
                "count",
            ],
        )
        .properties(
            height=400,
            title="Status dos Jobs"
        )
    )
