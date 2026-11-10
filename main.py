"""Tablero Streamlit del proyecto de siniestralidad vial (Ley 2251 de 2022).

La aplicación NO reentrena nada: lee los artefactos que exporta `main.ipynb`
(carpeta `artifacts/`), de modo que cada número del tablero es trazable al
notebook. Para desplegar en producción basta con este repositorio.

Ejecución local:
    streamlit run main.py
"""

import json
import os
from pathlib import Path
from unicodedata import normalize

import joblib
from openai import OpenAI, OpenAIError
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paleta de colores del proyecto (consistente con .streamlit/config.toml)
# ---------------------------------------------------------------------------
AZUL_NOCHE = "#003049"   # base / texto / series neutrales
ROJO = "#D62828"         # siniestros fatales, alertas
NARANJA = "#F77F00"      # color primario, énfasis
AMBAR = "#FCBF49"        # series secundarias, heridos
CREMA = "#EAE2B7"        # fondos suaves
GRIS_REJILLA = "#E8E4D8"

PALETA = [NARANJA, AZUL_NOCHE, ROJO, AMBAR, "#5E7B8C"]
COLOR_GRAVEDAD = {"CON HERIDOS": AMBAR, "CON MUERTOS": ROJO}

ARTIFACTS = Path(__file__).parent / "artifacts"

TIPOS_PESADOS = ["CAMION", "TRACTOCAMION", "VOLQUETA", "BUS", "MICROBUS", "BUSETA"]

NOMBRES_MODELOS = {
    "baseline_prevalencia": "Línea base de prevalencia",
    "logistica_base": "Regresión logística",
    "logistica_interacciones": "Logística + interacciones",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_csv(nombre: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / nombre)


@st.cache_data(show_spinner=False)
def cargar_json(nombre: str) -> dict:
    with (ARTIFACTS / nombre).open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def cargar_modelo() -> dict:
    return joblib.load(ARTIFACTS / "modelo_final.joblib")


def estilo_figura(fig: go.Figure, altura: int = 420) -> go.Figure:
    """Aplica el estilo común a todas las figuras del tablero."""
    fig.update_layout(
        height=altura,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=AZUL_NOCHE),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    fig.update_xaxes(gridcolor=GRIS_REJILLA, zerolinecolor=GRIS_REJILLA)
    fig.update_yaxes(gridcolor=GRIS_REJILLA, zerolinecolor=GRIS_REJILLA)
    return fig


# ---------------------------------------------------------------------------
# Secciones del tablero
# ---------------------------------------------------------------------------
def seccion_resumen(resumen: dict) -> None:
    st.title("Gravedad de siniestros viales en Colombia")
    st.caption(
        "Vehículos involucrados en accidentes de tránsito reportados bajo la "
        "Ley 2251 de 2022 ante el RUNT y publicados en datos.gov.co. El modelo "
        "prioriza registros ya reportados según su probabilidad de corresponder "
        "a un siniestro fatal."
    )

    test_metrics = resumen["metricas_test_2026"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registros analizados", f"{resumen['n_registros_vehiculo']:,}")
    col2.metric(
        "Modelo final",
        NOMBRES_MODELOS.get(resumen["mejor_modelo"], resumen["mejor_modelo"]),
    )
    col3.metric("Average Precision 2026", f"{test_metrics['average_precision']:.3f}")
    col4.metric("ROC AUC 2026", f"{test_metrics['roc_auc']:.3f}")
    col5.metric(
        "Recall de fatales 2026", f"{test_metrics['recall_muertes']:.1%}"
    )

    st.divider()
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        st.subheader("¿Qué hace este proyecto?")
        st.markdown(
            f"""
            - **Problema**: de los {resumen['n_registros_vehiculo']:,} registros de
              vehículos involucrados en siniestros, solo
              {resumen['tasa_global_muertes']:.1%} corresponde a accidentes
              **CON MUERTOS**. Identificarlos a tiempo permite priorizar
              investigación y recursos.
            - **Enfoque**: clasificación binaria con separación temporal estricta
              — entrena hasta 2024, calibra y fija umbral en 2025, evalúa en 2026 —
              y clustering descriptivo de perfiles territorio × tipo de vehículo.
            - **Decisión operacional**: el umbral de alerta se elige por costo
              esperado, donde un falso negativo cuesta
              {resumen['costo_falso_negativo']:.0f} veces un falso positivo, con
              una capacidad máxima de alertas del
              {resumen['capacidad_maxima_alertas_validacion']:.0%}.
            """
        )
    with derecha:
        st.subheader("Ficha técnica")
        st.markdown(
            f"""
            | | |
            |---|---|
            | Unidad de análisis | {resumen['unidad_analisis']} |
            | Entrenamiento | {resumen['n_train']:,} registros · 2022–2024 |
            | Validación | {resumen['n_valid']:,} registros · 2025 |
            | Prueba final | {resumen['n_test']:,} registros · 2026 |
            | Calibración | {resumen['calibracion'].split('_')[0]} |
            | Umbral de alerta | {resumen['umbral_mejor_modelo']:.3f} |
            """
        )

    st.info(
        "La fuente actual no trae un identificador oficial de accidente, por lo "
        "que la unidad es el **registro de vehículo involucrado**: el score indica "
        "gravedad reportada, no probabilidad de que ocurra un accidente."
    )


def seccion_datos() -> None:
    st.header("Datos y análisis exploratorio")
    st.caption(
        "Agregados calculados en el notebook sobre el CSV limpio: "
        "450 mil registros entre 2022 y 2026."
    )

    gravedad = cargar_csv("eda_gravedad.csv")
    anio = cargar_csv("eda_anio_gravedad.csv")
    tipo = cargar_csv("eda_tipo_vehiculo.csv")
    depto = cargar_csv("eda_departamento.csv")
    mes = cargar_csv("eda_mes.csv")

    col1, col2 = st.columns([2, 3])
    with col1:
        fig = px.pie(
            gravedad,
            values="registros",
            names="GRAVEDAD_ACCIDENTE",
            title="Distribución del objetivo",
            color="GRAVEDAD_ACCIDENTE",
            color_discrete_map=COLOR_GRAVEDAD,
            hole=0.45,
        )
        st.plotly_chart(estilo_figura(fig, 380), use_container_width=True)
    with col2:
        fig = px.bar(
            anio,
            x="ANIO_ACCIDENTE",
            y="registros",
            color="GRAVEDAD_ACCIDENTE",
            title="Registros por año y gravedad",
            color_discrete_map=COLOR_GRAVEDAD,
            labels={"ANIO_ACCIDENTE": "Año", "registros": "Registros"},
        )
        st.plotly_chart(estilo_figura(fig, 380), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(
            tipo.sort_values("registros"),
            x="registros",
            y="TIPO_VEHICULO",
            orientation="h",
            title="Top 15 tipos de vehículo",
            labels={"registros": "Registros", "TIPO_VEHICULO": ""},
            color_discrete_sequence=[AZUL_NOCHE],
            hover_data={"tasa_muertes": ":.2%"},
        )
        st.plotly_chart(estilo_figura(fig, 460), use_container_width=True)
    with col4:
        fig = px.bar(
            depto.sort_values("registros"),
            x="registros",
            y="DEPARTAMENTO_ACCIDENTE",
            orientation="h",
            title="Top 15 departamentos",
            labels={"registros": "Registros", "DEPARTAMENTO_ACCIDENTE": ""},
            color_discrete_sequence=[NARANJA],
            hover_data={"tasa_muertes": ":.2%"},
        )
        st.plotly_chart(estilo_figura(fig, 460), use_container_width=True)

    fig = px.line(
        mes,
        x="MES_ACCIDENTE",
        y="tasa_muertes",
        title="Tasa de registros fatales por mes del año",
        markers=True,
        labels={"MES_ACCIDENTE": "Mes", "tasa_muertes": "Tasa CON MUERTOS"},
        color_discrete_sequence=[ROJO],
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(estilo_figura(fig, 360), use_container_width=True)

    st.markdown(
        """
        **Lecturas rápidas**: la motocicleta domina el volumen de registros y la
        clase fatal ronda el 5%, lo que justifica métricas orientadas al
        desbalance, como Average Precision y recall de la clase fatal, en lugar
        de exactitud simple.
        """
    )


def seccion_metodologia(resumen: dict) -> None:
    st.header("Metodología")

    pasos = [
        ("1 · Limpieza", "Normalización de texto, fechas y edades; trazabilidad con hash por registro."),
        ("2 · EDA", "Distribuciones, calidad del dato y definición del objetivo binario."),
        ("3 · Tuning temporal", "Cortes 2022→2023 y 2023→2024; objetivo: Average Precision con recall mínimo 40%."),
        ("4 · Selección", "Validación rolling-origin por familia; gana el mejor objetivo operacional medio."),
        ("5 · Calibración", "Sigmoid vs isotónica ajustadas solo con train; se elige por Brier en 2025."),
        ("6 · Evaluación 2026", "Prueba final intocada + umbral operacional por costo y capacidad de alertas."),
    ]
    for inicio in range(0, len(pasos), 3):
        columnas = st.columns(3)
        for columna, (titulo, descripcion) in zip(columnas, pasos[inicio:inicio + 3]):
            with columna:
                st.markdown(
                    f"""
                    <div style="background:{CREMA};border-radius:10px;padding:14px;
                                min-height:130px;border-top:4px solid {NARANJA};
                                margin-bottom:12px;">
                    <b style="color:{AZUL_NOCHE};">{titulo}</b>
                    <p style="font-size:0.82rem;color:{AZUL_NOCHE};margin-top:6px;">
                    {descripcion}</p></div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("Búsqueda de hiperparámetros con validación temporal")
    tuning = cargar_csv("tuning_resumen.csv")
    tuning["familia"] = tuning["familia"].map(NOMBRES_MODELOS).fillna(tuning["familia"])
    fig = px.strip(
        tuning,
        x="familia",
        y="mean_average_precision",
        color="familia",
        hover_data=["candidato", "mean_recall"],
        title="Average Precision medio por candidato y familia",
        labels={"mean_average_precision": "AP medio en validaciones históricas", "familia": ""},
        color_discrete_sequence=PALETA,
    )
    st.plotly_chart(estilo_figura(fig, 400), use_container_width=True)

    st.subheader("Estabilidad temporal de las familias seleccionadas")
    rolling = cargar_csv("validacion_rolling.csv")
    rolling["modelo"] = rolling["modelo"].map(NOMBRES_MODELOS).fillna(rolling["modelo"])
    fig = px.line(
        rolling,
        x="validacion",
        y="average_precision",
        color="modelo",
        markers=True,
        title="Average Precision por año de validación, esquema rolling-origin",
        labels={"validacion": "Año de validación", "average_precision": "Average Precision", "modelo": ""},
        color_discrete_sequence=PALETA,
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(estilo_figura(fig, 400), use_container_width=True)

    with st.expander("Configuración ganadora por familia"):
        config = cargar_csv("configuracion_modelos.csv")
        config["modelo"] = config["modelo"].map(NOMBRES_MODELOS).fillna(config["modelo"])
        st.dataframe(config, use_container_width=True, hide_index=True)
        st.caption(
            f"Criterio de selección: {resumen['criterio_seleccion'].replace('_', ' ')}."
        )


def seccion_resultados(resumen: dict) -> None:
    st.header("Resultados del modelo")

    benchmark = cargar_csv("metricas_benchmark.csv")
    columnas_test = {
        "modelo": "Modelo",
        "test_roc_auc": "ROC AUC",
        "test_average_precision": "Average Precision",
        "test_brier": "Brier",
        "test_recall_muertes": "Recall fatales",
        "test_precision_muertes": "Precisión fatales",
        "test_f1_muertes": "F1 fatales",
        "test_alert_rate": "Tasa de alertas",
    }
    tabla = benchmark[list(columnas_test)].rename(columns=columnas_test)
    tabla["Modelo"] = tabla["Modelo"].map(NOMBRES_MODELOS).fillna(tabla["Modelo"])
    st.subheader("Benchmark en la prueba 2026 con umbral fijado en 2025")
    st.dataframe(
        tabla.style.format(
            {col: "{:.3f}" for col in tabla.columns if col != "Modelo"}
        ).highlight_max(
            subset=["ROC AUC", "Average Precision", "Recall fatales", "F1 fatales"],
            color=CREMA,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader(
        "Modelo final calibrado: "
        + NOMBRES_MODELOS.get(resumen["mejor_modelo"], resumen["mejor_modelo"])
    )
    roc = cargar_csv("curva_roc.csv")
    pr = cargar_csv("curva_pr.csv")
    calibracion = cargar_csv("curva_calibracion.csv")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines", name="Modelo",
            line=dict(color=NARANJA, width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Azar",
            line=dict(color=AZUL_NOCHE, dash="dash"),
        ))
        fig.update_layout(
            title=f"ROC · AUC = {resumen['metricas_test_2026']['roc_auc']:.3f}",
            xaxis_title="Tasa de falsos positivos",
            yaxis_title="Tasa de verdaderos positivos",
        )
        st.plotly_chart(estilo_figura(fig, 400), use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines", name="Modelo",
            line=dict(color=ROJO, width=3),
        ))
        fig.add_hline(
            y=resumen["prevalencia_muertes_test"],
            line_dash="dash", line_color=AZUL_NOCHE,
            annotation_text="Prevalencia (línea base)",
        )
        fig.update_layout(
            title=f"Precision-Recall · AP = {resumen['metricas_test_2026']['average_precision']:.3f}",
            xaxis_title="Recall", yaxis_title="Precisión",
        )
        st.plotly_chart(estilo_figura(fig, 400), use_container_width=True)
    with col3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
            line=dict(color=AZUL_NOCHE, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=calibracion["probabilidad_predicha"],
            y=calibracion["frecuencia_observada"],
            mode="lines+markers", name="Modelo calibrado",
            line=dict(color=NARANJA, width=3),
        ))
        fig.update_layout(
            title=f"Calibración · Brier = {resumen['metricas_test_2026']['brier']:.3f}",
            xaxis_title="Probabilidad predicha",
            yaxis_title="Frecuencia observada",
        )
        st.plotly_chart(estilo_figura(fig, 400), use_container_width=True)

    st.divider()
    st.subheader("Explora el umbral operacional")
    umbrales = cargar_csv("tabla_umbrales.csv")
    # El rango útil depende del modelo: las probabilidades calibradas se
    # concentran cerca de la prevalencia, así que el slider se acota a la zona
    # donde de verdad cambian las métricas.
    umbral_maximo = float(min(1.0, max(0.2, resumen["umbral_mejor_modelo"] * 4)))
    umbral_elegido = st.slider(
        "Umbral de probabilidad para generar alerta",
        min_value=0.0, max_value=umbral_maximo,
        value=float(resumen["umbral_mejor_modelo"]), step=0.001,
        help="El notebook fija el umbral minimizando el costo esperado en 2025 "
             "con una capacidad máxima de alertas del 10%.",
    )
    umbrales = umbrales[umbrales["threshold"] <= umbral_maximo]
    fila = umbrales.iloc[(umbrales["threshold"] - umbral_elegido).abs().idxmin()]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall de fatales", f"{fila['recall']:.1%}")
    col2.metric("Precisión de alertas", f"{fila['precision']:.1%}")
    col3.metric("Tasa de alertas", f"{fila['alert_rate']:.1%}")
    col4.metric("Costo esperado", f"{fila['expected_cost']:.3f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=umbrales["threshold"], y=umbrales["recall"],
        name="Recall", line=dict(color=ROJO, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=umbrales["threshold"], y=umbrales["precision"],
        name="Precisión", line=dict(color=AZUL_NOCHE, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=umbrales["threshold"], y=umbrales["alert_rate"],
        name="Tasa de alertas", line=dict(color=AMBAR, width=2.5),
    ))
    fig.add_vline(x=umbral_elegido, line_dash="dash", line_color=NARANJA)
    fig.update_layout(
        title="Métricas en validación 2025 según el umbral",
        xaxis_title="Umbral", yaxis_title="Valor",
    )
    st.plotly_chart(estilo_figura(fig, 420), use_container_width=True)

    if resumen["metricas_test_2026"]["alert_rate"] > resumen["capacidad_maxima_alertas_validacion"]:
        st.warning(
            "La tasa de alertas observada en 2026 supera la capacidad definida en "
            "validación: hay indicios de *drift* y el umbral debería revisarse "
            "antes de operar."
        )


def seccion_territorio() -> None:
    st.header("Territorios y clusters")

    ranking = cargar_csv("ranking_territorial.csv")
    clusters = cargar_csv("perfiles_cluster.csv")
    detalle = cargar_csv("perfil_detalle_cluster.csv")

    st.subheader("Priorización territorial con tasas suavizadas e IC 95%")
    st.caption(
        "Solo territorios con al menos 30 registros. La tasa se suaviza hacia la "
        "media global para no priorizar municipios pequeños por azar; el score "
        "combina tasa suavizada y volumen."
    )
    deptos = ["Todos"] + sorted(ranking["DEPARTAMENTO_ACCIDENTE"].dropna().unique())
    depto_filtro = st.selectbox("Departamento", deptos)
    visibles = ranking[ranking["volumen_suficiente"]].copy()
    if depto_filtro != "Todos":
        visibles = visibles[visibles["DEPARTAMENTO_ACCIDENTE"] == depto_filtro]
    top = visibles.head(15).sort_values("prioridad_score")
    top["municipio"] = top["MUNICIPIO_ACCIDENTE"] + " (" + top["DEPARTAMENTO_ACCIDENTE"].str[:12] + ")"
    fig = go.Figure(go.Bar(
        x=top["tasa_suavizada"],
        y=top["municipio"],
        orientation="h",
        marker_color=NARANJA,
        error_x=dict(
            type="data",
            symmetric=False,
            array=top["tasa_ic95_superior"] - top["tasa_suavizada"],
            arrayminus=(top["tasa_suavizada"] - top["tasa_ic95_inferior"]).clip(lower=0),
            color=AZUL_NOCHE,
        ),
        customdata=top[["unidades", "muertes"]],
        hovertemplate="%{y}<br>Tasa suavizada: %{x:.2%}<br>"
                      "Registros: %{customdata[0]:,} · Fatales: %{customdata[1]:,}"
                      "<extra></extra>",
    ))
    fig.update_layout(
        title="Top 15 municipios por score de prioridad",
        xaxis_title="Tasa suavizada de registros fatales",
        xaxis_tickformat=".1%",
    )
    st.plotly_chart(estilo_figura(fig, 480), use_container_width=True)

    with st.expander("Tabla completa del ranking"):
        st.dataframe(
            visibles.head(100).round(4), use_container_width=True, hide_index=True
        )

    st.divider()
    st.subheader("Clusters de perfiles departamento × tipo de vehículo")
    st.caption(
        "KMeans sobre perfiles agregados; la gravedad NO participa como feature, "
        "solo describe el resultado. k se elige por silhouette."
    )
    col1, col2 = st.columns([2, 3])
    with col1:
        fig = px.bar(
            clusters.sort_values("proporcion_muertes_ponderada"),
            x="proporcion_muertes_ponderada",
            y=clusters.sort_values("proporcion_muertes_ponderada")["cluster"].astype(str),
            orientation="h",
            labels={"proporcion_muertes_ponderada": "Proporción fatal observada", "y": "Cluster"},
            title="Severidad descriptiva por cluster",
            color_discrete_sequence=[ROJO],
        )
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(estilo_figura(fig, 420), use_container_width=True)
    with col2:
        muestra = detalle[detalle["unidades"] >= 50]
        fig = px.scatter(
            muestra,
            x="edad_mediana",
            y="tasa_muertes",
            size="unidades",
            color=muestra["cluster"].astype(str),
            hover_data=["DEPARTAMENTO_ACCIDENTE", "TIPO_VEHICULO"],
            title="Perfiles con al menos 50 registros: antigüedad vs tasa fatal",
            labels={
                "edad_mediana": "Antigüedad mediana del vehículo (años)",
                "tasa_muertes": "Tasa fatal observada",
                "color": "Cluster",
            },
            color_discrete_sequence=PALETA,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(estilo_figura(fig, 420), use_container_width=True)

    st.dataframe(clusters.round(4), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def resumen_departamentos(prior_strength: float, tasa_global: float) -> pd.DataFrame:
    """Agrega registros, fatales y desempeño 2026 por departamento.

    Usa `perfil_detalle_cluster.csv` porque cubre todos los registros (el CSV de
    EDA por departamento solo trae los 15 con más volumen). La tasa se suaviza
    hacia la media nacional con el mismo prior del ranking municipal.
    """
    detalle = cargar_csv("perfil_detalle_cluster.csv")
    detalle = detalle.assign(fatales=detalle["unidades"] * detalle["tasa_muertes"])
    deptos = (
        detalle.groupby("DEPARTAMENTO_ACCIDENTE")[["unidades", "fatales"]]
        .sum()
        .reset_index()
    )
    deptos["fatales"] = deptos["fatales"].round().astype(int)
    deptos["tasa_observada"] = deptos["fatales"] / deptos["unidades"]
    deptos["tasa_suavizada"] = (
        deptos["fatales"] + prior_strength * tasa_global
    ) / (deptos["unidades"] + prior_strength)
    deptos["puesto"] = deptos["tasa_suavizada"].rank(ascending=False, method="min").astype(int)

    # Desempeño del modelo en la prueba 2026, por departamento
    pred = cargar_csv("predicciones_test_2026.csv")
    pred = pred.assign(fatal=(pred["GRAVEDAD_ACCIDENTE"] == "CON MUERTOS").astype(int))
    pred["alerta_fatal"] = pred["alerta"] * pred["fatal"]
    test = pred.groupby("DEPARTAMENTO_ACCIDENTE").agg(
        registros_2026=("fatal", "size"),
        fatales_2026=("fatal", "sum"),
        alertas_2026=("alerta", "sum"),
        alertas_correctas_2026=("alerta_fatal", "sum"),
    ).reset_index()
    deptos = deptos.merge(test, on="DEPARTAMENTO_ACCIDENTE", how="left")
    return deptos.sort_values("puesto")


def centro_departamento(geojson: dict, nombre: str) -> tuple[float, float]:
    """Centro aproximado (lon, lat) del polígono más grande del departamento."""
    for feature in geojson["features"]:
        if feature["id"] != nombre:
            continue
        geometria = feature["geometry"]
        poligonos = (
            [geometria["coordinates"]] if geometria["type"] == "Polygon"
            else geometria["coordinates"]
        )
        anillo = max((p[0] for p in poligonos), key=len)
        lons = [punto[0] for punto in anillo]
        lats = [punto[1] for punto in anillo]
        return (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    return -74.0, 4.6


def tarjetas_html(tarjetas: list[tuple[str, str, str, str]]) -> str:
    """Fila de tarjetas compactas: (título, valor, detalle, color del borde)."""
    piezas = []
    for titulo, valor, detalle, color in tarjetas:
        piezas.append(
            f"""<div style="flex:1 1 140px;background:{CREMA}55;border-radius:10px;
                        padding:10px 14px;border-top:3px solid {color};">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.04em;
                            color:{AZUL_NOCHE}AA;">{titulo}</div>
                <div style="font-size:1.35rem;font-weight:700;color:{AZUL_NOCHE};
                            line-height:1.25;">{valor}</div>
                <div style="font-size:0.72rem;color:{AZUL_NOCHE}99;min-height:1em;">{detalle}</div>
            </div>"""
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:4px 0 6px 0;">'
        + "".join(piezas)
        + "</div>"
    )


def seccion_mapa(resumen: dict) -> None:
    st.markdown(
        f"<h1 style='text-align:center;color:{AZUL_NOCHE};margin-bottom:0;'>"
        "Siniestralidad vial en Colombia</h1>"
        f"<p style='text-align:center;color:{NARANJA};font-weight:600;margin-top:2px;'>"
        "Registros de vehículos en siniestros 2022–2026, por departamento</p>",
        unsafe_allow_html=True,
    )

    tasa_global = resumen["tasa_global_muertes"]
    deptos = resumen_departamentos(resumen["prior_strength_tasa_territorial"], tasa_global)
    geojson = cargar_json("colombia_departamentos.geojson")
    nombres = deptos["DEPARTAMENTO_ACCIDENTE"].tolist()

    metricas = {
        "Tasa fatal suavizada": ("tasa_suavizada", ".0%"),
        "Registros": ("unidades", "~s"),
        "Registros fatales": ("fatales", "~s"),
    }
    control1, control2 = st.columns([3, 2], vertical_alignment="bottom")
    with control1:
        metrica = st.segmented_control(
            "Colorear por", list(metricas), default="Tasa fatal suavizada",
            key="mapa_metrica",
        ) or "Tasa fatal suavizada"
    with control2:
        seleccionado = st.selectbox(
            "Departamento (ordenados por tasa suavizada)",
            nombres,
            key="mapa_departamento",
            format_func=str.title,
        )
    columna, formato = metricas[metrica]

    # Tarjetas compactas del departamento seleccionado
    fila = deptos.set_index("DEPARTAMENTO_ACCIDENTE").loc[seleccionado]
    diferencia = (fila["tasa_suavizada"] - tasa_global) * 100
    tarjetas = [
        ("Departamento", seleccionado.title(), f"Puesto {int(fila['puesto'])} de {len(deptos)}", AZUL_NOCHE),
        ("Registros", f"{fila['unidades']:,.0f}", "2022–2026", AMBAR),
        ("Fatales", f"{fila['fatales']:,.0f}", "registros con muertos", ROJO),
        ("Tasa suavizada", f"{fila['tasa_suavizada']:.1%}",
         f"{diferencia:+.1f} pp vs. nacional ({tasa_global:.1%})",
         ROJO if diferencia > 0 else AZUL_NOCHE),
    ]
    if pd.notna(fila["registros_2026"]) and fila["registros_2026"] > 0:
        alertas = fila["alertas_2026"] / fila["registros_2026"]
        detectados = (
            f"{fila['alertas_correctas_2026'] / fila['fatales_2026']:.0%}"
            if fila["fatales_2026"] else "—"
        )
        tarjetas += [
            ("Alertas 2026", f"{alertas:.1%}", f"de {fila['registros_2026']:,.0f} registros", NARANJA),
            ("Fatales detectados 2026", detectados, f"de {fila['fatales_2026']:,.0f} fatales", NARANJA),
        ]
    st.markdown(tarjetas_html(tarjetas), unsafe_allow_html=True)

    # Mapa principal
    lon, lat = centro_departamento(geojson, seleccionado)
    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=deptos["DEPARTAMENTO_ACCIDENTE"],
        z=deptos[columna],
        colorscale=[[0, "#F7F3E8"], [0.25, AMBAR], [0.6, NARANJA], [1, ROJO]],
        marker_line_color="white",
        marker_line_width=0.8,
        colorbar=dict(
            title=dict(text=metrica, side="top"),
            orientation="h",
            x=0.5, xanchor="center",
            y=0.0, yanchor="top",
            len=0.45, thickness=12,
            tickformat=formato,
            outlinewidth=0,
        ),
        customdata=deptos[["unidades", "fatales", "tasa_suavizada", "puesto"]],
        text=deptos["DEPARTAMENTO_ACCIDENTE"].str.title(),
        hovertemplate="<b>%{text}</b><br>"
                      "Registros: %{customdata[0]:,}<br>"
                      "Fatales: %{customdata[1]:,}<br>"
                      "Tasa suavizada: %{customdata[2]:.2%}<br>"
                      "Puesto nacional: %{customdata[3]}<extra></extra>",
    ))
    # Departamentos sin registros, en gris
    sin_datos = sorted({f["id"] for f in geojson["features"]} - set(nombres))
    if sin_datos:
        fig.add_trace(go.Choropleth(
            geojson=geojson,
            locations=sin_datos,
            z=[0] * len(sin_datos),
            colorscale=[[0, GRIS_REJILLA], [1, GRIS_REJILLA]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.8,
            text=[n.title() for n in sin_datos],
            hovertemplate="<b>%{text}</b><br>Sin registros en la fuente<extra></extra>",
        ))
    # Contorno y etiqueta del departamento seleccionado
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=[seleccionado],
        z=[1],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False,
        marker_line_color=AZUL_NOCHE,
        marker_line_width=3.5,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lon=[lon], lat=[lat],
        mode="markers+text",
        marker=dict(size=9, color=AZUL_NOCHE, line=dict(color="white", width=2)),
        text=[f"<b>{seleccionado.title()}</b>"],
        textposition="top center",
        textfont=dict(
            size=16, color=AZUL_NOCHE,
            shadow="0 0 3px white, 0 0 3px white, 0 0 6px white",
        ),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=860,
        margin=dict(l=0, r=0, t=0, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=AZUL_NOCHE),
        hoverlabel=dict(bgcolor="white", font_size=13),
        dragmode=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "La tasa suavizada acerca a la media nacional a los departamentos con pocos "
        "registros, para no exagerar diferencias por azar. Son tasas descriptivas: "
        "sin datos de tráfico no miden riesgo causal, y la calidad del reporte varía "
        "entre territorios."
        + (f" Sin registros en la fuente: {', '.join(sin_datos).title()}." if sin_datos else "")
    )

    # Detalle del departamento
    st.divider()
    st.subheader(f"Detalle de {seleccionado.title()}")
    col1, col2 = st.columns(2)
    with col1:
        detalle = cargar_csv("perfil_detalle_cluster.csv")
        tipos = detalle[detalle["DEPARTAMENTO_ACCIDENTE"] == seleccionado]
        tipos = tipos.sort_values("unidades", ascending=False).head(8)
        fig = go.Figure(go.Bar(
            x=tipos["tasa_muertes"],
            y=tipos["TIPO_VEHICULO"],
            orientation="h",
            marker_color=ROJO,
            customdata=tipos[["unidades", "cluster"]],
            hovertemplate="%{y}<br>Tasa fatal: %{x:.1%}<br>"
                          "Registros: %{customdata[0]:,}<br>"
                          "Cluster: %{customdata[1]}<extra></extra>",
        ))
        fig.add_vline(x=tasa_global, line_dash="dot", line_color=AZUL_NOCHE,
                      annotation_text="Media nacional", annotation_position="top")
        fig.update_layout(
            title="Tasa fatal por tipo de vehículo (8 más frecuentes)",
            xaxis_tickformat=".0%",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(estilo_figura(fig, 400), width="stretch")
    with col2:
        st.markdown("**Municipios con mayor prioridad**")
        ranking = cargar_csv("ranking_territorial.csv")
        municipios = ranking[
            ranking["volumen_suficiente"]
            & (ranking["DEPARTAMENTO_ACCIDENTE"] == seleccionado)
        ].head(10)
        if municipios.empty:
            st.caption("Ningún municipio con volumen suficiente para el ranking.")
        else:
            st.dataframe(
                municipios[[
                    "ranking_operativo", "MUNICIPIO_ACCIDENTE", "unidades",
                    "muertes", "tasa_suavizada",
                ]],
                column_config={
                    "ranking_operativo": st.column_config.NumberColumn("Puesto nacional", format="%d"),
                    "MUNICIPIO_ACCIDENTE": "Municipio",
                    "unidades": st.column_config.NumberColumn("Registros", format="%d"),
                    "muertes": st.column_config.NumberColumn("Fatales", format="%d"),
                    "tasa_suavizada": st.column_config.ProgressColumn(
                        "Tasa suavizada", format="percent", min_value=0, max_value=1,
                    ),
                },
                hide_index=True,
                width="stretch",
            )


def seccion_simulador(resumen: dict) -> None:
    st.header("Simulador de scoring")
    st.caption(
        "Usa el modelo final calibrado exportado por el notebook. El score es la "
        "probabilidad de que un registro como el descrito corresponda a un "
        "siniestro CON MUERTOS; no predice la ocurrencia de accidentes."
    )

    paquete = cargar_modelo()
    opciones = cargar_json("opciones_simulador.json")
    modelo = paquete["model"]
    umbral = paquete["threshold"]

    with st.form("simulador"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tipo = st.selectbox("Tipo de vehículo", opciones["tipos_vehiculo"],
                                index=opciones["tipos_vehiculo"].index("MOTOCICLETA")
                                if "MOTOCICLETA" in opciones["tipos_vehiculo"] else 0)
            marca = st.selectbox("Marca", opciones["marcas"])
            modelo_anio = st.number_input(
                "Año modelo del vehículo",
                min_value=opciones["modelo_vehiculo_min"],
                max_value=opciones["modelo_vehiculo_max"],
                value=2015,
            )
        with col2:
            departamentos = sorted(opciones["municipios_por_departamento"])
            departamento = st.selectbox("Departamento", departamentos)
            municipio = st.selectbox(
                "Municipio", opciones["municipios_por_departamento"][departamento]
            )
            autoridad = st.selectbox("Autoridad de tránsito", opciones["autoridades"])
        with col3:
            mes = st.selectbox("Mes del siniestro", list(range(1, 13)), index=5)
            st.markdown(
                f"<div style='background:{CREMA};border-radius:8px;padding:10px;"
                f"font-size:0.85rem;color:{AZUL_NOCHE};'>Año del escenario: "
                f"<b>{opciones['anio_prueba']}</b>, el de la prueba final. "
                f"Umbral de alerta: <b>{umbral:.3f}</b>.</div>",
                unsafe_allow_html=True,
            )
        enviar = st.form_submit_button("Calcular score", type="primary")

    if not enviar:
        return

    anio = opciones["anio_prueba"]
    antiguedad = max(0, min(100, anio - int(modelo_anio)))
    entrada = pd.DataFrame([{
        "MARCA_VEHICULO": marca,
        "TIPO_VEHICULO": tipo,
        "DEPARTAMENTO_ACCIDENTE": departamento,
        "MUNICIPIO_ACCIDENTE": municipio,
        "AUTORIDAD_DE_TRANSITO": autoridad,
        "TIPO_DEPARTAMENTO": f"{tipo}__{departamento}",
        "MODELO_VEHICULO": float(modelo_anio),
        "ANTIGUEDAD_REPORTADA": float(antiguedad),
        "ANIO_ACCIDENTE": float(anio),
        "MES_ACCIDENTE": float(mes),
        "MODELO_ADELANTADO": int(modelo_anio > anio),
        "VEHICULO_NUEVO": int(antiguedad <= 2),
        "NUM_VEHICULOS": 1,
        "NUM_TIPOS_VEHICULO": 1,
        "NUM_MOTOS": int(tipo == "MOTOCICLETA"),
        "NUM_PESADOS": int(tipo in TIPOS_PESADOS),
    }])
    probabilidad = float(
        modelo.predict_proba(entrada[paquete["categorical"] + paquete["numeric"]])[0, 1]
    )
    es_alerta = probabilidad >= umbral

    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric("Probabilidad calibrada de siniestro fatal", f"{probabilidad:.1%}")
        if es_alerta:
            st.error(f"ALERTA: score por encima del umbral {umbral:.3f}")
        else:
            st.success(f"Sin alerta: score por debajo del umbral {umbral:.3f}")
        st.caption(
            f"Prevalencia de referencia en la prueba 2026: "
            f"{resumen['prevalencia_muertes_test']:.1%}."
        )
    with col2:
        # El tope del gauge se adapta al umbral: con probabilidades calibradas
        # cercanas a la prevalencia, un rango 0-100% dejaría todo invisible.
        tope_gauge = max(25.0, umbral * 400, probabilidad * 120)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilidad * 100,
            number={"suffix": "%", "font": {"color": AZUL_NOCHE}},
            gauge={
                "axis": {"range": [0, tope_gauge], "ticksuffix": "%"},
                "bar": {"color": ROJO if es_alerta else NARANJA},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, umbral * 100], "color": CREMA},
                    {"range": [umbral * 100, tope_gauge], "color": "#F6D5C3"},
                ],
                "threshold": {
                    "line": {"color": AZUL_NOCHE, "width": 3},
                    "thickness": 0.9,
                    "value": umbral * 100,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(l=30, r=30, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

FUENTES_AGENTE = (
    "resumen.json",
    "metricas_benchmark.csv",
    "eda_anio_gravedad.csv",
    "eda_tipo_vehiculo.csv",
    "eda_departamento.csv",
    "eda_mes.csv",
    "ranking_territorial.csv",
    "perfiles_cluster.csv",
)


def consultar_tablero(fuente: str, departamento: str | None = None) -> str:
    """Consulta los resultados del tablero sin modificar los datos."""

    if fuente not in FUENTES_AGENTE:
        return "Fuente no disponible."

    if fuente == "resumen.json":
        return json.dumps(cargar_json(fuente), ensure_ascii=False)

    tabla = cargar_csv(fuente)

    if fuente == "ranking_territorial.csv":
        tabla = tabla[tabla["volumen_suficiente"]]

        if departamento:
            nombre = normalize("NFKD", departamento.strip().upper())
            nombre = nombre.encode("ascii", "ignore").decode("ascii")
            tabla = tabla[tabla["DEPARTAMENTO_ACCIDENTE"] == nombre]

        tabla = tabla.head(15)

    return tabla.to_json(orient="records", force_ascii=False)



def responder_agente(cliente: OpenAI, mensajes: list[dict]) -> str:
    herramientas = [
        {
            "type": "function",
            "function": {
                "name": "consultar_tablero",
                "description": (
                    "Consulta los archivos de resultados del tablero "
                    "para responder preguntas sobre accidentes, "
                    "metricas, territorios y clusters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fuente": {
                            "type": "string",
                            "enum": list(FUENTES_AGENTE),
                            "description": "Archivo de resultados a consultar."
                        },
                        "departamento": {
                            "type": ["string", "null"],
                            "description": (
                                "Departamento para filtrar el ranking "
                                "territorial, o null si no aplica."
                            )
                        }
                    },
                    "required": ["fuente", "departamento"],
                    "additionalProperties": False
                }
            }
        }
    ]

    conversacion = [
        {
            "role": "system",
            "content": (
                "Eres un agente de inteligencia artificial integrado en un "
                "tablero de siniestros viales de Colombia. "

                "Tu tarea es responder preguntas utilizando los archivos "
                "del proyecto mediante la herramienta consultar_tablero. "

                "Cuando el usuario pregunte por m├®tricas, resultados, "
                "accidentes, a├▒os, departamentos o rankings, debes consultar "
                "primero los archivos correspondientes. "

                "Selecciona t├║ mismo la fuente adecuada. "
                "No le pidas al usuario nombres de archivos. "

                "Para preguntas sobre recall, precision, F1, ROC AUC "
                "o desempe├▒o del modelo, consulta primero resumen.json "
                "y, si es necesario, metricas_benchmark.csv. "

                "El proyecto incluye informaci├│n correspondiente a 2026. "
                "No afirmes que los datos de ese a├▒o no est├ín disponibles "
                "sin haber consultado los archivos. "

                "Si el usuario no menciona un departamento, utiliza null. "

                "Responde siempre en espa├▒ol, de manera clara y sencilla. "
                "No inventes cifras. Si los archivos no contienen la "
                "informaci├│n solicitada, expl├¡calo."
            )
        },
        *mensajes
    ]

    for _ in range(4):
        respuesta = cliente.chat.completions.create(
            model="nvidia/nemotron-3.5-bolt-e4b",
            messages=conversacion,
            tools=herramientas,
            tool_choice="auto",
            temperature=1.0,
            max_tokens=1500,
        )

        mensaje = respuesta.choices[0].message
        conversacion.append(mensaje.model_dump(exclude_none=True))

        if not mensaje.tool_calls:
            return mensaje.content or "No se obtuvo una respuesta."

        for llamada in mensaje.tool_calls:
            if llamada.function.name != "consultar_tablero":
                resultado = "Herramienta no disponible."
            else:
                try:
                    argumentos = json.loads(llamada.function.arguments)
                    resultado = consultar_tablero(
                        fuente=argumentos["fuente"],
                        departamento=argumentos.get("departamento"),
                        )
                    print(f"[AGENTE NVIDIA] Consultando archivo: {argumentos['fuente']}"),
                except (ValueError, KeyError, TypeError) as error:
                    resultado = f"Error al consultar los datos: {error}"

            conversacion.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "content": str(resultado)
                }
            )

    return "No fue posible completar la consulta en este momento."



def seccion_agente() -> None:
    st.header("Agente de IA")
    st.caption(
        "Pregunta sobre los datos, las metricas "
        "o el ranking territorial del tablero."
    )

    if not os.getenv("NVIDIA_API_KEY"):
        st.info("Configura NVIDIA_API_KEY para activar el agente.")
        return

    mensajes = st.session_state.setdefault("mensajes_agente", [])

    for mensaje in mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input(
        "¿Cuál fue el recall de fatales en 2026?",
        max_chars=1000,
    )

    if not pregunta or not pregunta.strip():
        return

    usuario = {"role": "user", "content": pregunta.strip()}

    with st.chat_message("user"):
        st.markdown(usuario["content"])

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando el tablero..."):
                with OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.environ["NVIDIA_API_KEY"],
                    timeout=60.0,
                ) as cliente:
                    respuesta = responder_agente(cliente, [*mensajes[-8:], usuario])
        except (OpenAIError, OSError, ValueError, KeyError) as error:
            st.error("No se pudo completar la consulta del agente.")
            st.code(f"{type(error).__name__}: {error}")
            return

        if not respuesta:
            st.error("El agente no devolvi├│ una respuesta.")
            return

        st.markdown(respuesta)

    mensajes.extend([
        usuario,
        {"role": "assistant", "content": respuesta},
    ])

def main() -> None:
    st.set_page_config(
        page_title="Siniestralidad vial · Ley 2251",
        layout="wide",
    )

    if not ARTIFACTS.exists() or not (ARTIFACTS / "resumen.json").exists():
        st.error(
            "No se encontró la carpeta `artifacts/`. Ejecuta primero el notebook "
            "`main.ipynb` de inicio a fin para generar los artefactos del modelo."
        )
        st.stop()

    resumen = cargar_json("resumen.json")

    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{AZUL_NOCHE};margin-bottom:0;'>Siniestralidad vial</h2>"
            f"<p style='color:{NARANJA};font-weight:600;margin-top:2px;'>"
            f"Ley 2251 de 2022 · Colombia</p>",
            unsafe_allow_html=True,
        )
        seccion = st.radio(
            "Secciones",
            [
                "Resumen",
                "Datos y EDA",
                "Metodología",
                "Resultados",
                "Territorios y clusters",
                "Mapa de Colombia",
                "Simulador",
                "Agente de IA"
            ],
        )
        st.divider()
        st.caption(
            "Fuente: RUNT en datos.gov.co, conjunto 6jmc-vaxk. "
            "Resultados trazables a `main.ipynb`; la app no reentrena modelos."
        )

    if seccion == "Resumen":
        seccion_resumen(resumen)
    elif seccion == "Datos y EDA":
        seccion_datos()
    elif seccion == "Metodología":
        seccion_metodologia(resumen)
    elif seccion == "Resultados":
        seccion_resultados(resumen)
    elif seccion == "Territorios y clusters":
        seccion_territorio()
    elif seccion == "Mapa de Colombia":
        seccion_mapa(resumen)
    elif seccion == "Agente de IA":
        seccion_agente()
    else:
        seccion_simulador(resumen)


if __name__ == "__main__":
    main()
