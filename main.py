"""Tablero Streamlit del proyecto de siniestralidad vial (Ley 2251 de 2022).

La aplicación NO reentrena nada: lee los artefactos que exporta `main.ipynb`
(carpeta `artifacts/`), de modo que cada número del tablero es trazable al
notebook. Para desplegar en producción basta con este repositorio.

Ejecución local:
    streamlit run main.py
"""

import json
from pathlib import Path

import joblib
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
                "Simulador",
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
    else:
        seccion_simulador(resumen)


if __name__ == "__main__":
    main()
