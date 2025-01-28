import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, f_oneway
import streamlit as st

# Page Configuration

# Page Configuration
st.set_page_config(
    page_title="Validación UV-Vis",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Escudo_CUCEI.svg/424px-Escudo_CUCEI.svg.png",
    layout="wide"  # Centers all content
)
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for title and footer
st.markdown("""
    <style>
    /* Title container styling */
    .title-container {
        display: flex;
        justify-content: ; /* Center the title and logo */
        align-items: center;
        gap: 10px;
        margin-bottom: 20px; /* Spacing below title */
    }
    .title-container img {
        width: 250px; /* Adjust icon size */
        height: auto;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.05); /* Light background */
        color: gray;
        text-align: center;
        padding: 5px 15px;
        font-size: small;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# Title with Icon
st.markdown("""
    <div class="title-container">
        <img src="https://practicas.cucei.udg.mx/dist/imagenes/logo_cucei_blanco.png" alt="CUCEI Logo">
        <h1>Validación de Métodos Analíticos - Espectrofotometría UV-Vis</h1>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Hecho por Luis Angel Cardenas Medina
    </div>
""", unsafe_allow_html=True)


# Módulos disponibles
modulo = st.sidebar.selectbox(
    "Selecciona el módulo de análisis:",
    [
        "Linealidad y Rango",
        "Límites de Detección y Cuantificación",
        "Exactitud (Recuperación)",
        "Precisión (Repetibilidad e Intermedia)",
        "Robustez"
    ]
)

# Funciones generales

def previsualizar_datos(datos):
    """Previsualiza los datos cargados en la interfaz."""
    st.write("### Vista previa de los datos cargados:")
    st.write(datos.head())
    st.write(f"**Número de filas:** {datos.shape[0]}, **Número de columnas:** {datos.shape[1]}")

def validar_columnas(datos, columnas_necesarias):
    """Valida que las columnas requeridas estén presentes en el archivo subido."""
    columnas_faltantes = [col for col in columnas_necesarias if col not in datos.columns]
    if columnas_faltantes:
        st.error(f"El archivo no contiene las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
        return False
    return True

def calcular_regresion(datos_dia):
    """Calcula la regresión lineal y los parámetros LOD y LOQ."""
    x, y = datos_dia['Concentración'], datos_dia['Absorbancia']
    regresion = linregress(x, y)
    slope, intercept = regresion.slope, regresion.intercept
    residuals = y - (slope * x + intercept)
    std_dev = residuals.std()
    lod = (3.3 * std_dev) / slope if slope != 0 else None
    loq = (10 * std_dev) / slope if slope != 0 else None
    return slope, intercept, lod, loq, std_dev

def graficar_curva_calibracion_streamlit(datos):
    """Grafica la curva de calibración con líneas indicativas de LOD y LOQ para cada día en Streamlit."""
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        st.error("No se encontraron datos de tipo 'Estándar' para realizar el cálculo.")
        return

    dias_unicos = datos_estandar['Día'].unique()
    for dia in dias_unicos:
        st.subheader(f"Curva de Calibración para el Día {dia}")
        datos_dia = datos_estandar[datos_estandar['Día'] == dia]
        if len(datos_dia) < 2:
            st.warning(f"No hay suficientes datos para realizar la regresión en el Día {dia}. Se requieren al menos 2 puntos.")
            continue

        slope, intercept, lod, loq, std_dev = calcular_regresion(datos_dia)
        if slope is None:
            st.error(f"La pendiente de la regresión es 0 en el Día {dia}, no se pueden calcular LOD y LOQ.")
            continue

        y_pred = slope * datos_dia['Concentración'] + intercept
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(datos_dia['Concentración'], datos_dia['Absorbancia'], label="Datos experimentales", color='black')
        ax.plot(datos_dia['Concentración'], y_pred, color='red', label="Curva de calibración")
        ax.axvline(x=lod, color='green', linestyle='--', label=f"LOD ({lod:.4f})")
        ax.axvline(x=loq, color='blue', linestyle='--', label=f"LOQ ({loq:.4f})")
        ax.set_xlabel("Concentración")
        ax.set_ylabel("Absorbancia")
        ax.set_title(f"Curva de Calibración (Día {dia})")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

# Funciones específicas por módulo

def calcular_linealidad(datos):
    """Calcula la linealidad y rango del método considerando agrupación por días."""
    columnas_necesarias = ['Concentración', 'Absorbancia', 'Tipo', 'Día']
    if not validar_columnas(datos, columnas_necesarias):
        return

    for dia, grupo_dia in datos.groupby('Día'):
        st.write(f"### Día {dia}")
        estandar = grupo_dia[grupo_dia['Tipo'] == 'Estándar']
        if estandar.empty:
            st.warning(f"No se encontraron datos de Estándar para el Día {dia}.")
            continue

        estandar_promedio = estandar.groupby('Concentración')['Absorbancia'].mean().reset_index()
        slope, intercept, _, _, _ = calcular_regresion(estandar_promedio)
        r_value, p_value = linregress(estandar_promedio['Concentración'], estandar_promedio['Absorbancia'])[2:4]

        st.write(f"**Día {dia}:**")
        st.write(f"  - **Pendiente (Slope):** {slope:.4f}")
        st.write(f"  - **Intercepto (Intercept):** {intercept:.4f}")
        st.write(f"  - **Coeficiente de determinación (R²):** {r_value**2:.4f}")
        st.write(f"  - **Valor p:** {p_value:.4e}")

        if r_value**2 >= 0.995:
            st.success(f"Cumple con los criterios de linealidad para el Día {dia} (R² ≥ 0.995).")
        else:
            st.error(f"No cumple con los criterios de linealidad para el Día {dia} (R² < 0.995).")

        plt.figure(figsize=(8, 5))
        sns.regplot(x=estandar_promedio['Concentración'], y=estandar_promedio['Absorbancia'], ci=None, line_kws={'color': 'red'})
        plt.title(f"Linealidad Día {dia}: Concentración vs Absorbancia (Estándar Promedio)")
        plt.xlabel("Concentración")
        plt.ylabel("Absorbancia")
        st.pyplot(plt)

        muestra = grupo_dia[grupo_dia['Tipo'] == 'Muestra']
        if not muestra.empty:
            x_muestra = muestra['Absorbancia']
            concentraciones_estimadas = (x_muestra - intercept) / slope
            muestra['Concentración Estimada'] = concentraciones_estimadas

            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=muestra['Concentración Estimada'], y=muestra['Absorbancia'], color='blue', label='Muestra')
            plt.title(f"Concentración Estimada de la Muestra (Día {dia})")
            plt.xlabel("Concentración Estimada")
            plt.ylabel("Absorbancia")
            st.pyplot(plt)

def calcular_lod_loq(datos):
    """Calcula los límites de detección y cuantificación (LOD y LOQ) según el método del ICH."""
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        st.error("No se encontraron datos de tipo 'Estándar' para realizar el cálculo.")
        return

    dias_unicos = datos_estandar['Día'].unique()
    for dia in dias_unicos:
        st.subheader(f"Resultados para el Día {dia}")
        datos_dia = datos_estandar[datos_estandar['Día'] == dia]
        if len(datos_dia) < 2:
            st.warning(f"No hay suficientes datos para realizar la regresión en el Día {dia}. Se requieren al menos 2 puntos.")
            continue

        slope, intercept, lod, loq, std_dev = calcular_regresion(datos_dia)
        if slope is None:
            st.error(f"La pendiente de la regresión es 0 en el Día {dia}, no se pueden calcular LOD y LOQ.")
            continue

        st.write(f"**Pendiente de la regresión:** {slope:.4f}")
        st.write(f"**Intercepto de la regresión:** {intercept:.4f}")
        st.write(f"**Desviación estándar de los residuales:** {std_dev:.4f}")
        st.write(f"**Límite de Detección (LOD):** {lod:.4f}")
        st.write(f"**Límite de Cuantificación (LOQ):** {loq:.4f}")
        st.write(f"**Datos del Día {dia}:**")
        st.dataframe(datos_dia[['Concentración', 'Absorbancia']])

def calcular_precision(datos):
    """Evalúa la precisión siguiendo la guideline ICH Q2 mediante el cálculo del RSD (Relative Standard Deviation)."""
    columnas_necesarias = ['Día', 'Concentración', 'Absorbancia', 'Tipo']
    if not validar_columnas(datos, columnas_necesarias):
        return

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    datos_muestra = datos[datos['Tipo'] == 'Muestra']
    if datos_estandar.empty:
        st.error("No se encontraron datos de estándares en el conjunto de datos.")
        return

    datos_muestra['Concentración Real'] = np.nan
    for dia in datos['Día'].unique():
        estandares_dia = datos_estandar[datos_estandar['Día'] == dia]
        muestras_dia = datos_muestra[datos_muestra['Día'] == dia].copy()
        if estandares_dia.empty:
            st.warning(f"No se encontraron estándares para el día {dia}. Concentraciones no calculadas para este día.")
            continue

        try:
            X = estandares_dia['Absorbancia'].values.reshape(-1, 1)
            y = estandares_dia['Concentración'].values
            modelo = linregress()
            modelo.fit(X, y)
            muestras_dia['Concentración Real'] = modelo.predict(muestras_dia['Absorbancia'].values.reshape(-1, 1))
            datos_muestra.update(muestras_dia)
            pendiente, intercepto = modelo.coef_[0], modelo.intercept_
            st.write(f"**Curva de calibración para el día {dia}:** Concentración = {pendiente:.4f} * Absorbancia + {intercepto:.4f}")
        except Exception as e:
            st.error(f"Error ajustando la curva de calibración para el día {dia}: {e}")

    for tipo, datos_tipo in [('Estándar', datos_estandar), ('Muestra', datos_muestra)]:
        st.subheader(f"**Precisión para tipo: {tipo}**")
        grupos_intraensayo = datos_tipo.groupby(['Día', 'Concentración'])['Absorbancia']
        rsd_intraensayo = grupos_intraensayo.std() / grupos_intraensayo.mean() * 100
        st.write("**RSD por día y concentración (Repetibilidad intraensayo):**")
        st.table(rsd_intraensayo.reset_index().rename(columns={'Absorbancia': 'RSD (%)'}))

        grupos_intermedio = datos_tipo.groupby('Concentración')['Absorbancia']
        rsd_intermedio = grupos_intermedio.std() / grupos_intermedio.mean() * 100
        st.write("**RSD por concentración (Precisión intermedia):**")
        st.table(rsd_intermedio.reset_index().rename(columns={'Absorbancia': 'RSD (%)'}))

        pesos = grupos_intermedio.mean()
        rsd_general = (rsd_intermedio * pesos / pesos.sum()).sum()
        st.write(f"**RSD General (Precisión total) para {tipo}:** {rsd_general:.2f}%")

        if rsd_general <= 2:
            st.success(f"{tipo}: Cumple con los criterios de precisión (RSD ≤ 2%).")
        else:
            st.error(f"{tipo}: No cumple con los criterios de precisión (RSD > 2%).")

    st.write("**Concentraciones reales calculadas para las muestras:**")
    st.table(datos_muestra[['Día', 'Absorbancia', 'Concentración Real']])

def calcular_exactitud(datos):
    """Calcula la exactitud (recuperación) para muestras fortificadas según la guía del ICH Q2."""
    columnas_necesarias = ['Día', 'Concentración Teórica', 'Concentración Medida']
    if not validar_columnas(datos, columnas_necesarias):
        return

    dias = datos['Día'].unique()
    resultados_por_dia = []
    for dia in dias:
        st.subheader(f"Resultados para el Día {dia}")
        datos_dia = datos[datos['Día'] == dia]
        datos_dia['Recuperación (%)'] = (datos_dia['Concentración Medida'] / datos_dia['Concentración Teórica']) * 100
        st.write(f"### Resultados para el Día {dia}")
        st.dataframe(datos_dia[['Concentración Teórica', 'Concentración Medida', 'Recuperación (%)']])

        media_recuperacion = datos_dia['Recuperación (%)'].mean()
        std_recuperacion = datos_dia['Recuperación (%)'].std()
        st.write(f"**Media de Recuperación (%):** {media_recuperacion:.2f}")
        st.write(f"**Desviación Estándar de Recuperación (%):** {std_recuperacion:.2f}")

        rango_aceptable = (98, 102)
        muestras_fuera_rango = datos_dia[
            (datos_dia['Recuperación (%)'] < rango_aceptable[0]) | 
            (datos_dia['Recuperación (%)'] > rango_aceptable[1])
        ]

        if muestras_fuera_rango.empty:
            st.success(f"Todas las muestras fortificadas del Día {dia} tienen porcentajes de recuperación dentro del rango aceptable (98%-102%).")
        else:
            st.warning(f"Se encontraron muestras fortificadas fuera del rango aceptable para el Día {dia} ({rango_aceptable[0]}%-{rango_aceptable[1]}%):")
            st.dataframe(muestras_fuera_rango[['Concentración Teórica', 'Concentración Medida', 'Recuperación (%)']])

        resultados_por_dia.append({
            'Día': dia,
            'Media Recuperación (%)': media_recuperacion,
            'Desviación Estándar (%)': std_recuperacion,
            'Muestras Fuera de Rango': len(muestras_fuera_rango)
        })

    if resultados_por_dia:
        st.subheader("Resumen por Día")
        resumen_df = pd.DataFrame(resultados_por_dia)
        st.dataframe(resumen_df)

def evaluar_robustez(datos):
    """Evalúa la robustez del método analítico mediante ANOVA."""
    columnas_necesarias = ['Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    factores_posibles = ['Día', 'Concentración', 'Tipo']
    factor = st.selectbox("Selecciona el factor a evaluar:", factores_posibles)
    if factor not in datos.columns:
        st.error(f"El factor '{factor}' no está en los datos.")
        return

    grupos = [grupo['Absorbancia'].values for _, grupo in datos.groupby(factor)]
    estadistico, p_value = f_oneway(*grupos)

    st.write(f"**Factor evaluado:** {factor}")
    st.write(f"**Estadístico F:** {estadistico:.4f}")
    st.write(f"**Valor p:** {p_value:.4e}")

    if p_value > 0.05:
        st.success("No hay diferencias significativas (p > 0.05). El método es robusto.")
    else:
        st.error("Hay diferencias significativas (p ≤ 0.05). El método no es robusto.")

    st.write("**Gráfico de caja (Boxplot):**")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=factor, y='Absorbancia', data=datos)
    plt.title(f"Variabilidad de Absorbancia según {factor}")
    st.pyplot(plt)

def evaluar_estabilidad(datos):
    """Evalúa la estabilidad de la solución en el tiempo."""
    columnas_necesarias = ['Tiempo', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    x = datos['Tiempo']
    y = datos['Absorbancia']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    st.write(f"**Pendiente:** {slope:.4f}")
    st.write(f"**Intercepto:** {intercept:.4f}")
    st.write(f"**Coeficiente de determinación (R²):** {r_value**2:.4f}")

    if abs(slope) < 0.01:
        st.success("El método es estable en el tiempo (pendiente cercana a 0).")
    else:
        st.error("El método no es estable en el tiempo (pendiente alejada de 0).")

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=x, y=y, marker='o')
    plt.title("Estabilidad de la Solución")
    plt.xlabel("Tiempo")
    plt.ylabel("Absorbancia")
    st.pyplot(plt)

# Lógica principal para cada módulo
def procesar_archivo(datos, funcion_calculo):
    """Procesa el archivo subido y ejecuta la función de cálculo correspondiente."""
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            funcion_calculo(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

if modulo == "Linealidad y Rango":
    st.header("Análisis de Linealidad y Rango")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Absorbancia:** Valores de absorbancia medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Concentración y Absorbancia:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_linealidad)

elif modulo == "Límites de Detección y Cuantificación":
    st.header("Cálculo de LOD y LOQ")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Ruido:** Datos del ruido del sistema.
        - **Pendiente:** Datos de la pendiente de la curva de calibración.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Ruido y Pendiente:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            calcular_lod_loq(datos_df)
            graficar_curva_calibracion_streamlit(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

elif modulo == "Precisión (Repetibilidad e Intermedia)":
    st.header("Evaluación de Precisión")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Absorbancia:** Datos de absorbancia agrupados por días y repeticiones.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Absorbancia agrupados por Día:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_precision)

elif modulo == "Exactitud (Recuperación)":
    st.header("Cálculo de Exactitud (Fortificación)")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Día:** Día en que se realizó la medición.
        - **Concentración Teórica:** Concentración fortificada conocida.
        - **Concentración Medida:** Concentración obtenida tras el análisis experimental.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Concentración Teórica y Medida:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_exactitud)

elif modulo == "Robustez":
    st.header("Evaluación de Robustez")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Factores variables:** Datos que representan condiciones variables del experimento.
        - **Resultados:** Datos de resultados obtenidos bajo dichas condiciones.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos para evaluar robustez:", type=['csv', 'xlsx'])
    procesar_archivo(datos, evaluar_robustez)
