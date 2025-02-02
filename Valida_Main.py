import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, f_oneway
import streamlit as st
from io import BytesIO
from pathlib import Path

# Obtener la ruta del directorio actual
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
imagenes_dir = current_dir / "img"

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
        justify-content: center; /* Center the title and logo */
        align-items: center;
        gap: 10px;
        margin-bottom: 20px; /* Spacing below title */
    }
    .title-container img:first-child {
        width: 120px; /* Adjust first icon size */
        height: auto;
    }
    .title-container img:last-child {
        width: 200px; /* Adjust second icon size */
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
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Escudo_UdeG.svg/662px-Escudo_UdeG.svg.png" alt="UDG Logo">
        <h1>Validación de Métodos Analíticos - Espectrofotometría UV-Vis</h1>
        <img src="https://practicas.cucei.udg.mx/dist/imagenes/logo_cucei_blanco.png" alt="CUCEI Logo">
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        2025 - Luis Angel Cardenas Medina
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
    if len(x) < 2 or len(y) < 2:
        st.error("No hay suficientes datos para realizar la regresión. Se requieren al menos 2 puntos.")
        return None, None, None, None, None
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
        
        # Calcular parámetros de regresión
        regresion = linregress(estandar_promedio['Concentración'], estandar_promedio['Absorbancia'])
        slope = regresion.slope
        intercept = regresion.intercept
        r_value = regresion.rvalue
        p_value = regresion.pvalue
        
        # Calcular residuales
        predicciones = slope * estandar_promedio['Concentración'] + intercept
        residuales = estandar_promedio['Absorbancia'] - predicciones

        st.write(f"**Día {dia}:**")
        st.write(f"  - **Pendiente (Slope):** {slope:.4f}")
        st.write(f"  - **Intercepto (Intercept):** {intercept:.4f}")
        st.write(f"  - **Coeficiente de correlación (R):** {r_value:.4f}")
        st.write(f"  - **Coeficiente de determinación (R²):** {r_value**2:.4f}")
        st.write(f"  - **Valor p:** {p_value:.4e}")

        if r_value**2 >= 0.995:
            st.success(f"Cumple con los criterios de linealidad para el Día {dia} (R² ≥ 0.995).")
        else:
            st.error(f"No cumple con los criterios de linealidad para el Día {dia} (R² < 0.995).")

        # Gráfica de regresión lineal
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.regplot(x=estandar_promedio['Concentración'], y=estandar_promedio['Absorbancia'], 
                   ci=None, line_kws={'color': 'red'})
        plt.title(f"Regresión Lineal (Día {dia})")
        plt.xlabel("Concentración")
        plt.ylabel("Absorbancia")

        # Gráfica de residuales
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=estandar_promedio['Concentración'], y=residuales, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f"Análisis de Residuales (Día {dia})")
        plt.xlabel("Concentración")
        plt.ylabel("Residuales")
        plt.tight_layout()
        st.pyplot(plt)

        # Interpretación de residuales
        with st.expander("Interpretación de Residuales"):
            st.markdown("""
            **Patrones a observar:**
            - **Distribución aleatoria alrededor de cero:** Indica buen ajuste del modelo
            - **Patrón no lineal:** Sugiere relación no capturada por el modelo
            - **Funnel shape (Cono):** Indica heterocedasticidad (varianza no constante)
            - **Outliers evidentes:** Puntos que se desvían significativamente
            """)
            
            if (abs(residuales) > 2 * residuales.std()).any():
                st.warning("Se detectaron posibles outliers en los residuales")
            else:
                st.success("Residuales dentro del rango esperado (±2σ)")

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
            slope, intercept, _, _, _ = linregress(estandares_dia['Absorbancia'], estandares_dia['Concentración'])
            muestras_dia['Concentración Real'] = slope * muestras_dia['Absorbancia'] + intercept
            datos_muestra.update(muestras_dia)
            pendiente, intercepto = slope, intercept
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

        if rsd_general <= 3:
            st.success(f"{tipo}: Cumple con los criterios de precisión (RSD ≤ 3%).")
        else:
            st.error(f"{tipo}: No cumple con los criterios de precisión (RSD > 3%).")

    st.write("**Concentraciones reales calculadas para las muestras:**")
    st.table(datos_muestra[['Día', 'Absorbancia', 'Concentración Real']])

def calcular_exactitud(datos):
    """Calcula la exactitud mediante recuperación usando curva de calibración diaria según ICH Q2."""
    # Validar columnas requeridas
    columnas_necesarias = ['Día', 'Concentración', 'Absorbancia', 'Tipo']
    if not validar_columnas(datos, columnas_necesarias):
        return
    
    # Separar estándares y muestras
    estandares = datos[datos['Tipo'] == 'Estándar']
    muestras = datos[datos['Tipo'] == 'Muestra']
    
    if estandares.empty:
        st.error("Error: No se encontraron datos de estándares para generar la curva de calibración")
        return
    
    # Calcular concentraciones reales para muestras
    muestras_calculadas = []
    for dia in datos['Día'].unique():
        # Filtrar datos del día
        est_dia = estandares[estandares['Día'] == dia]
        mues_dia = muestras[muestras['Día'] == dia]
        
        # Validar estándares del día
        if len(est_dia) < 2:
            st.warning(f"Día {dia}: Insuficientes estándares para generar curva. Mínimo 2 requeridos.")
            continue
            
        try:
            # Generar curva de calibración
            slope, intercept, r_value, p_value, std_err = linregress(
                est_dia['Absorbancia'], 
                est_dia['Concentración']
            )
            
            # Calcular concentraciones reales para muestras con redondeo
            mues_dia = mues_dia.copy()
            mues_dia['Concentración Medida'] = (slope * mues_dia['Absorbancia'] + intercept).round(2)  # Redondeo a 2 decimales
            mues_dia['Recuperación (%)'] = ((mues_dia['Concentración Medida'] / mues_dia['Concentración']) * 100).round(2)  # Redondeo a 2 decimales
            
            muestras_calculadas.append(mues_dia)
            
            # Mostrar parámetros de la curva (manteniendo 4 decimales para precisión técnica)
            st.subheader(f"Día {dia} - Parámetros de la curva")
            st.markdown(f"""
            - **Ecuación:** y = {slope:.4f}x + {intercept:.4f}
            - **Coeficiente de determinación (R²):** {r_value**2:.4f}
            - **Error estándar:** {std_err:.4f}
            """)
            
        except Exception as e:
            st.error(f"Error en día {dia}: {str(e)}")
            continue
    
    if not muestras_calculadas:
        st.error("No se pudo calcular ninguna concentración. Verifica los datos de entrada.")
        return
    
    # Unificar todos los resultados
    resultados = pd.concat(muestras_calculadas)
    
    # Análisis de exactitud
    st.header("Análisis de Exactitud (ICH Q2)")
    
    # Cálculo de métricas por día con redondeo final
    resumen = resultados.groupby('Día').agg(
        Muestras_analizadas=('Recuperación (%)', 'size'),
        Media_Recuperación=('Recuperación (%)', lambda x: round(x.mean(), 2)),
        DE_Recuperación=('Recuperación (%)', lambda x: round(x.std(), 2)),
        Mínimo=('Recuperación (%)', lambda x: round(x.min(), 2)),
        Máximo=('Recuperación (%)', lambda x: round(x.max(), 2))
    ).reset_index()
    
    # Evaluación de criterios ICH Q2
    resumen['Cumple_ICH'] = (
        (resumen['Media_Recuperación'] >= 98) & 
        (resumen['Media_Recuperación'] <= 102) & 
        (resumen['DE_Recuperación'] <= 3)
    )
    
    # Mostrar resultados
    st.subheader("Resumen Estadístico por Día")
    st.dataframe(resumen.style.format({
        'Media_Recuperación': '{:.2f}%',
        'DE_Recuperación': '{:.2f}%',
        'Mínimo': '{:.2f}%',
        'Máximo': '{:.2f}%'
    }))
    
    # Detalle de recuperaciones
    st.subheader("Detalle de Muestras")
    st.dataframe(resultados[['Día', 'Concentración', 'Absorbancia', 'Concentración Medida', 'Recuperación (%)']]
                 .style.format({
                     'Concentración Medida': '{:.2f}',  # Reducido a 2 decimales
                     'Recuperación (%)': '{:.2f}%'
                 }))
    
    # Generar archivo descargable
    generar_descarga(resultados)

def validar_columnas(datos, columnas):
    """Valida la presencia de columnas requeridas en el dataset"""
    faltantes = [col for col in columnas if col not in datos.columns]
    if faltantes:
        st.error(f"Columnas faltantes: {', '.join(faltantes)}")
        return False
    return True

def generar_descarga(datos):
    """Genera archivo Excel descargable con los resultados"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        datos.to_excel(writer, index=False, sheet_name='Resultados')
    
    st.download_button(
        label="📥 Descargar Resultados Completos",
        data=output.getvalue(),
        file_name="exactitud_analitica.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descarga todos los resultados en formato Excel"
    )

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
    st.info("""
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Absorbancia:** Valores de absorbancia medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        - **Día:** Identificar el día de la medición.""")  
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    
    
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_linealidad)

# Módulo de LOD y LOQ
elif modulo == "Límites de Detección y Cuantificación":
    st.header("Cálculo de LOD y LOQ")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Absorbancia:** Valores de absorbancia medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        - **Día:** Día en que se realizó la medición.
        """)  
    
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    if datos:
        datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
        procesar_archivo(datos, calcular_lod_loq)
        graficar_curva_calibracion_streamlit(datos_df)

# Módulo de Precisión

elif modulo == "Precisión (Repetibilidad e Intermedia)":
    st.header("Evaluación de Precisión")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Absorbancia:** Datos de absorbancia agrupados por días y repeticiones.
        """
    )
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_precision)

# Módulo de Exactitud
elif modulo == "Exactitud (Recuperación)":
    st.header("Cálculo de Exactitud")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Día:** Día en que se realizó la medición.
        - **Concentración Teórica:** Concentración fortificada conocida.
        - **Concentración Medida:** Concentración obtenida tras el análisis experimental.
        """
    )
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_exactitud)

# Módulo de Robustez
elif modulo == "Robustez":
    st.header("Evaluación de Robustez")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Factores variables:** Datos que representan condiciones variables del experimento.
        - **Resultados:** Datos de resultados obtenidos bajo dichas condiciones.
        """) 
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, evaluar_robustez)