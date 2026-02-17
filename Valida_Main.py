import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO
from pathlib import Path
from datetime import datetime
from scipy.stats import linregress, t, ttest_1samp
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import pytz
import plotly.express as px


# Obtener la ruta del directorio actual
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
imagenes_dir = current_dir / "img"
import streamlit as st
from datetime import datetime

# 1. Configuración de la página
st.set_page_config(
    page_title="Plataforma de Validación Analítica CUCEI",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Escudo_CUCEI.svg/424px-Escudo_CUCEI.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Moderno con diseño mejorado + theme support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    :root {
        /* Primary brand colors (dark theme defaults) */
        --color-primary: #2E8BFF; /* vibrant blue */
        --color-secondary: #00C897; /* fresh green */
        --bg-gradient: linear-gradient(152deg, #071124 0%, #0b1a2a 100%);
        --sidebar-bg: rgba(8,12,20,0.95);
        --card-bg: rgba(255,255,255,0.03);
        --text-color: #EAF4FF;
        --muted-color: rgba(234,244,255,0.8);
        --border-color: rgba(255,255,255,0.08);
        --accent-shadow: rgba(46,139,255,0.12);
        --border-radius: 16px;
        --transition-speed: 0.35s;
    }

    /* Light theme overrides */
    .light-theme {
        --color-primary: #0b63ff;
        --color-secondary: #00a678;
        --bg-gradient: linear-gradient(152deg,#f7fbff 0%, #eef6ff 100%);
        --sidebar-bg: #ffffff;
        --card-bg: rgba(11,23,38,0.02);
        --text-color: #07203a;
        --muted-color: rgba(7,32,58,0.75);
        --border-color: rgba(7,32,58,0.06);
        --accent-shadow: rgba(11,99,255,0.06);
    }

    /* Base styling uses variables for full theming */
    body {
        background: var(--bg-gradient);
        color: var(--text-color);
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        line-height: 1.6;
        transition: background var(--transition-speed) ease, color var(--transition-speed) ease;
    }

    .title-container {
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 1.25rem;
        padding: 1.25rem 2rem;
        margin: 1.75rem 0;
        background: var(--card-bg);
        border-radius: var(--border-radius);
        backdrop-filter: blur(14px) saturate(140%);
        border: 1px solid var(--border-color);
        box-shadow: 0 10px 28px rgba(2,6,12,0.45);
        transition: all var(--transition-speed) ease;
    }

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--text-color);
        background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.6px;
        margin: 0;
        padding: 0.5rem 1.25rem;
        position: relative;
        text-shadow: 0 6px 14px rgba(0,0,0,0.45);
    }

    .title-container img { height: 92px; transition: all var(--transition-speed) cubic-bezier(.2,.9,.3,1); filter: drop-shadow(0 6px 18px rgba(0,0,0,0.35)); opacity:0.95 }
    .title-container img:hover { transform: scale(1.06) rotate(-1.5deg); filter: drop-shadow(0 10px 24px rgba(0,0,0,0.42)); opacity:1 }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        backdrop-filter: blur(18px) saturate(150%);
        border-right: 1px solid var(--border-color);
        box-shadow: 6px 0 24px rgba(0,0,0,0.18);
    }

    .stSelectbox [data-baseweb="select"] { background: rgba(255,255,255,0.04) !important; border-radius:12px !important; padding:0.8rem 1rem !important }

    .footer-container { position: fixed; bottom:0; left:0; right:0; background: var(--sidebar-bg); color: var(--muted-color); text-align:center; padding:1rem; font-size:0.95rem; backdrop-filter: blur(12px); z-index:999; border-top:1px solid var(--border-color) }

    .sidebar-title { font-size:1.6rem; margin:1.5rem 0; text-align:center; color:var(--text-color); font-weight:700 }

    .sidebar-section { padding:1.5rem 0; border-top:1px solid var(--border-color) }

    .sidebar-link { display:flex; align-items:center; gap:0.9rem; padding:0.9rem 1.1rem; border-radius:10px; color:var(--text-color) !important; background: rgba(255,255,255,0.02); margin:0.45rem 0 }
    .sidebar-link:hover { transform: translateX(6px); box-shadow: 4px 8px 20px var(--accent-shadow) }

    .contact-info { margin-top:1.6rem; padding:1rem; background: rgba(255,255,255,0.02); border-radius:12px; border:1px solid var(--border-color); text-align:center }

    @media (max-width: 768px){ .title-container{ grid-template-columns:1fr; gap:1rem; padding:1rem } .main-title{ font-size:1.9rem } .title-container img{ height:72px } }

    </style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="footer-container">
    <div>© {current_year} Centro Universitario de Ciencias Exactas e Ingenierías</div>
    <div>Desarrollado por: Luis Angel Cardenas Medina</div>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar con título, selectbox y sección de ayuda/contacto
with st.sidebar:
    # CSS adicional para mejoras estéticas
    st.markdown("""
    <style>
        .sidebar-pro {
            background: linear-gradient(195deg, 
                hsl(210, 35%, 12%) 0%, 
                hsl(210, 35%, 15%) 100%) !important;
            border-right: 1px solid rgba(79, 172, 254, 0.15) !important;
        }
        
        .module-title {
            font-size: 1.8rem !important;
            font-weight: 600;
            color: #ffffff; /* fallback for contrast */
            background: linear-gradient(45deg, #4facfe, #00f2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 1.5rem 0 2.5rem 0;
            text-align: center;
            letter-spacing: -0.5px;
            text-shadow: 0 4px 10px rgba(0,0,0,0.45);
        }
        
        .stSelectbox [data-baseweb="select"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(79, 172, 254, 0.3) !important;
            border-radius: 10px !important;
            padding: 0.8rem 1rem !important;
            margin-bottom: 2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .stSelectbox [data-baseweb="select"]:hover {
            border-color: #4facfe !important;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.15);
        }
        
        .support-card {
            background: rgba(79, 172, 254, 0.08) !important;
            border: 1px solid rgba(79, 172, 254, 0.15) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
            transition: transform 0.3s ease;
        }
        
        .support-card:hover {
            transform: translateY(-2px);
            background: rgba(79, 172, 254, 0.12) !important;
        }
    </style>
    """, unsafe_allow_html=True)

        # Theme toggle widget (persists choice in localStorage)
        try:
                st.components.v1.html("""
                <div style="display:flex;gap:8px;align-items:center;padding:8px 4px;">
                    <label style="font-weight:600;margin-right:6px;color:var(--text-color);">Tema</label>
                    <button id="btn-light" style="padding:6px 10px;border-radius:8px;border:1px solid rgba(0,0,0,0.06);background:transparent;cursor:pointer;">Blanco</button>
                    <button id="btn-dark" style="padding:6px 10px;border-radius:8px;border:1px solid rgba(0,0,0,0.06);background:transparent;cursor:pointer;">Oscuro</button>
                </div>
                <script>
                const applyTheme = (t)=>{
                    if(t==='light'){document.documentElement.classList.add('light-theme'); localStorage.setItem('sv_theme','light');}
                    else {document.documentElement.classList.remove('light-theme'); localStorage.setItem('sv_theme','dark');}
                }
                document.getElementById('btn-light').addEventListener('click',()=>applyTheme('light'));
                document.getElementById('btn-dark').addEventListener('click',()=>applyTheme('dark'));
                window.addEventListener('load',()=>{const t=localStorage.getItem('sv_theme')||'dark'; applyTheme(t);});
                </script>
                """, height=64)
        except Exception:
                pass

    # Título con diseño premium
    st.markdown('<div class="module-title">Módulos Analíticos</div>', unsafe_allow_html=True)
    
    # Selectbox mejorado
    modulo = st.selectbox(
        label="Seleccionar módulo analítico:",
        options=[
            "Linealidad y Rango",
            "Límites de Detección y Cuantificación",
            "Exactitud (Recuperación)",
            "Precisión (Repetibilidad e Intermedia)",
            "Robustez",
            "Estabilidad"
        ],
        index=0,
        key="modulo_principal",
        help="Seleccione el tipo de validación a realizar"
    )

    # Sección de soporte premium
    with st.container():
        st.markdown("""
        <div class="support-card">
            <div style="margin-bottom: 1.5rem;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #4facfe; margin-bottom: 0.5rem;">
                    Guía De Uso 
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5;">
                    Guía completa de uso con especificaciones técnicas y protocolos detallados.
                </div>
            </div>
            <a href="https://drive.google.com/file/d/1ut1P-crNf7wDLaN_ieXwVvmkIr1UU1ZA/view" 
               target="_blank"
               style="text-decoration: none;">
                <button style="
                    width: 100%;
                    padding: 0.7rem;
                    background: rgba(79, 172, 254, 0.1);
                    border: 1px solid rgba(79, 172, 254, 0.3);
                    border-radius: 8px;
                    color: #4facfe;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s ease;">
                    Abrir Guía
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    with st.container():
        st.markdown("""
        <div class="support-card">
            <div style="margin-bottom: 1.5rem;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #4facfe; margin-bottom: 0.5rem;">
                    Contacto
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5;">
                    Correo del desarrollador para dudas y soporte técnico.
                </div>
            </div>
            <a href="mailto:lui.cardenas784@gmail.com" 
               style="text-decoration: none;">
                <button style="
                    width: 100%;
                    padding: 0.7rem;
                    background: rgba(79, 172, 254, 0.1);
                    border: 1px solid rgba(79, 172, 254, 0.3);
                    border-radius: 8px;
                    color: #4facfe;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s ease;">
                    Contactar Soporte
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)    
        


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

def procesar_archivo(archivo, funcion_procesamiento, modulo):
    if archivo:
        try:
            # Inicializar generador de PDF
            pdf_gen = PDFGenerator(modulo)
            
            # Cargar datos
            if archivo.name.endswith('.csv'):
                data = pd.read_csv(archivo)
            else:
                data = pd.read_excel(archivo)
            
            # Verificar que funcion_procesamiento es callable o iterable de callables
            if isinstance(funcion_procesamiento, (list, tuple)):
                resultados = []
                for func in funcion_procesamiento:
                    if callable(func):
                        resultado = func(data, pdf_gen)
                        resultados.append(resultado)
                    else:
                        st.error("Error: Uno de los elementos de la lista/tuple de funciones no es callable.")
                        return False
                overall_result = all(resultados)
            else:
                if callable(funcion_procesamiento):
                    overall_result = funcion_procesamiento(data, pdf_gen)
                else:
                    st.error("Error: La función de procesamiento no es callable.")
                    return False
            
            # Generar PDF si el procesamiento es exitoso
            if overall_result is not False:
                pdf = pdf_gen.generate_pdf()
                st.session_state['current_pdf'] = pdf
                st.session_state['current_module'] = modulo
                return True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    return None


def calcular_linealidad(data, pdf_gen):
    """Calcula la linealidad y rango del método agrupando por Analista y Día."""
    sns.set_theme(style="whitegrid", palette="muted")
    COLORS = ['#2ecc71', '#e74c3c']
    
    # 1. Validación de columnas y datos
    req_cols = ['Concentración', 'Respuesta', 'Tipo', 'Día', 'Analista']
    if not validar_columnas(data, req_cols):
        return False

    data = data.copy()
    for col in ['Concentración', 'Respuesta']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=req_cols, inplace=True)

    # 2. Verificación de niveles de concentración
    niveles = data[data['Tipo'] == 'Estándar']['Concentración'].nunique()
    if niveles < 5:
        st.error("✘ Requiere 5 niveles de concentración (guía ICH Q2(R1))")
        return False

    cumplimiento_global = True
    metricas_modulo = []

    # 3. Análisis por Analista y Día
    for (analista, dia), grupo in data.groupby(['Analista', 'Día']):
        with st.container():
            st.markdown(f"## 🔬 Analista: {analista} | 📅 Día: {dia}")
            pdf_gen.add_subsection(f"Analista: {analista} - Día: {dia}")

            # 4. Procesamiento de estándares
            estandar = grupo[grupo['Tipo'] == 'Estándar']
            if estandar.empty:
                st.warning(f"⚠️ Sin estándares para {analista} - Día {dia}")
                continue

            # 5. Validación de triplicados
            conteo = estandar.groupby('Concentración').size()
            if any(conteo < 3):
                st.error(f"✘ Triplicados incompletos en {analista}-{dia} (USP <1225>)")
                continue

            # 6. Cálculos estadísticos
            try:
                estandar_prom = estandar.groupby('Concentración').agg(
                    Respuesta=('Respuesta', 'mean'),
                    DE=('Respuesta', 'std')
                ).reset_index()

                X = estandar_prom['Concentración']
                y = estandar_prom['Respuesta']
                reg = linregress(X, y)
                
                # Intervalo de confianza pendiente
                n = len(X)
                t_val = t.ppf(0.975, n-2)
                ci_slope = (reg.slope - t_val*reg.stderr, reg.slope + t_val*reg.stderr)
                slope_cumple = not (ci_slope[0] <= 0 <= ci_slope[1])
                
                # Residuales porcentuales
                pred_y = reg.slope*X + reg.intercept
                residual_pct = ((y - pred_y)/pred_y)*100
                residual_cumple = all(abs(residual_pct) <= 2)

            except Exception as e:
                st.error(f"Error en análisis: {str(e)}")
                continue

            # 7. Criterios de aceptación
            adj_r2 = 1 - (1 - reg.rvalue**2)*(n-1)/(n-2)
            cumple = all([adj_r2 >= 0.98, slope_cumple, residual_cumple])

            # 8. Registro de métricas
            metricas = {
                'Analista': analista,
                'Día': dia,
                'R²': adj_r2,
                'Pendiente': reg.slope,
                'IC Pendiente': f"[{ci_slope[0]:.4f}, {ci_slope[1]:.4f}]",
                'Residual Max (%)': f"{abs(residual_pct).max():.2f}",
                'Cumplimiento': cumple
            }
            metricas_modulo.append(metricas)

            # 9. Visualización y tablas
            fig = plt.figure(figsize=(14, 6), dpi=150)
            gs = fig.add_gridspec(2, 2)
            
            # Gráfico de regresión
            ax1 = fig.add_subplot(gs[0, 0])
            sns.regplot(x=X, y=y, ax=ax1, ci=95, 
                        scatter_kws={'s':80, 'edgecolor':'black'},
                        line_kws={'color':COLORS[0], 'lw':2})
            ax1.set_title(f"Regresión Lineal - {analista} (Día {dia})")
            
            # Residuales
            ax2 = fig.add_subplot(gs[0, 1])
            sns.scatterplot(x=X, y=residual_pct, ax=ax2, color=COLORS[1])
            ax2.axhline(0, color='black', ls='--')
            ax2.fill_between(X, -2, 2, color='gray', alpha=0.1)
            ax2.set_title("Residuales (%)")
            
            # Tabla para Streamlit
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis('off')
            tabla_streamlit = [
                ["Parámetro", "Valor", "Cumple"],
                [f"R² Ajustado", f"{adj_r2:.4f}", "✔" if adj_r2 >=0.98 else "✘"],
                [f"IC Pendiente", f"{ci_slope[0]:.4f}-{ci_slope[1]:.4f}", "✔" if slope_cumple else "✘"],
                [f"Residual Máx.", f"{abs(residual_pct).max():.2f}%", "✔" if residual_cumple else "✘"]
            ]
            tabla = ax3.table(
                cellText=tabla_streamlit, 
                loc='center', 
                colWidths=[0.3,0.4,0.3],
                cellLoc='center'
            )
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(12)
            
            # Estilizar celdas
            for row in [1,2,3]:
                celda = tabla.get_celld()[(row,2)]
                celda.set_text_props(
                    color='green' if "✔" in celda.get_text().get_text() else 'red',
                    weight='bold'
                )

            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla para PDF
            pdf_gen.add_table(
                table_data=[
                    ["Parámetro", "Valor", "Cumplimiento"],
                    ["R² Ajustado", f"{adj_r2:.4f}", "Cumple" if adj_r2 >=0.98 else "No cumple"],
                    ["IC Pendiente", f"{ci_slope[0]:.4f} a {ci_slope[1]:.4f}", "Cumple" if slope_cumple else "No cumple"],
                    ["Residual Máximo", f"{abs(residual_pct).max():.2f}%", "Cumple" if residual_cumple else "No cumple"]
                ],
                title=f"Resultados {analista} - Día {dia}"
            )
            
            pdf_gen.capture_figure(fig, f"Linealidad_{analista}_{dia}")

    # 10. Resumen consolidado
    if metricas_modulo:
        df_metricas = pd.DataFrame(metricas_modulo)
        
        # Tabla resumen PDF
        tabla_resumen = [
            ["Analista", "Día", "R²", "IC Pendiente", "Cumplimiento"]
        ] + [
            [
                row['Analista'],
                str(row['Día']),
                f"{row['R²']:.4f}",
                row['IC Pendiente'],
                "Cumple" if row['Cumplimiento'] else "No cumple"
            ] for _, row in df_metricas.iterrows()
        ]
        
        pdf_gen.add_table(
            table_data=tabla_resumen,
            title="Resumen General de Linealidad"
        )

        # Conclusión final
        cumplimiento_global = all(df_metricas['Cumplimiento'])
        st.metric("Cumplimiento Global", 
                 "✅ Cumple" if cumplimiento_global else "❌ No Cumple",
                 delta=f"{df_metricas['Cumplimiento'].sum()}/{len(df_metricas)} grupos válidos")
        pdf_gen.add_metric("Cumplimiento Global", 
                          "Cumple" if cumplimiento_global else "No Cumple", 
                          cumplimiento_global)

    return True

def calcular_lod_loq(datos, pdf_gen):
    """Calcula LOD y LOQ con visualización y reporte PDF integrados según las guías de Farmacéuticos y CCAYAC."""
    COLORS = ['#3498db', '#2ecc71', '#e74c3c']  # Azul, Verde, Rojo
    sns.set_theme(style="whitegrid", font_scale=0.95)
    
    # Sección inicial en el PDF
    pdf_gen.add_section_title("Resultados de LOD y LOQ")
    pdf_gen.add_text_block("Este reporte presenta el cálculo de LOD (Límite de Detección) y LOQ (Límite de Cuantificación) "
                           "según la metodología validada por las guías de Farmacéuticos y CCAYAC.")
    
    # Validación de columnas y tipos numéricos
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Respuesta']
    if not validar_columnas(datos, columnas_necesarias):
        pdf_gen.add_text_block("✘ Error: Faltan columnas necesarias en el dataset.", style='error')
        return False

    if not np.issubdtype(datos['Concentración'].dtype, np.number) or \
       not np.issubdtype(datos['Respuesta'].dtype, np.number):
        pdf_gen.add_text_block("✘ Error: Las columnas 'Concentración' y 'Respuesta' deben ser numéricas.", style='error')
        st.error("✘ Las columnas 'Concentración' y 'Respuesta' deben ser numéricas")
        return False

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        pdf_gen.add_text_block("✘ Error: No se encontraron datos de tipo 'Estándar'.", style='error')
        st.error("✘ No se encontraron datos de tipo 'Estándar'")
        return False

    # Mostrar la metodología en Streamlit y agregar versión "limpia" para el PDF
    with st.expander("📊 Método de Cálculo", expanded=True):
        contenido_markdown = """
        **Fórmulas aplicadas:**
        - $LOD = \\frac{3 \\times σ}{S}$  
        - $LOQ = \\frac{10 \\times σ}{S}$  
        Donde:
        - $σ$: Desviación estándar de los residuales (o blancos)  
        - $S$: Pendiente de la curva de calibración
        """
        st.markdown(contenido_markdown)
        contenido_pdf = (
            "Fórmulas aplicadas:\n"
            "- LOD = (3 × σ) / S\n"
            "- LOQ = (10 × σ) / S\n"
            "Donde:\n"
            "- σ: Desviación estándar de los residuales (o blancos)\n"
            "- S: Pendiente de la curva de calibración"
        )
        pdf_gen.add_subsection("Metodología de Cálculo")
        pdf_gen.add_text_block(contenido_pdf)
    
    # Procesar cada día
    for dia in datos_estandar['Día'].unique():
        with st.container():
            st.markdown(f"## 📅 Día {dia}")
        
            datos_dia = datos_estandar[datos_estandar['Día'] == dia]
            if len(datos_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 puntos requeridos para un cálculo confiable (Día {dia})")
                pdf_gen.add_text_block(f"⚠️ Mínimo 3 puntos requeridos para un cálculo confiable (Día {dia})", style='error')
                continue
            try:
                # Cálculos de regresión y parámetros
                regresion = linregress(datos_dia['Concentración'], datos_dia['Respuesta'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                r_squared = r_value**2
                residuals = datos_dia['Respuesta'] - (slope * datos_dia['Concentración'] + intercept)
                std_dev = residuals.std()
                # Uso de factores 3 y 10 según las guías
                lod = (3 * std_dev) / slope if slope != 0 else np.nan
                loq = (10 * std_dev) / slope if slope != 0 else np.nan
                
                # Nota: Los criterios de aceptación requieren evaluar que:
                # - LOD < especificación de impurezas
                # - LOQ ≤ 50% del límite de especificación
                # Dado que estas especificaciones dependen de la matriz, se deben comparar externamente.
                
                # Mostrar métricas en Streamlit (se usan tres columnas)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pendiente (S)", f"{slope:.4f}")
                    st.metric("R²", f"{r_squared:.4f}")
                with col2:
                    st.metric("LOD (μg/mL)", f"{lod:.4f}", help="(3×σ)/S")
                    st.metric("LOQ (μg/mL)", f"{loq:.4f}", help="(10×σ)/S")
                
                # Agregar métricas al PDF (tabla)
                metricas_dia = {
                    'Pendiente (S)': f"{slope:.4f}",
                    'R²': f"{r_squared:.4f}",
                    'LOD (μg/mL)': f"{lod:.4f}",
                    'LOQ (μg/mL)': f"{loq:.4f}"
                }
                pdf_gen.add_metrics_table(f"Métricas Día {dia}", metricas_dia)
                
                # Crear gráfica: mostrar línea de regresión y líneas verticales en LOD y LOQ (si están en rango)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(datos_dia['Concentración'], datos_dia['Respuesta'], color='lightgray', label='Datos estándar')
                x_vals = np.linspace(datos_dia['Concentración'].min(), datos_dia['Concentración'].max(), 100)
                y_vals = slope * x_vals + intercept
                ax.plot(x_vals, y_vals, color=COLORS[0], label='Línea de regresión')
                if datos_dia['Concentración'].min() <= lod <= datos_dia['Concentración'].max():
                    ax.axvline(lod, color=COLORS[2], linestyle='--', label=f'LOD = {lod:.2f}')
                if datos_dia['Concentración'].min() <= loq <= datos_dia['Concentración'].max():
                    ax.axvline(loq, color=COLORS[1], linestyle='-.', label=f'LOQ = {loq:.2f}')
                ax.set_title(f"Calibración y Límites - Día {dia} (R² = {r_squared:.4f})", fontsize=14)
                ax.set_xlabel("Concentración (μg/mL)", fontsize=12)
                ax.set_ylabel("Respuesta (UA)", fontsize=12)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                pdf_gen.capture_figure(fig, f"Limites_Dia_{dia}")
                plt.close(fig)
                
            except Exception as e:
                st.error(f"✘ Error en Día {dia}: {str(e)}")
                pdf_gen.add_text_block(f"✘ Error en Día {dia}: {str(e)}", style='error')
                continue
    
    return True

def graficar_curva_calibracion_streamlit(datos, pdf_gen):
    """Grafica la curva de calibración con estilo profesional e integra la gráfica y la tabla en el reporte PDF, según las guías de Farmacéuticos y CCAYAC."""
    COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Verde, Azul, Rojo
    plt.style.use('seaborn-v0_8-talk')
    
    # Validación de datos
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Respuesta']
    if not validar_columnas(datos, columnas_necesarias):
        pdf_gen.add_text_block("✘ Error: Faltan columnas necesarias en el dataset para calibración.", style='error')
        return False

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        pdf_gen.add_text_block("✘ Error: No se encontraron datos de calibración.", style='error')
        st.error("✘ No se encontraron datos de calibración")
        return False

    pdf_gen.add_section_title("Curvas de Calibración")
    
    for dia in datos_estandar['Día'].unique():
        with st.container():
            st.markdown(f"## 📈 Curva de Calibración - Día {dia}")
            datos_dia = datos_estandar[datos_estandar['Día'] == dia]
            if len(datos_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 puntos recomendados para curva confiable (Día {dia})")
                pdf_gen.add_text_block(f"⚠️ Mínimo 3 puntos recomendados para curva confiable (Día {dia})", style='error')
                continue
            try:
                # Cálculos de regresión y parámetros
                regresion = linregress(datos_dia['Concentración'], datos_dia['Respuesta'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                r_squared = r_value**2
                residuals = datos_dia['Respuesta'] - (slope * datos_dia['Concentración'] + intercept)
                std_dev = residuals.std()
                # Uso de los factores 3 y 10 según la metodología
                lod = (3 * std_dev) / slope if slope != 0 else np.nan
                loq = (10 * std_dev) / slope if slope != 0 else np.nan

                # Crear gráfica profesional de calibración
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
                sns.regplot(x=datos_dia['Concentración'], y=datos_dia['Respuesta'],
                           ax=ax, ci=95,
                           scatter_kws={'s': 80, 'color': COLORS[0], 'edgecolor': 'black', 'alpha': 0.8},
                           line_kws={'color': COLORS[1], 'lw': 2, 'alpha': 0.8})
                ax.axvline(lod, color=COLORS[2], linestyle='--', lw=2, alpha=0.8, label=f'LOD ({lod:.2f})')
                ax.axvline(loq, color=COLORS[1], linestyle='-.', lw=2, alpha=0.8, label=f'LOQ ({loq:.2f})')
                textstr = '\n'.join((
                    f'$R^2 = {r_squared:.4f}$',
                    f'$y = {slope:.4f}x + {intercept:.4f}$',
                    f'$σ = {std_dev:.4f}$'
                ))
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.set_title(f"Curva de Calibración - Día {dia}", fontsize=16, pad=20)
                ax.set_xlabel("Concentración (μg/mL)", fontsize=14)
                ax.set_ylabel("Respuesta (UA)", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='lower right', frameon=True, shadow=True)
                # Opcional: resaltar la zona comprendida entre 0 y LOQ
                ax.axvspan(0, loq, facecolor='#f8d7da', alpha=0.3, label='Zona LOD/LOQ')
                st.pyplot(fig)
                pdf_gen.capture_figure(fig, f"Calibracion_Dia_{dia}")
                plt.close(fig)

                # Mostrar tabla de datos en Streamlit
                with st.expander(f"📋 Datos de Calibración - Día {dia}"):
                    st.dataframe(
                        datos_dia[['Concentración', 'Respuesta']]
                        .style.format("{:.4f}")
                    )
                    
            except Exception as e:
                st.error(f"✘ Error en Día {dia}: {str(e)}")
                pdf_gen.add_text_block(f"✘ Error en Día {dia}: {str(e)}", style='error')
                continue

    return True

def agrupar_valores(valores, umbral=1.0):
    """
    Agrupa una lista ordenada de valores en grupos donde la diferencia entre elementos consecutivos es <= umbral.
    Devuelve una lista de grupos (cada grupo es una lista de valores).
    """
    grupos = []
    if len(valores) == 0:
        return grupos
    grupo_actual = [valores[0]]
    for v in valores[1:]:
        if v - grupo_actual[-1] <= umbral:
            grupo_actual.append(v)
        else:
            grupos.append(grupo_actual)
            grupo_actual = [v]
    grupos.append(grupo_actual)
    return grupos

def calcular_precision_por_rango(datos, pdf_gen, umbral_agrupacion=1.0):
    """
    Realiza el análisis de precisión para cada rango de concentración, siguiendo criterios CCYAC.
    
    Se evalúa:
      A. Precisión del Sistema (8.1): Se calcula el CV para cada grupo (Analista-Día) en los datos de estándar,
         y se identifica el grupo con el menor CV.
      B. Precisión Intermedia (8.7): Se calcula el CV para cada combinación (Analista, Día) y se resumen
         los valores (se reporta el CV mínimo y el promedio).
    
    Se espera que 'datos' contenga las columnas:
       ['Concentración', 'Respuesta', 'Día', 'Analista', 'Tipo']
       
    Los resultados y gráficos se muestran en Streamlit y se documentan en el PDF mediante 'pdf_gen'.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("🎯 Análisis de Precisión por Rango (Agrupamiento Automático)")
    pdf_gen.add_section_title("Análisis de Precisión por Rango (Agrupamiento Automático)")
    
    # 1. Validar columnas requeridas
    required_cols = ['Concentración', 'Respuesta', 'Día', 'Analista', 'Tipo']
    for col in required_cols:
        if col not in datos.columns:
            st.error(f"Falta la columna '{col}' en el DataFrame.")
            pdf_gen.add_text_block(f"✘ Error: Falta la columna '{col}' en el DataFrame.", style="error")
            return False

    # 2. Conversión a numérico y limpieza
    for col in ['Concentración', 'Respuesta']:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos.dropna(subset=['Concentración', 'Respuesta'], inplace=True)
    datos['Tipo'] = datos['Tipo'].str.lower()

    # 3. Agrupar automáticamente los valores de concentración con tolerancia ±umbral_agrupacion
    unique_vals = sorted(datos['Concentración'].unique())
    grupos_conc = agrupar_valores(unique_vals, umbral=umbral_agrupacion)
    mapping = {}
    for grupo in grupos_conc:
        rep = np.median(grupo)
        for v in grupo:
            mapping[v] = rep
    datos['Rango'] = datos['Concentración'].map(mapping).apply(lambda x: f"{x:.2f}")

    # Mostrar la agrupación para depuración
    st.subheader("Agrupación de Concentraciones")
    df_grupos = datos[['Concentración', 'Rango']].drop_duplicates().sort_values(by='Concentración')
    st.dataframe(df_grupos)
    tabla_grupos = [["Concentración", "Rango asignado"]]
    for _, row in df_grupos.iterrows():
        tabla_grupos.append([f"{row['Concentración']:.2f}", row['Rango']])
    pdf_gen.add_table(tabla_grupos, title="Agrupación de Concentraciones (Automática)")

    # 4. Iterar sobre cada rango y evaluar precisión
    rangos_disponibles = sorted(datos['Rango'].unique(), key=lambda x: float(x))
    for rango_label in rangos_disponibles:
        df_rango = datos[datos['Rango'] == rango_label].copy()
        if df_rango.empty:
            continue

        st.markdown(f"### Análisis para el Rango {rango_label}")
        pdf_gen.add_subsection(f"Rango {rango_label}")

        # Filtrar únicamente los datos de tipo 'estándar'
        estandares = df_rango[df_rango['Tipo'] == 'estándar'].copy()
        if estandares.empty:
            st.warning(f"No se encontraron datos de tipo 'estándar' en el rango {rango_label}.")
            pdf_gen.add_text_block(f"⚠️ No se encontraron datos de tipo 'estándar' en el rango {rango_label}.", style="warning")
            continue

        # -----------------------------------------------------
        # A: Precisión del Sistema (8.1)
        # -----------------------------------------------------
        with st.expander(f"📊 Precisión del Sistema - Rango {rango_label}", expanded=False):
            pdf_gen.add_subsection(f"Precisión del Sistema (8.1) - Rango {rango_label}")
            grupos = estandares.groupby(['Analista', 'Día'])
            resultados_sistema = []
            # Calcular CV para cada grupo (se requiere al menos 6 réplicas)
            for (analista, dia), grupo in grupos:
                if len(grupo) < 6:
                    st.warning(f"Analista {analista} (Día {dia}): Insuficientes réplicas ({len(grupo)}/6)")
                    continue
                media = grupo['Respuesta'].mean()
                desv = grupo['Respuesta'].std()
                cv = (desv / media) * 100 if media != 0 else np.nan
                resultados_sistema.append({
                    'Analista': analista,
                    'Día': dia,
                    'Réplicas': len(grupo),
                    'Media': media,
                    'Std': desv,
                    'CV (%)': cv
                })
            if not resultados_sistema:
                st.error(f"Rango {rango_label}: No hay grupos válidos (Analista-Día) con ≥6 réplicas para la precisión del sistema.")
                pdf_gen.add_text_block(f"✘ Rango {rango_label}: Insuficientes grupos (Analista-Día) para calcular precisión del sistema.", style="error")
            else:
                df_resultados_sistema = pd.DataFrame(resultados_sistema)
                # Seleccionar el grupo con menor CV
                mejor_grupo = df_resultados_sistema.loc[df_resultados_sistema['CV (%)'].idxmin()]
                st.markdown(f"**Mejor grupo:** Analista {mejor_grupo['Analista']} - Día {mejor_grupo['Día']} con CV = {mejor_grupo['CV (%)']:.2f}%")
                pdf_gen.add_text_block(
                    f"Mejor grupo (Precisión del Sistema) en el rango {rango_label}: "
                    f"Analista {mejor_grupo['Analista']} - Día {mejor_grupo['Día']} con CV = {mejor_grupo['CV (%)']:.2f}%.")
                st.dataframe(df_resultados_sistema.style.format({"CV (%)": "{:.2f}"}), use_container_width=True)

                # Seleccionar tipo de método para definir el umbral (por CCYAC)
                metodo_sistema = st.selectbox(f"Tipo de Método para Precisión del Sistema (rango {rango_label}):",
                                              ["fisico_quimico", "biologico"], key=f"sistema_{rango_label}")
                umbral_sistema = 1.5 if metodo_sistema == "fisico_quimico" else 3.0

                st.metric("Media Respuesta (Mejor grupo)", f"{mejor_grupo['Media']:.4f}")
                st.metric("Desv. Estándar (Mejor grupo)", f"{mejor_grupo['Std']:.4f}")
                st.metric("CV (Mejor grupo)", f"{mejor_grupo['CV (%)']:.2f}%", 
                          delta="Cumple" if mejor_grupo['CV (%)'] <= umbral_sistema else "No Cumple")

                # Visualización: Distribución de las respuestas en el mejor grupo
                grupo_mejor = grupos.get_group((mejor_grupo['Analista'], mejor_grupo['Día']))
                fig_sistema = plt.figure(figsize=(6, 4))
                # Boxplot con simbología neutra y marcador "x" para outliers
                sns.boxplot(data=grupo_mejor, x='Analista', y='Respuesta', 
                            flierprops=dict(marker='x', markersize=8, markerfacecolor='none', markeredgecolor='black'),
                            palette="Greys")
                # Stripplot con marcador de círculo para diferenciar
                sns.stripplot(data=grupo_mejor, x='Analista', y='Respuesta', 
                              marker="o", size=8, edgecolor='darkgray', linewidth=1, color="black", dodge=True)
                plt.axhline(mejor_grupo['Media'], color='red', linestyle='--', label='Media')
                plt.title(f"Precisión del Sistema - Mejor grupo (Rango {rango_label})")
                plt.legend()
                st.pyplot(fig_sistema)
                pdf_gen.capture_figure(fig_sistema, f"dispersion_sistema_{rango_label}")
                plt.close(fig_sistema)

                tabla_sistema = [
                    ["Parámetro", "Valor"],
                    ["Analista", mejor_grupo['Analista']],
                    ["Día", mejor_grupo['Día']],
                    ["Réplicas", f"{mejor_grupo['Réplicas']}"],
                    ["Media Respuesta", f"{mejor_grupo['Media']:.4f}"],
                    ["Desv. Estándar", f"{mejor_grupo['Std']:.4f}"],
                    ["CV Sistema", f"{mejor_grupo['CV (%)']:.2f}%"],
                    ["Umbral", f"{umbral_sistema}%"],
                    ["Cumplimiento", "Cumple" if mejor_grupo['CV (%)'] <= umbral_sistema else "No Cumple"]
                ]
                pdf_gen.add_table(tabla_sistema, title=f"Precisión del Sistema - Rango {rango_label}")

                # Gráfico complementario: Heatmap de CV por (Analista, Día) para este rango
                pivot_cv = df_resultados_sistema.pivot(index="Analista", columns="Día", values="CV (%)")
                fig_heat, ax_heat = plt.subplots(figsize=(6, 4))
                sns.heatmap(pivot_cv, annot=True, fmt=".2f", cmap="Greys", ax=ax_heat)
                ax_heat.set_title(f"Heatmap CV (Sistema) - Rango {rango_label}")
                st.pyplot(fig_heat)
                pdf_gen.capture_figure(fig_heat, f"heatmap_sistema_{rango_label}")
                plt.close(fig_heat)

        # -----------------------------------------------------
        # B: Precisión Intermedia (8.7)
        # -----------------------------------------------------
        with st.expander(f"📈 Precisión Intermedia - Rango {rango_label}", expanded=False):
            pdf_gen.add_subsection(f"Precisión Intermedia (8.7) - Rango {rango_label}")
            if estandares['Día'].nunique() < 2 or estandares['Analista'].nunique() < 2:
                st.error(f"Rango {rango_label}: Se requieren datos de al menos 2 días y 2 analistas para la precisión intermedia.")
                pdf_gen.add_text_block(f"✘ Rango {rango_label}: Insuficientes días o analistas para precisión intermedia.", style="error")
            else:
                cv_inter_df = (estandares.groupby(['Día', 'Analista'])['Respuesta']
                               .apply(lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else np.nan)
                               .reset_index(name='CV'))
                cv_min = cv_inter_df['CV'].min()
                cv_mean = cv_inter_df['CV'].mean()
                
                st.markdown("#### CV por Día y Analista")
                st.dataframe(cv_inter_df)
                # Gráfico de barras con anotación en cada barra
                fig_inter = plt.figure(figsize=(6, 4))
                ax = sns.barplot(data=cv_inter_df, x='Día', y='CV', hue='Analista', dodge=True, edgecolor="black", palette="Greys")
                for p in ax.patches:
                    altura = p.get_height()
                    ax.annotate(f'{altura:.1f}%', 
                                (p.get_x() + p.get_width() / 2., altura),
                                ha='center', va='bottom', fontsize=9, color='black')
                plt.axhline(cv_min, color='gray', linestyle=':', label=f"CV Mínimo = {cv_min:.2f}%")
                plt.axhline(cv_mean, color='blue', linestyle=':', label=f"CV Promedio = {cv_mean:.2f}%")
                plt.title(f"Precisión Intermedia - Rango {rango_label}")
                plt.legend()
                st.pyplot(fig_inter)
                pdf_gen.capture_figure(fig_inter, f"cv_inter_{rango_label}")
                plt.close(fig_inter)

                metodo_inter = st.selectbox(f"Tipo de Método para Precisión Intermedia (rango {rango_label}):",
                                            ["cromatografico", "quimico", "espectrofotometrico", "biologico"],
                                            key=f"inter_{rango_label}")
                if metodo_inter == "cromatografico":
                    umbral_inter = 2.0
                elif metodo_inter in ["quimico", "espectrofotometrico"]:
                    umbral_inter = 3.0
                elif metodo_inter == "biologico":
                    umbral_inter = 5.0
                else:
                    umbral_inter = 2.0

                st.metric("CV Mínimo (Día-Analista)", f"{cv_min:.2f}%", 
                          delta="Cumple" if cv_min <= umbral_inter else "No Cumple")
                st.metric("CV Promedio (Rango)", f"{cv_mean:.2f}%", 
                          delta="Cumple" if cv_mean <= umbral_inter else "No Cumple")

                tabla_inter = [
                    ["Parámetro", "Valor"],
                    ["CV Mínimo", f"{cv_min:.2f}%"],
                    ["CV Promedio", f"{cv_mean:.2f}%"],
                    ["Umbral Intermedio", f"{umbral_inter}%"]
                ]
                pdf_gen.add_table(tabla_inter, title=f"Precisión Intermedia - Rango {rango_label}")

                # Gráfico adicional: Boxplot comparativo de CV por grupo (día y analista)
                # Aquí se usa simbología neutra: marcador "x" para outliers, paleta en grises
                fig_box, ax_box = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=cv_inter_df, x='Día', y='CV', hue='Analista', 
                            flierprops=dict(marker='x', markersize=8, markerfacecolor='none', markeredgecolor='black'),
                            palette="Greys", dodge=True)
                ax_box.set_title(f"Distribución de CV - Precisión Intermedia (Rango {rango_label})")
                st.pyplot(fig_box)
                pdf_gen.capture_figure(fig_box, f"box_cv_inter_{rango_label}")
                plt.close(fig_box)

    st.success("✔ Análisis de Precisión por Rango finalizado para todos los rangos.")
    pdf_gen.add_text_block("✔ Análisis de Precisión por Rango finalizado para todos los rangos.")
    return True

#####################################
# Clase PDFGenerator
#####################################
class PDFGenerator:
    def __init__(self, modulo):
        self.modulo = modulo
        self.buffers = []           # Para imágenes (figuras)
        self.metrics = []           # Para métricas individuales (opcional)
        self.metrics_tables = []    # Para almacenar tablas de métricas (diccionarios)
        self.custom_tables = []     # Para tablas personalizadas (listas de listas)
        self.text_blocks = []       # Para bloques de texto
        self.sections = []          # Para títulos de sección
        self.styles = getSampleStyleSheet()
        if 'error' not in self.styles.byName:
            self.styles.add(ParagraphStyle(name='error', parent=self.styles['Normal'], textColor=colors.red))
        if 'conclusion' not in self.styles.byName:
            self.styles.add(ParagraphStyle(name='conclusion', parent=self.styles['Normal'], textColor=colors.green))

    def add_table(self, table_data, title=None, headers=None):
        """Versión corregida que maneja headers personalizados"""
        if headers:
            # Insertar headers como primera fila
            table_data = [headers] + table_data
        self.custom_tables.append((title, table_data))

    def capture_figure(self, fig, fig_name=None):
        """
        Captura la figura y la almacena en buffers.
        Si la figura es Matplotlib (tiene savefig), se usa ese método;
        de lo contrario, se asume que es Plotly y se utiliza plotly.io.to_image.
        """
        buf = BytesIO()
        import plotly.io as pio
        if hasattr(fig, "savefig"):
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            # Se convierte la figura Plotly a imagen PNG
            image_bytes = pio.to_image(fig, format="png", scale=2)
            buf.write(image_bytes)
        buf.seek(0)
        self.buffers.append(buf)

    def add_metric(self, label, value, cumplimiento):
        self.metrics.append({
            'label': label,
            'value': value,
            'cumplimiento': cumplimiento
        })

    def add_metrics_table(self, title, metrics):
        self.metrics_tables.append((title, metrics))

    def add_subsection(self, title):
        subsection_style = ParagraphStyle(
            name='SubsectionStyle',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.darkblue,
            spaceAfter=8,
            alignment=0
        )
        self.text_blocks.append((title, subsection_style))

    def add_section_title(self, title):
        self.sections.append(title)

    def add_text_block(self, text, style=None):
        if isinstance(style, str):
            style = self.styles.get(style, self.styles['Normal'])
        elif style is None or not hasattr(style, "name"):
            style = self.styles['Normal']
        self.text_blocks.append((text, style))

    def generate_pdf(self):
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []
        
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=self.styles['Title'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=20,
            alignment=1
        )
        section_style = ParagraphStyle(
            name='SectionStyle',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=10,
            alignment=0
        )
        
        # Portada
        mexico_tz = pytz.timezone("America/Mexico_City")
        elements.append(Paragraph(f"Reporte de Validación: {self.modulo}", title_style))
        elements.append(Paragraph(f"Fecha: {datetime.now(mexico_tz).strftime('%d/%m/%Y %H:%M')}", self.styles['Normal']))
        elements.append(Spacer(1, 30))
        
        for sec in self.sections:
            elements.append(Paragraph(sec, section_style))
            elements.append(Spacer(1, 15))
        
        for text, style in self.text_blocks:
            elements.append(Paragraph(text, style))
            elements.append(Spacer(1, 12))
        
        for buf in self.buffers:
            img = RLImage(buf, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 15))
        
        if self.metrics_tables:
            for title, met_dict in self.metrics_tables:
                elements.append(Paragraph(title, self.styles['Heading2']))
                elements.append(Spacer(1, 12))
                table_data = [["Métrica", "Valor"]]
                for key, value in met_dict.items():
                    table_data.append([key, value])
                tabla = Table(table_data, colWidths=[3*inch, 3*inch])
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(tabla)
                elements.append(Spacer(1, 20))
        
        if self.custom_tables:
            for title, table_data in self.custom_tables:
                if title:
                    elements.append(Paragraph(title, self.styles['Heading2']))
                    elements.append(Spacer(1, 12))
                num_cols = len(table_data[0])
                col_width = 6*inch / num_cols
                tabla = Table(table_data, colWidths=[col_width]*num_cols)
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(tabla)
                elements.append(Spacer(1, 20))
        
        if self.metrics:
            elements.append(Paragraph("Métricas Clave", self.styles['Heading2']))
            metric_data = [["Parámetro", "Valor", "Cumplimiento"]]
            for metric in self.metrics:
                metric_data.append([
                    metric['label'],
                    metric['value'],
                    "✔" if metric['cumplimiento'] else "✘"
                ])
            metric_table = Table(metric_data, colWidths=[3*inch, 2*inch, 1.5*inch])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(metric_table)
            elements.append(Spacer(1, 20))
        
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer

# Función para generar archivo Excel descargable
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

# Función para procesar el archivo
def procesar_archivo(archivo, funcion_procesamiento, modulo):
    if archivo:
        try:
            # Inicializar generador de PDF
            pdf_gen = PDFGenerator(modulo)
            
            # Cargar datos según extensión
            if archivo.name.endswith('.csv'):
                data = pd.read_csv(archivo)
            else:
                data = pd.read_excel(archivo)
            
            # Procesamiento: admitir una función o una lista de funciones
            if isinstance(funcion_procesamiento, (list, tuple)):
                resultados = []
                for func in funcion_procesamiento:
                    if callable(func):
                        resultado = func(data, pdf_gen)
                        resultados.append(resultado)
                    else:
                        st.error("Error: Uno de los elementos de la lista/tuple de funciones no es callable.")
                        return False
                # Verificar que ninguno de los resultados sea False o un DataFrame vacío
                overall_result = True
                for res in resultados:
                    if isinstance(res, bool) and res is False:
                        overall_result = False
                        break
                    elif isinstance(res, pd.DataFrame) and res.empty:
                        overall_result = False
                        break
            else:
                if callable(funcion_procesamiento):
                    overall_result = funcion_procesamiento(data, pdf_gen)
                    if isinstance(overall_result, pd.DataFrame) and overall_result.empty:
                        overall_result = False
                else:
                    st.error("Error: La función de procesamiento no es callable.")
                    return False
            
            # Verificar explícitamente el resultado sin evaluarlo directamente en un contexto booleano
            if overall_result is False:
                st.error("El procesamiento no generó resultados válidos.")
                return False
            else:
                pdf = pdf_gen.generate_pdf()
                st.session_state['current_pdf'] = pdf
                st.session_state['current_module'] = modulo
                return True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    return None

def calcular_exactitud(datos, pdf_gen):
    """
    Calcula la exactitud mediante recuperación según las guías de Farmacéuticos y CCAYAC y genera un reporte PDF integrado.
    Se incluye:
      - Cálculo de la recuperación (%) redondeada a dos decimales.
      - Estadísticos descriptivos: media, DE, mediana, Q1, Q3, IQR.
      - Intervalo de confianza (IC) al 95% para la media.
      - Prueba t (H0: μ = 100) con p-valor.
      - Detección de outliers (método IQR).
      - Visualizaciones: boxplot con límites de aceptación, histograma, gráfico de tendencia y datos detallados.
    """
    # Validar columnas requeridas
    columnas_necesarias = ['Día', 'Concentración Teórica', 'Concentración Real']
    if not validar_columnas(datos, columnas_necesarias):
        pdf_gen.add_text_block("✘ Error: Faltan columnas necesarias en el dataset.", style="error")
        return False

    # Calcular porcentaje de recuperación redondeado a dos decimales
    datos['Recuperación (%)'] = ((datos['Concentración Real'] / datos['Concentración Teórica']) * 100).round(2)

    # Detección de outliers usando el método IQR para cada día
    def detectar_outliers(x):
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ~((x >= lower_bound) & (x <= upper_bound))
    datos['Outlier'] = datos.groupby('Día')['Recuperación (%)'].transform(detectar_outliers)
    
    # Sección inicial del PDF
    pdf_gen.add_section_title("Análisis de Exactitud - Guía Farmacéuticos y CCAYAC")
    pdf_gen.add_text_block(
        "Este reporte presenta el análisis de exactitud mediante recuperación, comparando la concentración teórica "
        "versus la concentración real. Se evalúan los siguientes criterios:\n"
        "- Recuperación media entre 98-102%\n"
        "- Desviación estándar (DE) ≤ 3%\n"
        "- Prueba t (H0: μ = 100) sin significación (p > 0.05)"
    )
    
    # Análisis estadístico agrupado por día
    resumen_list = []
    dias = sorted(datos['Día'].unique())
    for dia in dias:
        grupo = datos[datos['Día'] == dia]
        n = len(grupo)
        media = grupo['Recuperación (%)'].mean()
        de = grupo['Recuperación (%)'].std()
        mediana = grupo['Recuperación (%)'].median()
        Q1 = grupo['Recuperación (%)'].quantile(0.25)
        Q3 = grupo['Recuperación (%)'].quantile(0.75)
        # Calcular intervalo de confianza al 95% para la media (si n > 1)
        if n > 1:
            t_crit = t.ppf(0.975, df=n-1)
            ci_lower = media - t_crit * (de/np.sqrt(n))
            ci_upper = media + t_crit * (de/np.sqrt(n))
        else:
            ci_lower, ci_upper = np.nan, np.nan
        # Prueba t de una muestra: H0: μ = 100
        t_stat, p_val = ttest_1samp(grupo['Recuperación (%)'], 100)
        # Evaluación de criterios de aceptación
        cumple_media = 98 <= media <= 102
        cumple_de = de <= 3
        cumplimiento = "✔" if (cumple_media and cumple_de) else "✘"
        
        resumen_list.append({
            'Día': dia,
            'Muestras': n,
            'Media (%)': media,
            'DE (%)': de,
            'Mediana (%)': mediana,
            'Q1 (%)': Q1,
            'Q3 (%)': Q3,
            'IC Media (%)': f"[{ci_lower:.2f}%, {ci_upper:.2f}%]" if n > 1 else "N/A",
            'p-valor': p_val,
            'Cumplimiento': cumplimiento
        })
    resumen = pd.DataFrame(resumen_list)
    
    # Mostrar resultados en la app Streamlit en pestañas
    st.header("Análisis de Exactitud - Guía Farmacéuticos y CCAYAC")
    tab1, tab2, tab3, tab4 = st.tabs(["Resumen Estadístico", "Distribución de Recuperaciones", "Tendencia", "Datos Detallados"])
    
    # --- Pestaña 1: Resumen Estadístico ---
    with tab1:
        st.subheader("Estadísticos por Día")
        st.dataframe(
            resumen.style.apply(lambda row: ['background: #e6f4ea' if row['Cumplimiento'] == '✔' else 'background: #fce8e6']*len(row), axis=1)
        )
        st.markdown("""
        **Criterios de Aceptación:**
        - Recuperación media entre 98-102%
        - Desviación estándar (DE) ≤ 3%
        - Prueba t (H0: μ = 100) sin significación (p > 0.05)
        """)
        
        # Dividir el resumen en dos tablas:
        # Tabla 1: Datos generales
        tabla_general_columns = ['Día', 'Muestras', 'Media (%)', 'DE (%)', 'Cumplimiento']
        tabla_detalle_columns = ['Día', 'Mediana (%)', 'Q1 (%)', 'Q3 (%)', 'IC Media (%)', 'p-valor']
        
        tabla_general = []
        tabla_detalle = []
        
        # Formatear encabezados con ancho fijo (15 caracteres)
        encabezado_general = [f"{col:<15}" for col in tabla_general_columns]
        encabezado_detalle = [f"{col:<15}" for col in tabla_detalle_columns]
        tabla_general.append(encabezado_general)
        tabla_detalle.append(encabezado_detalle)
        
        # Recorrer cada fila del resumen para agregar a las dos tablas
        for _, row in resumen.iterrows():
            fila_general = []
            fila_detalle = []
            for col in tabla_general_columns:
                valor = row[col]
                if isinstance(valor, (int, float)):
                    fila_general.append(f"{valor:<15.2f}")
                else:
                    fila_general.append(f"{str(valor):<15}")
            for col in tabla_detalle_columns:
                valor = row[col]
                # Para los valores numéricos en detalle, se formatean a 2 decimales
                if isinstance(valor, (int, float)):
                    fila_detalle.append(f"{valor:<15.2f}")
                else:
                    fila_detalle.append(f"{str(valor):<15}")
            tabla_general.append(fila_general)
            tabla_detalle.append(fila_detalle)
        
        # Agregar ambas tablas al PDF con títulos distintos
        pdf_gen.add_table(tabla_general, title="Resumen General de Recuperación")
        pdf_gen.add_table(tabla_detalle, title="Detalle de Recuperación")
        # --- Pestaña 2: Distribución de Recuperaciones ---
    with tab2:
        st.subheader("Distribución de Recuperaciones por Día")
        # Boxplot con límites de aceptación
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=datos, x='Día', y='Recuperación (%)', palette='viridis', ax=ax_box)
        ax_box.axhline(98, color='red', linestyle='--', label='Límite Inferior (98%)')
        ax_box.axhline(102, color='green', linestyle='--', label='Límite Superior (102%)')
        ax_box.set_title("Boxplot de Recuperación (%) por Día")
        ax_box.legend()
        st.pyplot(fig_box)
        pdf_gen.capture_figure(fig_box, "Boxplot_Recuperacion")
        plt.close(fig_box)
        
        # Histograma global
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        sns.histplot(data=datos, x='Recuperación (%)', bins=12, kde=True, hue='Día', palette='viridis', ax=ax_hist)
        ax_hist.set_title("Histograma de Recuperación Global")
        st.pyplot(fig_hist)
        pdf_gen.capture_figure(fig_hist, "Histograma_Recuperacion")
        plt.close(fig_hist)
    
    # --- Pestaña 3: Tendencia de la Recuperación ---
    with tab3:
        st.subheader("Evolución de la Recuperación Media por Día")
        fig_line, ax_line = plt.subplots(figsize=(10, 6))
        dias_sorted = resumen['Día']
        medias = resumen['Media (%)']
        # Extraer y convertir los límites inferior y superior del intervalo
        ci_lower_vals = resumen['IC Media (%)'].apply(
            lambda x: float(x.split('%')[0][1:].strip()) if x != "N/A" else np.nan
        )
        ci_upper_vals = resumen['IC Media (%)'].apply(
            lambda x: float(x.split('%')[1].replace(',', '').replace(']', '').strip()) if x != "N/A" else np.nan
        )
        err_lower = medias - ci_lower_vals
        err_upper = ci_upper_vals - medias
        err = [err_lower, err_upper]
        ax_line.errorbar(dias_sorted, medias, yerr=err, fmt='-o', capsize=5, color='blue', label='Media Recuperación')
        ax_line.axhline(98, color='red', linestyle='--', label='Límite Inferior (98%)')
        ax_line.axhline(102, color='green', linestyle='--', label='Límite Superior (102%)')
        ax_line.set_title("Tendencia de la Recuperación Media")
        ax_line.set_xlabel("Día")
        ax_line.set_ylabel("Recuperación (%)")
        ax_line.legend()
        st.pyplot(fig_line)
        pdf_gen.capture_figure(fig_line, "Tendencia_Recuperacion")
        plt.close(fig_line)
    
    # --- Pestaña 4: Datos Detallados ---
    with tab4:
        st.subheader("Datos Completos")
        st.dataframe(
            datos.style.format({
                'Concentración Teórica': '{:.2f}',
                'Concentración Real': '{:.2f}',
                'Recuperación (%)': '{:.2f}%'
            }).apply(lambda row: ['color: #2ecc71' if 98 <= row['Recuperación (%)'] <= 102 else 'color: #e74c3c' for _ in row], axis=1)
        )
        # Agregar una muestra (primeras 10 filas) al PDF con redondeo a 2 decimales
        datos_tabla = datos.head(10).copy()
        tabla_data = [list(datos_tabla.columns)]
        for _, row in datos_tabla.iterrows():
            fila = []
            for valor in row:
                if isinstance(valor, (int, float)):
                    fila.append(f"{valor:<15.2f}")
                else:
                    fila.append(f"{str(valor):<15}")
            tabla_data.append(fila)
        pdf_gen.add_table(tabla_data, title="Datos Detallados (Primeras 10 filas)")
        
        # Botón de descarga (se asume que la función generar_descarga está definida)
        generar_descarga(datos)
    
    return True

def capture_figure(self, fig, fig_name=None):
    from io import BytesIO
    buf = BytesIO()
    import plotly.io as pio
    # Si la figura tiene el método savefig, asumimos que es Matplotlib
    if hasattr(fig, "savefig"):
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        # Si es una figura Plotly, usamos pio.to_image para obtener la imagen en PNG
        image_bytes = pio.to_image(fig, format="png", scale=2)
        buf.write(image_bytes)
    buf.seek(0)
    self.buffers.append(buf)

def agrupar_valores(valores, umbral=1.0):
    """
    Agrupa valores numéricos con una tolerancia especificada usando clustering básico.
    Cada grupo se forma si la diferencia entre el valor y la mediana del grupo es menor o igual al umbral.
    """
    grupos = []
    for valor in sorted(valores):
        agregado = False
        for grupo in grupos:
            if abs(valor - np.median(grupo)) <= umbral:
                grupo.append(valor)
                agregado = True
                break
        if not agregado:
            grupos.append([valor])
    return grupos

def evaluar_robustez(datos, pdf_gen):
    """
    Evalúa la robustez del método analítico agrupando concentraciones en rangos (±1 unidad)
    y analizando las respuestas de la señal según múltiples factores, siguiendo las guías CCYAC.
    
    Parámetros:
      - datos: DataFrame con columnas ['Respuesta', 'Concentración', 'Día'] y otros factores adicionales.
      - pdf_gen: Objeto PDFGenerator configurado para reportar los resultados.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Configuración visual inicial
    sns.set_theme(style="whitegrid", palette="pastel")
    st.header("🧪 Análisis de Robustez por Concentración (CCYAC)")
    pdf_gen.add_section_title("Análisis de Robustez por Concentración - CCYAC")

    # 1. Validación de columnas esenciales
    required_cols = ['Respuesta', 'Concentración', 'Día']
    for col in required_cols:
        if col not in datos.columns:
            st.error(f"Columna requerida faltante: {col}")
            pdf_gen.add_text_block(f"✘ Error: Falta columna {col}", style="error")
            return False

    # 2. Procesamiento y limpieza de datos
    try:
        # Conversión a tipo numérico
        for col in ['Respuesta', 'Concentración']:
            datos[col] = pd.to_numeric(datos[col], errors='coerce')
        
        # Conversión de fechas y cálculo de días relativos
        if not np.issubdtype(datos['Día'].dtype, np.datetime64):
            datos['Día'] = pd.to_datetime(datos['Día'], errors='coerce')
        datos['Día_num'] = (datos['Día'] - datos['Día'].min()).dt.days
        
        # Eliminar filas con valores faltantes en columnas esenciales
        datos = datos.dropna(subset=required_cols + ['Día_num'])
    except Exception as e:
        st.error(f"Error procesando datos: {str(e)}")
        pdf_gen.add_text_block(f"✘ Error en procesamiento: {str(e)}", style="error")
        return False

    # 3. Agrupación de concentraciones (±1 unidad) según CCYAC
    unique_conc = datos['Concentración'].dropna().unique()
    grupos_conc = agrupar_valores(unique_conc, umbral=1.0)
    
    # Crear un mapeo de cada valor a su grupo representativo (mediana del grupo)
    conc_mapping = {}
    for grupo in grupos_conc:
        representante = round(np.median(grupo), 2)
        for valor in grupo:
            conc_mapping[valor] = representante
    
    datos['Rango_Conc'] = datos['Concentración'].map(conc_mapping)
    # Convertir el rango a string formateado para facilitar la visualización
    datos['Rango_Conc'] = datos['Rango_Conc'].apply(lambda x: f"{x:.2f}")

    # 4. Identificar factores adicionales (excluyendo las columnas esenciales y auxiliares)
    factores = [col for col in datos.columns if col not in required_cols + ['Día_num', 'Rango_Conc']]
    if not factores:
        st.warning("No se detectaron factores adicionales para análisis")
        pdf_gen.add_text_block("Advertencia: No hay factores adicionales para evaluar", style="warning")
        return True

    # 5. Interfaz de usuario para configurar el análisis
    col1, col2 = st.columns(2)
    with col1:
        metodo = st.selectbox(
            "Tipo de método:",
            options=["Cromatográfico", "Químico", "Biológico"],
            index=0
        )
    with col2:
        factores_seleccionados = st.multiselect(
            "Factores a evaluar:",
            options=factores,
            default=factores[:2] if len(factores) >= 2 else factores
        )

    # 6. Definir umbral de aceptación según el tipo de método (guías CCYAC)
    umbral_map = {
        "Cromatográfico": 2.0,
        "Químico": 3.0,
        "Biológico": 5.0
    }
    umbral = umbral_map.get(metodo, 2.0)
    pdf_gen.add_text_block(f"Configuración:\n- Método: {metodo}\n- Umbral: {umbral}%", style="info")

    # 7. Análisis por rango de concentración
    resultados = []
    rangos_ordenados = sorted(datos['Rango_Conc'].unique(), key=lambda x: float(x))
    
    for rango in rangos_ordenados:
        df_rango = datos[datos['Rango_Conc'] == rango].copy()
        if df_rango.empty:
            continue

        # 7.1 Cálculo de estadísticos base para el rango
        media_global = df_rango['Respuesta'].mean()
        stats = {
            'Rango': rango,
            'Media_Global': media_global,
            'CV_Global': (df_rango['Respuesta'].std() / media_global * 100) if media_global != 0 else 0,
            'Factores': {}
        }

        # 7.2 Análisis y visualización para cada factor seleccionado
        fig, axs = plt.subplots(len(factores_seleccionados), 2, figsize=(12, 4*len(factores_seleccionados)))
        if len(factores_seleccionados) == 1:
            axs = [axs]
            
        for i, factor in enumerate(factores_seleccionados):
            # Agrupar el factor si es numérico
            if pd.api.types.is_numeric_dtype(df_rango[factor]):
                df_rango = df_rango.copy()  # Evitar advertencias de SettingWithCopy
                df_rango[f'{factor}_grupo'] = pd.cut(df_rango[factor], bins=3, precision=1).astype(str)
                col_factor = f'{factor}_grupo'
            else:
                col_factor = factor

            # Cálculo de métricas para cada subgrupo dentro del rango
            grupo_factor = df_rango.groupby(col_factor)['Respuesta']
            metricas = {
                'CV': (grupo_factor.std() / grupo_factor.mean() * 100).max(),
                'Dif_Media': abs((grupo_factor.mean() - media_global) / media_global * 100).max(),
                'N': grupo_factor.count().max()
            }
            stats['Factores'][factor] = metricas

            # Visualización 1: Boxplot
            ax = axs[i][0]
            sns.boxplot(x=col_factor, y='Respuesta', data=df_rango, ax=ax, palette="viridis")
            ax.axhline(media_global, color='r', linestyle='--', label='Media Global')
            ax.fill_between(
                ax.get_xlim(),
                media_global * (1 - umbral/100),
                media_global * (1 + umbral/100),
                color='gray',
                alpha=0.2
            )
            ax.set_title(f"{factor} - Rango: {rango}")
            ax.tick_params(axis='x', rotation=45)
            
            # Visualización 2: Stripplot para observar la dispersión de datos
            ax2 = axs[i][1]
            sns.stripplot(x=col_factor, y='Respuesta', data=df_rango, ax=ax2, jitter=True, palette="viridis")
            ax2.axhline(media_global, color='r', linestyle='--')
            ax2.set_title(f"Distribución: {factor} - Rango: {rango}")
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        pdf_gen.capture_figure(fig)
        st.pyplot(fig)  # Mostrar la figura en la página de Streamlit
        resultados.append(stats)

    # 7.3 Gráficos adicionales para evaluación global

    # Gráfico global: Distribución de Respuestas vs. Concentración con hue de Rango
    fig_global, ax_global = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=datos, x='Concentración', y='Respuesta', hue='Rango_Conc',
                    palette='viridis', ax=ax_global)
    ax_global.set_title("Distribución Global: Respuesta vs. Concentración")
    ax_global.legend(title="Rango Concentración")
    pdf_gen.capture_figure(fig_global)
    st.pyplot(fig_global)  # Mostrar el gráfico global

    # Crear DataFrame resumen para heatmaps
    resumen_rows = []
    for res in resultados:
        for factor, metricas in res['Factores'].items():
            resumen_rows.append({
                'Rango': res['Rango'],
                'Factor': factor,
                'CV': metricas['CV'],
                'Dif_Media': metricas['Dif_Media']
            })
    df_resumen = pd.DataFrame(resumen_rows)

    # Heatmap de CV
    if not df_resumen.empty:
        pivot_cv = df_resumen.pivot(index="Rango", columns="Factor", values="CV")
        fig_cv, ax_cv = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_cv, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_cv)
        ax_cv.set_title("Heatmap de CV por Rango y Factor")
        pdf_gen.capture_figure(fig_cv)
        st.pyplot(fig_cv)  # Mostrar heatmap de CV

        # Heatmap de Diferencia de Media
        pivot_dif = df_resumen.pivot(index="Rango", columns="Factor", values="Dif_Media")
        fig_dif, ax_dif = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_dif, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_dif)
        ax_dif.set_title("Heatmap de Dif. Media (%) por Rango y Factor")
        pdf_gen.capture_figure(fig_dif)
        st.pyplot(fig_dif)  # Mostrar heatmap de Diferencia de Media

    # 8. Generación de reporte final de robustez
    st.subheader("Resumen de Robustez")
    tabla_resumen = [["Rango", "Factor", "CV Máx (%)", "Dif. Media (%)", "Cumplimiento"]]
    
    for stats in resultados:
        for factor, metricas in stats['Factores'].items():
            cumplimiento = "✅" if metricas['CV'] <= umbral and metricas['Dif_Media'] <= umbral else "❌"
            tabla_resumen.append([
                stats['Rango'],
                factor,
                f"{metricas['CV']:.2f}",
                f"{metricas['Dif_Media']:.2f}",
                cumplimiento
            ])
    
    # Mostrar tabla en Streamlit
    st.dataframe(pd.DataFrame(tabla_resumen[1:], columns=tabla_resumen[0]))
    pdf_gen.add_table(tabla_resumen, title="Resumen de Robustez por Factor y Concentración")

    # Conclusión final según criterios de robustez (cumplimiento en todos los rangos y factores)
    todos_cumplen = all(row[-1] == "✅" for row in tabla_resumen[1:])
    conclusion = "✅ MÉTODO ROBUSTO" if todos_cumplen else "❌ REQUIERE OPTIMIZACIÓN"
    if todos_cumplen:
        st.success(conclusion)
    else:
        st.error(conclusion)
    pdf_gen.add_text_block(f"Conclusión Final: {conclusion}", style="conclusion")

    return True

def evaluar_estabilidad(datos, pdf_gen, test_type="assay"):
    """
    Evalúa la estabilidad del método comparando la respuesta analítica a lo largo del tiempo 
    con la medición inicial (baseline). Se agrupan automáticamente los valores de 'Concentración'
    utilizando una tolerancia de ±1 y, para cada grupo, se calcula el % de recuperación, la diferencia 
    respecto a 100% y el coeficiente de variación (CV).

    Además, se permite seleccionar interactivamente el tipo de método para definir el umbral de aceptación:
       - Cromatográfico: umbral de 2%
       - Químico: umbral de 3%
       - Biológico: umbral de 5%
       
    Parámetros:
      - datos (DataFrame): Debe contener al menos las columnas ['Día', 'Respuesta', 'Concentración'].
         Se asume que 'Día' es numérico o de tipo datetime (en cuyo caso se convierte a días transcurridos).
      - pdf_gen: Objeto para generar el PDF (con métodos como add_section_title, add_text_block, add_table, capture_figure, etc.).
      - test_type (str): Tipo de test; se incluye para compatibilidad (por defecto "assay").

    La función genera gráficos y tablas en Streamlit y documenta los resultados en el PDF.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns    
    sns.set_theme(style="whitegrid", palette="pastel")
    st.header("🧪 Análisis de Estabilidad por Grupo de Concentración")
    pdf_gen.add_section_title("Análisis de Estabilidad por Grupo de Concentración")
    pdf_gen.add_text_block(
        "Este reporte evalúa la estabilidad del método midiendo la respuesta analítica a lo largo del tiempo "
        "y comparándola con la medición inicial (baseline). Los datos se agrupan automáticamente en rangos de "
        "concentración utilizando una tolerancia de ±1 unidad. Para cada rango se calcula el % de recuperación, la "
        "diferencia respecto a 100% y el CV. Se permite seleccionar el tipo de método para definir el umbral de "
        "aceptación."
    )

    # 1. Validar columnas requeridas
    required_cols = ['Día', 'Respuesta', 'Concentración']
    for col in required_cols:
        if col not in datos.columns:
            st.error(f"Falta la columna '{col}' en el DataFrame.")
            pdf_gen.add_text_block(f"✘ Error: Falta la columna '{col}' en el DataFrame.", style="error")
            return False

    # 2. Conversión numérica y limpieza
    for col in ['Respuesta', 'Concentración']:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos.dropna(subset=['Respuesta', 'Concentración'], inplace=True)

    # 3. Procesar la columna 'Día'
    try:
        if np.issubdtype(datos['Día'].dtype, np.datetime64):
            datos['Día_num'] = (datos['Día'] - datos['Día'].min()).dt.days
        else:
            datos['Día_num'] = pd.to_numeric(datos['Día'], errors='coerce')
    except Exception as e:
        st.error(f"Error al convertir 'Día': {str(e)}")
        pdf_gen.add_text_block(f"✘ Error al convertir 'Día': {str(e)}", style="error")
        return False
    datos.dropna(subset=['Día_num'], inplace=True)

    # 4. Agrupar automáticamente los valores de concentración usando tolerancia ±1
    unique_vals = sorted(datos['Concentración'].unique())
    grupos_conc = agrupar_valores(unique_vals, umbral=1.0)
    mapping = {}
    for grupo in grupos_conc:
        rep = np.median(grupo)
        for v in grupo:
            mapping[v] = rep
    datos['Rango'] = datos['Concentración'].map(mapping).apply(lambda x: f"{x:.2f}")

    # Mostrar la agrupación para depuración
    st.subheader("Agrupación de Concentraciones")
    df_grupos = datos[['Concentración', 'Rango']].drop_duplicates().sort_values(by='Concentración')
    st.dataframe(df_grupos)
    tabla_grupos = [["Concentración", "Rango asignado"]]
    for _, row in df_grupos.iterrows():
        tabla_grupos.append([f"{row['Concentración']:.2f}", row['Rango']])
    pdf_gen.add_table(tabla_grupos, title="Agrupación de Concentraciones (Automática)")

    # 5. Seleccionar el tipo de método (para establecer umbral)
    metodo_seleccionado = st.selectbox(
        "Seleccione el tipo de método:",
        options=["cromatografico", "quimico", "biologico"],
        index=0
    )
    if metodo_seleccionado == "cromatografico":
        umbral = 2.0
    elif metodo_seleccionado == "quimico":
        umbral = 3.0
    elif metodo_seleccionado == "biologico":
        umbral = 5.0
    else:
        umbral = 2.0
    st.write(f"Umbral de aceptación para el método {metodo_seleccionado}: {umbral}%")
    pdf_gen.add_text_block(f"Tipo de método seleccionado: {metodo_seleccionado}. Umbral de aceptación: {umbral}%.", style="info")

    # 6. Permitir al usuario seleccionar uno o más factores variables adicionales (opcional)
    columnas_factor = [col for col in datos.columns if col not in ['Día', 'Día_num', 'Respuesta', 'Concentración', 'Rango']]
    factores_variables = st.multiselect("Seleccione factor(es) variables adicionales (opcional):", columnas_factor, default=[])
    pdf_gen.add_text_block(f"Factor(es) variable(s) seleccionado(s): {', '.join(factores_variables) if factores_variables else 'Ninguno'}", style="info")

    # 7. Iterar sobre cada rango y realizar el análisis de estabilidad
    resultados_totales = []
    rangos_disponibles = sorted(datos['Rango'].unique(), key=lambda x: float(x))
    for rango_label in rangos_disponibles:
        df_rango = datos[datos['Rango'] == rango_label].copy()
        if df_rango.empty:
            continue

        st.markdown(f"### Análisis para el Rango {rango_label}")
        pdf_gen.add_subsection(f"Rango {rango_label}")

        # Agrupar por día en este rango
        resumen = df_rango.groupby('Día_num')['Respuesta'].agg(['mean', 'std', 'count']).reset_index()
        resumen.rename(columns={'mean': 'Media', 'std': 'DE', 'count': 'N'}, inplace=True)

        # Baseline: se toma la medición del día inicial
        baseline_row = resumen.loc[resumen['Día_num'].idxmin()]
        baseline = baseline_row['Media']
        if baseline == 0:
            st.error("La media en el día inicial es 0, no es posible calcular la recuperación.")
            pdf_gen.add_text_block("✘ Error: La media en el tiempo inicial es 0.", style="error")
            continue

        resumen['Recuperación (%)'] = (resumen['Media'] / baseline) * 100
        resumen['Diferencia (%)'] = abs(resumen['Recuperación (%)'] - 100)
        resumen['CV'] = (resumen['DE'] / resumen['Media']) * 100

        st.dataframe(resumen)
        
        # Gráficos para este rango: Recuperación y CV vs Tiempo
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.lineplot(data=resumen, x='Día_num', y='Recuperación (%)', marker='o', ax=ax[0])
        ax[0].axhline(100, color='red', linestyle='--', label="100% (Baseline)")
        ax[0].axhline(100+umbral, color='gray', linestyle='--', label=f"100% ± {umbral}%")
        ax[0].axhline(100-umbral, color='gray', linestyle='--')
        ax[0].set_title(f"Recuperación (%) vs. Tiempo - Rango {rango_label}")
        ax[0].set_xlabel("Día (numérico)")
        ax[0].set_ylabel("Recuperación (%)")
        ax[0].legend()
        
        sns.barplot(data=resumen, x='Día_num', y='CV', ax=ax[1], palette='Blues', edgecolor='black')
        ax[1].axhline(umbral, color='red', linestyle='--', label=f"Umbral {umbral}%")
        ax[1].set_title(f"Variabilidad (CV) - Rango {rango_label}")
        ax[1].set_xlabel("Día (numérico)")
        ax[1].set_ylabel("CV (%)")
        ax[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        pdf_gen.capture_figure(fig, f"estabilidad_{rango_label}")
        plt.close(fig)
        
        # 8. Análisis complementario de los factores variables seleccionados (agrupación por factor)
        if factores_variables:
            for factor in factores_variables:
                st.markdown(f"#### Análisis del factor variable '{factor}' en el rango {rango_label}")
                pdf_gen.add_text_block(f"Análisis del factor variable '{factor}' en el rango {rango_label}.", style="info")
                df_factor = df_rango.dropna(subset=[factor])
                # Agrupación y cálculo de métricas para el factor
                resumen_factor = df_factor.groupby(factor)['Respuesta'].agg(['mean', 'std', 'count']).reset_index()
                resumen_factor.rename(columns={'mean': 'Media', 'std': 'DE', 'count': 'N'}, inplace=True)
                st.dataframe(resumen_factor)
                fig_factor, ax_factor = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df_factor, x=factor, y='Respuesta', ax=ax_factor, palette='viridis')
                ax_factor.set_title(f"Distribución de Respuesta por '{factor}' - Rango {rango_label}")
                plt.xticks(rotation=45)
                st.pyplot(fig_factor)
                pdf_gen.capture_figure(fig_factor, f"factor_{factor}_{rango_label}")
                plt.close(fig_factor)
        
        # Registro de métricas clave para este rango
        resultado = {
            'Rango': rango_label,
            'Recuperación Mínima (%)': f"{resumen['Recuperación (%)'].min():.2f}%",
            'Diferencia Máxima (%)': f"{resumen['Diferencia (%)'].max():.2f}%",
            'CV Máximo (%)': f"{resumen['CV'].max():.2f}%"
        }
        resultados_totales.append(resultado)
    
    # Reporte consolidado
    if resultados_totales:
        tabla_header = ["Rango", "Recup. Mínima", "Dif. Máxima", "CV Máximo"]
        tabla_data = [[r['Rango'], r['Recuperación Mínima (%)'], r['Diferencia Máxima (%)'], r['CV Máximo (%)']] for r in resultados_totales]
        pdf_gen.add_table([tabla_header] + tabla_data, title="Resumen de Estabilidad por Rango")
        todas_estables = all(float(r['Diferencia Máxima (%)'].replace('%','')) <= umbral for r in resultados_totales)
        conclusion = "El método es estable en todos los rangos." if todas_estables else "El método presenta variaciones superiores al umbral en algunos rangos."
        st.markdown(f"### Conclusión: {conclusion}")
        pdf_gen.add_text_block(f"Conclusión de estabilidad: {conclusion}", style="info")
        return resultados_totales
    else:
        st.warning("No se pudieron obtener resultados de estabilidad en ningún rango.")
        pdf_gen.add_text_block("⚠️ No se obtuvieron resultados de estabilidad en ningún rango.", style="warning")
        return False



if modulo == "Linealidad y Rango":
    with st.container():
        # Encabezado profesional
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; margin-bottom: 2rem;'>
                <h1 style='color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; display: inline-block;'>
                    Análisis de Linealidad y Rango
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Contenedor de dos columnas para la información y el área de carga
        col_info, col_upload = st.columns([1, 1], gap="large")

        with col_info:
           st.markdown("""
    <div style='background: rgba(46, 204, 113, 0.05); 
                padding: 1.5rem; 
                border-radius: 8px;
                border: 1px solid rgba(46, 204, 113, 0.2);'>
        <h3 style='color: #2ecc71; margin-top: 0;'>Estructura Requerida</h3>
        <div style='color: #bdc3c7; line-height: 1.6;'>
            <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                <div style='min-width: 30px; text-align: center;'></div>
                <div style='margin-left: 10px;'><strong>Día:</strong> Número o fecha de medición</div>
            </div>
            <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                <div style='min-width: 30px; text-align: center;'></div>
                <div style='margin-left: 10px;'><strong>Concentración:</strong> Valores numéricos</div>
            </div>
            <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                <div style='min-width: 30px; text-align: center;'></div>
                <div style='margin-left: 10px;'><strong>Respuesta:</strong> Mediciones instrumentales</div>
            </div>
            <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                <div style='min-width: 30px; text-align: center;'></div>
                <div style='margin-left: 10px;'><strong>Tipo:</strong> "Estándar" o "Muestra"</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


        with col_upload:
            st.markdown("""
                <style>
                    .upload-container-precision {
                        border: 2px dashed #2ecc71;
                        border-radius: 10px;
                        padding: 2rem;
                        text-align: center;
                        background: rgba(46, 204, 113, 0.03);
                        min-height: 150px;
                        display: flex !important;
                        flex-direction: column;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .upload-container-precision:hover {
                        background: rgba(46, 204, 113, 0.08);
                        transform: translateY(-2px);
                    }
                </style>
                <div class="upload-container-precision">
                    <div style="font-size: 1.5rem; color: #2ecc71; font-weight: 500;">Subir archivo</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 1rem;">Formatos soportados: CSV, Excel</div>
            """, unsafe_allow_html=True)
            datos = st.file_uploader(
                " ",
                type=['csv', 'xlsx'],
                key="uploader_precision",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Procesamiento del archivo en un contenedor separado
    if datos:
        with st.spinner('Analizando datos...'):
            if procesar_archivo(datos, calcular_linealidad, modulo):
                st.success("Análisis completado exitosamente")
                
                if st.session_state.get('current_pdf'):
                    st.download_button(
                        label="Descargar Reporte Completo",
                        data=st.session_state.current_pdf,
                        file_name=f"Reporte_{modulo.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                        
elif modulo == "Límites de Detección y Cuantificación":
    with st.container():
        # Encabezado profesional sin emojis
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; margin-bottom: 2rem;'>
                <h1 style='color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; display: inline-block;'>
                    Límites de Detección y Cuantificación
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Contenedor de dos columnas para la explicación y el área de carga
        col_info, col_upload = st.columns([1, 1], gap="large")

        with col_info:
            st.markdown("""
                <div style='background: rgba(46, 204, 113, 0.05);
                            padding: 1.5rem;
                            border-radius: 8px;
                            border: 1px solid rgba(46, 204, 113, 0.2);'>
                    <h3 style='color: #2ecc71; margin-top: 0;'>Estructura Requerida</h3>
                    <div style='color: #bdc3c7; line-height: 1.6;'>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                            <div style='min-width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Día:</strong> Serie temporal de mediciones</div>
                        </div>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                            <div style='min-width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Concentración:</strong> Valores de calibración</div>
                        </div>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                            <div style='min-width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Respuesta:</strong> Señal instrumental</div>
                        </div>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;'>
                            <div style='min-width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Tipo:</strong> Identificación estándar</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_upload:
            st.markdown("""
                <style>
                    .upload-container-precision {
                        border: 2px dashed #2ecc71;
                        border-radius: 10px;
                        padding: 2rem;
                        text-align: center;
                        background: rgba(46, 204, 113, 0.03);
                        min-height: 150px;
                        display: flex !important;
                        flex-direction: column;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .upload-container-precision:hover {
                        background: rgba(46, 204, 113, 0.08);
                        transform: translateY(-2px);
                    }
                </style>
                <div class="upload-container-precision">
                    <div style="font-size: 1.5rem; color: #2ecc71; font-weight: 500;">Subir archivo</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 1rem;">Formatos soportados: CSV, Excel</div>
            """, unsafe_allow_html=True)
            datos = st.file_uploader(
                " ",
                type=['csv', 'xlsx'],
                key="uploader_precision",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Procesamiento del archivo en un contenedor separado
    if datos:
        with st.spinner('Calculando límites analíticos...'):
            if procesar_archivo(datos, [calcular_lod_loq, graficar_curva_calibracion_streamlit], modulo):
                st.markdown("""
                    <div style='background: #27ae60;
                                color: white;
                                padding: 1rem;
                                border-radius: 8px;
                                margin: 2rem 0;
                                text-align: center;'>
                        <div style='font-size: 1.2rem;'>Proceso completado - LOD/LOQ calculados</div>
                    </div>
                """, unsafe_allow_html=True)

                if st.session_state.get('current_pdf'):
                    st.download_button(
                        label="Descargar Reporte Técnico",
                        data=st.session_state.current_pdf,
                        file_name="reporte_limites.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_limites"
                    )
# Módulo de Precisión

elif modulo == "Precisión (Repetibilidad e Intermedia)":
    with st.container():
        # Encabezado profesional
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem; margin-bottom: 2rem;">
                <h1 style="color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; display: inline-block;">
                    Evaluación de Precisión
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Contenedor de dos columnas para la estructura requerida y el área de carga
        col_info, col_upload = st.columns([1, 1], gap="large")

        with col_info:
            st.markdown("""
                <div style="background: rgba(46, 204, 113, 0.05);
                            padding: 1.5rem;
                            border-radius: 8px;
                            border: 1px solid rgba(46, 204, 113, 0.2);">
                    <h3 style="color: #2ecc71; margin-top: 0;">Estructura Requerida</h3>
                    <div style="color: #bdc3c7; line-height: 1.6;">
                        <div style="display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;">
                            <div style="min-width: 30px; text-align: center;"></div>
                            <div style="margin-left: 10px;"><strong>Día:</strong> Día de la medición o fecha</div>
                        </div>
                        <div style="display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;">
                            <div style="min-width: 30px; text-align: center;"></div>
                            <div style="margin-left: 10px;"><strong>Concentración:</strong> Valores numéricos</div>
                        </div>
                        <div style="display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;">
                            <div style="min-width: 30px; text-align: center;"></div>
                            <div style="margin-left: 10px;"><strong>Respuesta:</strong> Mediciones instrumentales</div>
                        </div>
                        <div style="display: flex; align-items: center; margin: 0.8rem 0; flex-wrap: wrap;">
                            <div style="min-width: 30px; text-align: center;"></div>
                            <div style="margin-left: 10px;"><strong>Tipo:</strong> "Estándar" o "Muestra"</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_upload:
            st.markdown("""
                <style>
                    .upload-container-precision {
                        border: 2px dashed #2ecc71;
                        border-radius: 10px;
                        padding: 2rem;
                        text-align: center;
                        background: rgba(46, 204, 113, 0.03);
                        min-height: 150px;
                        display: flex !important;
                        flex-direction: column;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .upload-container-precision:hover {
                        background: rgba(46, 204, 113, 0.08);
                        transform: translateY(-2px);
                    }
                </style>
                <div class="upload-container-precision">
                    <div style="font-size: 1.5rem; color: #2ecc71; font-weight: 500;">Subir archivo</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 1rem;">Formatos soportados: CSV, Excel</div>
            """, unsafe_allow_html=True)
            datos = st.file_uploader(
                " ",
                type=['csv', 'xlsx'],
                key="uploader_precision",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Procesamiento del archivo en un contenedor separado
    if datos:
        with st.spinner("Analizando datos..."):
            if procesar_archivo(datos, [calcular_precision_por_rango], "Precisión (Repetibilidad e Intermedia)"):
                st.success("Análisis completado.")
                if st.session_state.get('current_pdf'):
                    st.download_button(
                        label="Descargar Reporte",
                        data=st.session_state.current_pdf,
                        file_name="reporte_Precision.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


elif modulo == "Exactitud (Recuperación)":
    with st.container():
        # Encabezado profesional
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; margin-bottom: 2rem;'>
                <h1 style='color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; display: inline-block;'>
                    Evaluación de Exactitud
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Contenedor de dos columnas para la estructura requerida y el área de carga
        col_info, col_upload = st.columns([1, 1], gap="large")

        with col_info:
            st.markdown("""
                <div style='background: rgba(46, 204, 113, 0.05);
                            padding: 1.5rem;
                            border-radius: 8px;
                            border: 1px solid rgba(46, 204, 113, 0.2);'>
                    <h3 style='color: #2ecc71; margin-top: 0;'>Requisitos de Datos</h3>
                    <div style='color: #bdc3c7; line-height: 1.6;'>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0;'>
                            <div style='width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Día:</strong> Serie temporal de mediciones</div>
                        </div>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0;'>
                            <div style='width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Concentración Teórica:</strong> Valores de referencia fortificados</div>
                        </div>
                        <div style='display: flex; align-items: center; margin: 0.8rem 0;'>
                            <div style='width: 30px; text-align: center;'></div>
                            <div style='margin-left: 10px;'><strong>Concentración Real:</strong> Valores obtenidos experimentalmente</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_upload:
            st.markdown("""
                <style>
                    .upload-container-precision {
                        border: 2px dashed #2ecc71;
                        border-radius: 10px;
                        padding: 2rem;
                        text-align: center;
                        background: rgba(46, 204, 113, 0.03);
                        min-height: 150px;
                        display: flex !important;
                        flex-direction: column;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .upload-container-precision:hover {
                        background: rgba(46, 204, 113, 0.08);
                        transform: translateY(-2px);
                    }
                </style>
                <div class="upload-container-precision">
                    <div style="font-size: 1.5rem; color: #2ecc71; font-weight: 500;">Subir archivo</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 1rem;">Formatos soportados: CSV, Excel</div>
            """, unsafe_allow_html=True)
            
            datos = st.file_uploader(
                " ",
                type=['csv', 'xlsx'],
                key="uploader_limites",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Procesamiento y resultados
    if datos:
        with st.spinner('Calculando exactitud...'):
            if procesar_archivo(datos, [calcular_exactitud], modulo):
                st.markdown("""
                    <div style='background: #27ae60;
                                color: white;
                                padding: 1rem;
                                border-radius: 8px;
                                margin: 2rem 0;
                                display: flex;
                                align-items: center;
                                gap: 1rem;'>
                        <div style='font-size: 1.5rem;'>Éxito:</div>
                        <div>Análisis de exactitud completado</div>
                    </div>
                """, unsafe_allow_html=True)

                if 'current_pdf' in st.session_state:
                    st.download_button(
                        label="Descargar Reporte de Exactitud",
                        data=st.session_state.current_pdf,
                        file_name="reporte_exactitud.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_exactitud"
                    )

# Módulo de Robustez
elif modulo == "Robustez":
    with st.container():
        # Encabezado profesional
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem; margin-bottom: 2rem;">
                <h1 style="color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; display: inline-block;">
                    Evaluación de Robustez
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Contenedor de dos columnas para mostrar la estructura requerida y el área de carga
        col_info, col_upload = st.columns([1, 1], gap="large")

        with col_info:
            st.markdown("""
                <div style="background: rgba(46, 204, 113, 0.05);
                            padding: 1.5rem;
                            border-radius: 8px;
                            border: 1px solid rgba(46, 204, 113, 0.2);">
                    <h3 style="color: #2ecc71; margin-top: 0;">Requisitos de Datos</h3>
                    <div style="color: #bdc3c7; line-height: 1.6;">
                        <p><strong>Factores Variables:</strong> Datos que representan condiciones variables del experimento.</p>
                        <p><strong>Resultados:</strong> Valores obtenidos bajo dichas condiciones (por ejemplo, Respuesta analítica, parámetros experimentales, etc.).</p>
                        <p><strong>Columna Adicional:</strong> Se requiere que los datos incluyan al menos las columnas <em>Día</em>, <em>Concentración</em>, <em>Respuesta</em> y <em>Tipo</em> (para clasificar si son estándares o muestras).</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_upload:
            st.markdown("""
                <style>
                    .upload-container-robustez {
                        border: 2px dashed #2ecc71;
                        border-radius: 10px;
                        padding: 2rem;
                        text-align: center;
                        background: rgba(46, 204, 113, 0.03);
                        min-height: 150px;
                        display: flex !important;
                        flex-direction: column;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .upload-container-robustez:hover {
                        background: rgba(46, 204, 113, 0.08);
                        transform: translateY(-2px);
                    }
                </style>
                <div class="upload-container-robustez">
                    <div style="font-size: 1.5rem; color: #2ecc71; font-weight: 500;">Subir archivo</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 1rem;">Formatos soportados: CSV, Excel</div>
            """, unsafe_allow_html=True)
            datos = st.file_uploader(
                " ",
                type=['csv', 'xlsx'],
                key="uploader_robustez",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if datos:
        with st.spinner("Analizando datos..."):
            if procesar_archivo(datos, [evaluar_robustez], "Robustez del Metodo"):
                st.success("Análisis completado.")
                if st.session_state.get('current_pdf'):
                    st.download_button(
                        label="Descargar PDF",
                        data=st.session_state.current_pdf,
                        file_name="reporte_robustez.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

elif modulo == "Estabilidad":
    with st.container():
        # Encabezado profesional
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem; margin-bottom: 2rem;">
                <h1 style="color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; display: inline-block;">
                    Evaluación de Estabilidad
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Contenedor de dos columnas para la estructura requerida y el área de carga
        col_info, col_upload = st.columns([1, 1], gap="large")

        with col_info:
            st.markdown("""
                <div style="background: rgba(46, 204, 113, 0.05);
                            padding: 1.5rem;
                            border-radius: 8px;
                            border: 1px solid rgba(46, 204, 113, 0.2);">
                    <h3 style="color: #2ecc71; margin-top: 0;">Requisitos de Datos</h3>
                    <div style="color: #bdc3c7; line-height: 1.6;">
                        <p><strong>Día:</strong> Fecha o día numérico de la medición.</p>
                        <p><strong>Respuesta:</strong> Valores de la respuesta analítica.</p>
                        <p><strong>Tipo:</strong> Clasificación de la muestra (Ej: Estándar, Muestra).</p>
                        <p><strong>Factores Variables:</strong> Columnas adicionales para análisis multivariable.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_upload:
            st.markdown("""
                <style>
                    .upload-container-estabilidad {
                        border: 2px dashed #2ecc71;
                        border-radius: 10px;
                        padding: 2rem;
                        text-align: center;
                        background: rgba(46, 204, 113, 0.03);
                        min-height: 150px;
                        display: flex !important;
                        flex-direction: column;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .upload-container-estabilidad:hover {
                        background: rgba(46, 204, 113, 0.08);
                        transform: translateY(-2px);
                    }
                </style>
                <div class="upload-container-estabilidad">
                    <div style="font-size: 1.5rem; color: #2ecc71; font-weight: 500;">Subir archivo</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 1rem;">Formatos soportados: CSV, Excel</div>
            """, unsafe_allow_html=True)
            datos = st.file_uploader(
                " ",
                type=['csv', 'xlsx'],
                key="uploader_estabilidad",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if datos:
        with st.spinner("Analizando datos..."):
            if procesar_archivo(datos, [evaluar_estabilidad], "Estabilidad del Método"):
                st.success("Análisis completado.")
                if st.session_state.get('current_pdf'):
                    st.download_button(
                        label="Descargar PDF",
                        data=st.session_state.current_pdf.getvalue(),
                        file_name="reporte_estabilidad.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
