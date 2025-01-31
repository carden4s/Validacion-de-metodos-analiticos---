import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

def generar_dataset_realista():
    np.random.seed(42)
    dias = ['19/02/2024', '20/02/2024']
    conc_estandares = [0.1, 0.5, 1.0, 2.0, 3.0]  # 5 puntos de calibración
    conc_muestras = [0.3, 0.8, 1.5]  # Concentraciones diferentes a estándares
    
    data = []
    for dia in dias:
        # Generar estándares
        for conc in conc_estandares:
            for _ in range(3): # 3 réplicas
                abs_value = 0.5 * conc + np.random.normal(0, 0.015)
                data.append({
                    'Día': dia,
                    'Concentración': conc,
                    'Absorbancia': round(abs_value, 3),
                    'Tipo': 'Estándar'
                })
        
        # Generar muestras
        for conc in conc_muestras:
            for _ in range(3):
                abs_value = 0.5 * conc + np.random.normal(0, 0.02)
                data.append({
                    'Día': dia,
                    'Concentración': conc,
                    'Absorbancia': round(abs_value, 3),
                    'Tipo': 'Muestra'
                })
    
    return pd.DataFrame(data)
    
    # Crear archivo Excel
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    
    st.download_button(
        label="⬇️ Descargar Dataset de Prueba",
        data=output,
        file_name="dataset_uvvis_test.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Llamar a la función en tu app Streamlit
generar_dataset_realista()