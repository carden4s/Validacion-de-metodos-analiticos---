import pandas as pd
import numpy as np
import os

def generar_datos_validacion():
    """
    Genera datos para un Excel con concentraciones, absorbancias, tipo y día.
    Se asegura de que los datos sean consistentes con validación de métodos analíticos.
    """
    # Parámetros iniciales
    dias = ['Día 1', 'Día 2']
    concentraciones = [0.15, 1.0, 3.0]  # ppm
    replicas = 3  # Número de réplicas por concentración y tipo
    pendiente_real = 0.5  # Pendiente esperada de la curva
    intercepto_real = 0.05  # Intercepto de la curva
    std_dev_absorbancia = 0.01  # Variabilidad aleatoria (desviación estándar)

    # Lista para almacenar los datos
    datos = []

    # Generar datos para cada día
    for dia in dias:
        for concentracion in concentraciones:
            for replica in range(1, replicas + 1):
                # Generar absorbancias con ruido (para Estándar)
                absorbancia_estandar = (
                    pendiente_real * concentracion
                    + intercepto_real
                    + np.random.normal(0, std_dev_absorbancia)
                )
                datos.append({
                    'Día': dia,
                    'Concentración': concentracion,
                    'Absorbancia': round(absorbancia_estandar, 4),
                    'Tipo': 'Estándar'
                })

                # Generar absorbancias con mayor variabilidad (para Muestra)
                absorbancia_muestra = (
                    pendiente_real * concentracion
                    + intercepto_real
                    + np.random.normal(0, std_dev_absorbancia * 1.5)
                )
                datos.append({
                    'Día': dia,
                    'Concentración': concentracion,
                    'Absorbancia': round(absorbancia_muestra, 4),
                    'Tipo': 'Muestra'
                })

    # Convertir a DataFrame
    df_datos = pd.DataFrame(datos)

    # Guardar en el directorio actual
    ruta_actual = os.getcwd()
    nombre_archivo = os.path.join(ruta_actual, "datos_validacion_metodos.xlsx")
    df_datos.to_excel(nombre_archivo, index=False)

    print(f"Archivo Excel generado en: {nombre_archivo}")

# Ejecutar la función
if __name__ == "__main__":
    generar_datos_validacion()
