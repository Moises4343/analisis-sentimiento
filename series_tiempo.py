import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Cargar datos
df = pd.read_csv('sentimientos_data.csv', parse_dates=['date'], index_col='date')

# Graficar series de tiempo
plt.figure(figsize=(12, 6))
plt.plot(df['polarity'], label='Polaridad')
plt.plot(df['subjectivity'], label='Subjetividad')
plt.plot(df['calificacion'], label='Calificación')
plt.title('Series de Tiempo de Polaridad, Subjetividad y Calificación')
plt.xlabel('Fecha')
plt.ylabel('Valores')
plt.legend()
plt.savefig('series_de_tiempo.png')
plt.show()

# Descomposición estacional de la polaridad
result_polarity = seasonal_decompose(df['polarity'], model='additive')
result_polarity.plot()
plt.title('Descomposición de la Polaridad')
plt.savefig('series_de_tiempo polaridad.png')
plt.show()

# Descomposición estacional de la subjetividad
result_subjectivity = seasonal_decompose(df['subjectivity'], model='additive')
result_subjectivity.plot()
plt.title('Descomposición de la Subjetividad')
plt.savefig('series_de_tiempo subjetividad.png')
plt.show()

# Descomposición estacional de la calificación
result_calificacion = seasonal_decompose(df['calificacion'], model='additive')
result_calificacion.plot()
plt.title('Descomposición de la Calificación')
plt.savefig('series_de_tiempo calificacion.png')
plt.show()
