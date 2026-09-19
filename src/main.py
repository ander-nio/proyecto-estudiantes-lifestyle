import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# CONFIGURACIÓN DE RUTAS DINÁMICAS
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 'student_lifestyle_dataset.csv'
GRAPHICS_DIR = BASE_DIR.parent / 'graphics'
GRAPHICS_DIR.mkdir(exist_ok=True)

# CARGA DE DATASET

df = pd.read_csv(DATA_PATH)

print("--- Primeras Filas ---")
print(df.head())

print("\n--- Información de Columnas ---")
df.info()

print("\n--- Estadística General ---")
print(df.describe())


# CONFIGURACIÓN ESTÉTICA

sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11

# HISTOGRAMAS (Notas y Sueño)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['Grades'], kde=True, color='purple', ax=axes[0])
axes[0].set_title('Distribución de Notas (Grades)')
axes[0].set_xlabel('Nota')
axes[0].set_ylabel('Cantidad de Estudiantes')

sns.histplot(df['Sleep_Hours_Per_Day'], kde=True, color='teal', ax=axes[1])
axes[1].set_title('Distribución de Horas de Sueño al Día')
axes[1].set_xlabel('Horas de Sueño')
axes[1].set_ylabel('Cantidad de Estudiantes')

plt.tight_layout()
plt.savefig(GRAPHICS_DIR / 'distribucion_variables.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# SCATTER PLOT (Estudio vs Notas)
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Study_Hours_Per_Day',
    y='Grades',
    hue='Stress_Level',
    hue_order=['Low', 'Moderate', 'High'],
    palette={'Low': 'mediumseagreen', 'Moderate': 'orange', 'High': 'crimson'},
    alpha=0.7
)

sns.regplot(
    data=df,
    x='Study_Hours_Per_Day',
    y='Grades',
    scatter=False,
    color='black',
    line_kws={'linestyle': '--', 'linewidth': 2, 'label': 'Tendencia general'}
)

plt.title('Relación entre Horas de Estudio Diarias y Notas', fontsize=14, fontweight='bold')
plt.xlabel('Horas de Estudio al Día')
plt.ylabel('Notas (Grades)')
plt.legend(title='Nivel de Estrés')
plt.savefig(GRAPHICS_DIR / 'dispersion_estudio_notas.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# BOX PLOT (Estudio vs Notas)
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x='Stress_Level',
    y='Grades',
    order=['Low', 'Moderate', 'High'],
    palette='Blues'
)
plt.title('Distribución de Notas por Nivel de Estrés', fontsize=14, fontweight='bold')
plt.xlabel('Nivel de Estrés')
plt.ylabel('Notas (Grades)')
plt.savefig(GRAPHICS_DIR / 'boxplot_estres_notas', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# MODELO PREDICTIVO (Regresión Lineal)
features = [
    'Study_Hours_Per_Day',
    'Extracurricular_Hours_Per_Day',
    'Sleep_Hours_Per_Day',
    'Social_Hours_Per_Day',
    'Physical_Activity_Hours_Per_Day'
]

X = df[features]
y = df['Grades']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n==============================================")
print("--- RESULTADOS DEL MODELO PREDICTIVO ---")
print("==============================================")
print(f"R² (Coeficiente de Determinación): {r2:.4f}")
print(f"RMSE (Error Cuadrático Medio): {rmse:.4f}\n")

coef_df = pd.DataFrame({
    'Hábito / Variable': features,
    'Impacto en la Nota (Coeficiente)': model.coef_
})
print(coef_df.sort_values(by='Impacto en la Nota (Coeficiente)', ascending=False).to_string(index=False))