# ===== IMPORTACIÓN DE LIBRERÍAS =====
import json  # Para leer archivos de configuración
import pandas as pd  # Para manipular datos en forma tabular
import pickle  # Para guardar modelos y objetos en archivos binarios
import time  # Para medir el tiempo de ejecución
import os  # Para interactuar con el sistema de archivos
from sklearn.feature_extraction.text import TfidfVectorizer  # Convertir texto en vectores numéricos
from sklearn.linear_model import LogisticRegression  # Algoritmo de clasificación
from sklearn.model_selection import train_test_split  # División de datos en entrenamiento y prueba
from sklearn.metrics import classification_report, confusion_matrix  # Evaluación del modelo
import matplotlib.pyplot as plt  # Visualización de gráficas
import seaborn as sns  # Mejorar las gráficas con estilos
from datasets import load_dataset  # Cargar datasets desde Hugging Face
import fitz  # PyMuPDF para leer texto desde archivos PDF
from tqdm import tqdm  # Barra de progreso visual

# ===== FUNCIONES AUXILIARES =====
def leer_reseñas_pdf(ruta_pdf):
    # Lee línea por línea los textos de un PDF
    reseñas = []
    if os.path.exists(ruta_pdf):
        doc = fitz.open(ruta_pdf)
        for page in doc:
            texto = page.get_text()
            for linea in texto.splitlines():
                if linea.strip():
                    reseñas.append(linea.strip())
    return reseñas

# ===== CARGAR CONFIGURACIÓN =====
with open("config.json") as f:
    config = json.load(f)  # Lee configuración como número de iteraciones, n-gramas, etc.

# ===== MEDIR TIEMPO DE EJECUCIÓN TOTAL =====
inicio = time.time()

# ===== CARGAR DATOS DESDE HUGGING FACE =====
print("Cargando datasets desde Hugging Face (máx 500,000 registros en total)...")
MAX_REGISTROS = 500_000  # Límite para evitar exceso de RAM
textos = []  # Lista de textos
etiquetas = []  # Lista de etiquetas (0 = negativa, 1 = positiva)
total = 0

# Función que agrega registros sin pasarse del límite total
def agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, max_total):
    restantes = max_total - len(textos)
    if restantes <= 0:
        return textos, etiquetas
    textos += nuevos_textos[:restantes]
    etiquetas += nuevas_etiquetas[:restantes]
    return textos, etiquetas

# Dataset 1: yelp_review_full (solo reseñas muy positivas y muy negativas)
if len(textos) < MAX_REGISTROS:
    dataset3 = load_dataset("yelp_review_full")
    nuevos_textos = []
    nuevas_etiquetas = []
    for ejemplo in tqdm(dataset3["train"], desc="yelp_review_full"):
        if ejemplo["label"] == 0:
            nuevos_textos.append(ejemplo["text"])
            nuevas_etiquetas.append(0)
        elif ejemplo["label"] == 4:
            nuevos_textos.append(ejemplo["text"])
            nuevas_etiquetas.append(1)
        if len(nuevos_textos) >= MAX_REGISTROS - len(textos):
            break
    textos, etiquetas = agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, MAX_REGISTROS)

# Dataset 2: tweet_eval (solo polaridad: negativa o positiva)
dataset = load_dataset("tweet_eval", "sentiment")
nuevos_textos = []
nuevas_etiquetas = []
for ejemplo in tqdm(dataset["train"], desc="tweet_eval"):
    if ejemplo["label"] != 1:  # 1 = neutro, no lo usamos
        nuevos_textos.append(ejemplo["text"])
        nuevas_etiquetas.append(1 if ejemplo["label"] == 2 else 0)
    if len(nuevos_textos) >= MAX_REGISTROS - len(textos):
        break
textos, etiquetas = agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, MAX_REGISTROS)

# Dataset 3: amazon_polarity
if len(textos) < MAX_REGISTROS:
    dataset4 = load_dataset("amazon_polarity")
    nuevos_textos = []
    nuevas_etiquetas = []
    for ejemplo in tqdm(dataset4["train"], desc="amazon_polarity"):
        nuevos_textos.append(ejemplo["content"])
        nuevas_etiquetas.append(ejemplo["label"])
        if len(nuevos_textos) >= MAX_REGISTROS - len(textos):
            break
    textos, etiquetas = agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, MAX_REGISTROS)

print(f"Total de registros cargados: {len(textos)}")

# ===== CARGAR DATOS DESDE PDF (opcional) =====
# Aunque entrenamos con Hugging Face, opcionalmente podemos cargar reseñas adicionales desde PDFs.
if len(textos) < MAX_REGISTROS:
    positivas_pdf = leer_reseñas_pdf("reseñas_positivas.pdf") + leer_reseñas_pdf("otras_positivas.pdf")
    negativas_pdf = leer_reseñas_pdf("reseñas_negativas.pdf") + leer_reseñas_pdf("otras_negativas.pdf")
    restantes = MAX_REGISTROS - len(textos)
    total_pdf = len(positivas_pdf) + len(negativas_pdf)
    if total_pdf > restantes:
        positivas_pdf = positivas_pdf[:restantes//2]
        negativas_pdf = negativas_pdf[:restantes - len(positivas_pdf)]
    textos += positivas_pdf + negativas_pdf
    etiquetas += [1] * len(positivas_pdf) + [0] * len(negativas_pdf)

# ===== CREAR DATAFRAME PARA ENTRENAMIENTO =====
df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas})
print(f"Total de reseñas cargadas: {len(df)}")

# ===== ENTRENAMIENTO DEL MODELO =====
print("Entrenando modelo...")
X_train, X_test, y_train, y_test = train_test_split(df["texto"], df["etiqueta"], test_size=0.2, random_state=42)

# Convertimos el texto en vectores usando TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=tuple(config["ngram_range"]),
    min_df=config["min_df"]
)
X_train_vec = vectorizer.fit_transform(tqdm(X_train, desc="Vectorizando train"))
X_test_vec = vectorizer.transform(tqdm(X_test, desc="Vectorizando test"))

# Entrenamos modelo de regresión logística
model = LogisticRegression(max_iter=config["max_iter"])
for _ in tqdm(range(1), desc="Entrenando modelo"):
    model.fit(X_train_vec, y_train)

# Guardamos modelo
with open("modelo_sentimiento.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
print("Modelo guardado como modelo_sentimiento.pkl")

# ===== MÉTRICAS DE EVALUACIÓN =====
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Crear y guardar matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativa", "Positiva"], yticklabels=["Negativa", "Positiva"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.savefig("matriz_confusion.png")
plt.close()
print("Matriz de confusión guardada como matriz_confusion.png")

# Mostrar tiempo total de entrenamiento
total = time.time() - inicio
print(f"Tiempo total de entrenamiento y guardado: {total:.2f} segundos")