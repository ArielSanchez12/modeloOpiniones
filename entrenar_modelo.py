import json
import pandas as pd
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import fitz  # PyMuPDF para leer PDFs
from tqdm import tqdm

# ========== FUNCIONES AUXILIARES ==========
def leer_reseñas_pdf(ruta_pdf):
    reseñas = []
    if os.path.exists(ruta_pdf):
        doc = fitz.open(ruta_pdf)
        for page in doc:
            texto = page.get_text()
            for linea in texto.splitlines():
                if linea.strip():
                    reseñas.append(linea.strip())
    return reseñas

# ========== CARGAR CONFIGURACIÓN ==========
with open("config.json") as f:
    config = json.load(f)

# ========== MEDIR TIEMPO DE ENTRENAMIENTO ==========
inicio = time.time()

dataset2 = load_dataset("imdb")

# ========== CARGAR DATOS DESDE HUGGING FACE (LIMITADO) ========== 
print("Cargando datasets desde Hugging Face (máx 9,500 registros en total)...")
MAX_REGISTROS = 9_500  # máximo total de registros
textos = []
etiquetas = []
total = 0

def agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, max_total):
    restantes = max_total - len(textos)
    if restantes <= 0:
        return textos, etiquetas
    textos += nuevos_textos[:restantes]
    etiquetas += nuevas_etiquetas[:restantes]
    return textos, etiquetas

# Dataset 1: amazon_polarity
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

# Dataset 2: yelp_review_full (solo 1 y 5 estrellas para polaridad)
if len(textos) < MAX_REGISTROS:
    dataset3 = load_dataset("yelp_review_full")
    nuevos_textos = []
    nuevas_etiquetas = []
    for ejemplo in tqdm(dataset3["train"], desc="yelp_review_full"):
        if ejemplo["label"] == 0:  # 1 estrella = negativo
            nuevos_textos.append(ejemplo["text"])
            nuevas_etiquetas.append(0)
        elif ejemplo["label"] == 4:  # 5 estrellas = positivo
            nuevos_textos.append(ejemplo["text"])
            nuevas_etiquetas.append(1)
        if len(nuevos_textos) >= MAX_REGISTROS - len(textos):
            break
    textos, etiquetas = agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, MAX_REGISTROS)

# Dataset 3: tweet_eval
dataset = load_dataset("tweet_eval", "sentiment")
nuevos_textos = []
nuevas_etiquetas = []
for ejemplo in tqdm(dataset["train"], desc="tweet_eval"):
    if ejemplo["label"] != 1:  # 0 = negativa, 2 = positiva
        nuevos_textos.append(ejemplo["text"])
        nuevas_etiquetas.append(1 if ejemplo["label"] == 2 else 0)
    if len(nuevos_textos) >= MAX_REGISTROS - len(textos):
        break
textos, etiquetas = agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, MAX_REGISTROS)

# Dataset 4: imdb
if len(textos) < MAX_REGISTROS:
    dataset2 = load_dataset("imdb")
    nuevos_textos = []
    nuevas_etiquetas = []
    for ejemplo in tqdm(dataset2["train"], desc="imdb"):
        nuevos_textos.append(ejemplo["text"])
        nuevas_etiquetas.append(1 if ejemplo["label"] == 1 else 0)
        if len(nuevos_textos) >= MAX_REGISTROS - len(textos):
            break
    textos, etiquetas = agregar_registros(textos, etiquetas, nuevos_textos, nuevas_etiquetas, MAX_REGISTROS)

print(f"Total de registros cargados: {len(textos)}")

# ========== CARGAR DATOS DESDE PDF (opcional) ==========
"""

En este modelo no hace falta entrenar con PDFs, pero si se desea se pueden añadir opiniones de PDFs.
Yo saque lod dataset para entrenar el modelo desde Hugging Face, pero si quieres se puede agregar PDFs 
con contexto del proyecto.

Pero no hace falta porque los dataset que traje desde HUgging Face son suficientes para que el modelo puede predecir
si las opiniones son positivas o negativas.


"""

# Si aún hay espacio, agregar PDFs
if len(textos) < MAX_REGISTROS:
    positivas_pdf = leer_reseñas_pdf("reseñas_positivas.pdf") + leer_reseñas_pdf("otras_positivas.pdf")
    negativas_pdf = leer_reseñas_pdf("reseñas_negativas.pdf") + leer_reseñas_pdf("otras_negativas.pdf")
    restantes = MAX_REGISTROS - len(textos)
    total_pdf = len(positivas_pdf) + len(negativas_pdf)
    if total_pdf > restantes:
        # recortar para no pasarse
        positivas_pdf = positivas_pdf[:restantes//2]
        negativas_pdf = negativas_pdf[:restantes - len(positivas_pdf)]
    textos += positivas_pdf + negativas_pdf
    etiquetas += [1] * len(positivas_pdf) + [0] * len(negativas_pdf)

# ========== CREAR DATAFRAME ==========
df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas})
print(f"Total de reseñas cargadas: {len(df)}")


# ========== ENTRENAMIENTO ========== 
print("Entrenando modelo...")
X_train, X_test, y_train, y_test = train_test_split(df["texto"], df["etiqueta"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(
    ngram_range=tuple(config["ngram_range"]),
    min_df=config["min_df"]
)
X_train_vec = vectorizer.fit_transform(tqdm(X_train, desc="Vectorizando train"))
X_test_vec = vectorizer.transform(tqdm(X_test, desc="Vectorizando test"))

model = LogisticRegression(max_iter=config["max_iter"])
for _ in tqdm(range(1), desc="Entrenando modelo"):
    model.fit(X_train_vec, y_train)

with open("modelo_sentimiento.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
print("Modelo guardado como modelo_sentimiento.pkl")

# ========== MÉTRICAS ==========
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativa", "Positiva"], yticklabels=["Negativa", "Positiva"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.savefig("matriz_confusion.png")
plt.close()
print("Matriz de confusión guardada como matriz_confusion.png")
fin = time.time()
print(f"Tiempo total de entrenamiento y guardado: {fin - inicio:.2f} segundos")