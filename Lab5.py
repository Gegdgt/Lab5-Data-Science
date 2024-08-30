# Importar las bibliotecas necesarias
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Descargar recursos de nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

# 1. Cargar el dataset
df = pd.read_csv('C:\\Users\\manue\\OneDrive\\Escritorio\\Data_Science\\Lab5\\nlp-getting-started\\train.csv')

# Descripción inicial de los datos
print("Descripción del conjunto de datos:")
print(df.info())
print("\nPrimeras filas del conjunto de datos:")
print(df.head())

# 2. Limpieza y preprocesamiento de los datos

# Convertir el texto a minúsculas
df['text'] = df['text'].str.lower()

# Lista personalizada de stopwords
custom_stopwords = set(stopwords.words('english')).union({"wa", "im", "amp", "u", "like", "ur"})

# Función para limpiar texto
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Eliminar URLs
    text = re.sub(r'\@w+|\#', '', text)  # Eliminar hashtags y menciones
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar signos de puntuación
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'\b(?:wa|im|amp|u|ur|like|get)\b', '', text)  # Eliminar palabras no informativas específicas
    return text

# Aplicar limpieza al texto
df['cleaned_text'] = df['text'].apply(clean_text)

# Lematización
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

df['cleaned_text'] = df['cleaned_text'].apply(lemmatize_text)