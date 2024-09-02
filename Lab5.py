import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from scipy.stats import uniform, randint

# Descargar recursos de nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Cargar el dataset
df = pd.read_csv('C:\\Users\\gegdg\\OneDrive\\Documentos\\.UVG\\Anio4\\Ciclo2\\Data_Science\\Lab5\\Lab5-Data-Science\\train.csv')
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

# Función para eliminar stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stopwords]
    return ' '.join(filtered_words)

# Aplicar eliminación de stopwords
df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

# Mostrar el dataframe después de la limpieza
print("\nDatos después del preprocesamiento:")
print(df[['text', 'cleaned_text']].head())

# 3. Análisis Exploratorio de Unigramas y Bigramas
# Separar tweets de desastres y no desastres
disaster_tweets = df[df['target'] == 1]['cleaned_text']
non_disaster_tweets = df[df['target'] == 0]['cleaned_text']

# Obtener la frecuencia de palabras (unigramas)
disaster_words = ' '.join(disaster_tweets).split()
non_disaster_words = ' '.join(non_disaster_tweets).split()

disaster_word_freq = Counter(disaster_words)
non_disaster_word_freq = Counter(non_disaster_words)

print("\nPalabras más comunes en tweets de desastres:")
print(disaster_word_freq.most_common(10))

print("\nPalabras más comunes en tweets de no desastres:")
print(non_disaster_word_freq.most_common(10))

# Crear nubes de palabras para tweets de desastres y no desastres
disaster_wordcloud = WordCloud(width=800, height=400).generate(' '.join(disaster_words))
non_disaster_wordcloud = WordCloud(width=800, height=400).generate(' '.join(non_disaster_words))

plt.figure(figsize=(10, 5))
plt.imshow(disaster_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras para Tweets de Desastres')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(non_disaster_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras para Tweets de No Desastres')
plt.show()

# Análisis de bigramas
def get_top_ngrams(corpus, n=None, ngram_range=(2, 2)):
    vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

print("\nBigramas más comunes en tweets de desastres:")
print(get_top_ngrams(disaster_tweets, n=10))

print("\nBigramas más comunes en tweets de no desastres:")
print(get_top_ngrams(non_disaster_tweets, n=10))

# 4. Preparación del conjunto de datos para el modelo de predicción
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['target'], test_size=0.2, random_state=42)

# Vectorización de texto
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Añadir nuevas características
df['length'] = df['text'].apply(len)
df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))
df['num_hashtags'] = df['text'].apply(lambda x: x.count('#'))

# Concatenar nuevas características con la matriz vectorizada
X_train_extra = hstack((X_train_vect, df.loc[X_train.index, ['length', 'num_exclamations', 'num_hashtags']].values))
X_test_extra = hstack((X_test_vect, df.loc[X_test.index, ['length', 'num_exclamations', 'num_hashtags']].values))

# 5. Modelos de Predicción
# Logistic Regression con RandomizedSearchCV
logreg_params = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l2', 'none'],
    'solver': ['lbfgs', 'saga']
}

logreg_search = RandomizedSearchCV(LogisticRegression(max_iter=200, random_state=42), 
                                   param_distributions=logreg_params, 
                                   n_iter=20, 
                                   cv=5, 
                                   scoring='accuracy', 
                                   random_state=42, 
                                   n_jobs=-1)

logreg_search.fit(X_train_vect, y_train)
print("Mejores hiperparámetros para Logistic Regression:")
print(logreg_search.best_params_)

y_pred_logreg_opt = logreg_search.best_estimator_.predict(X_test_vect)
print("\nInforme de clasificación para Logistic Regression Optimizado:")
print(classification_report(y_test, y_pred_logreg_opt))
print("\nMatriz de confusión para Logistic Regression Optimizado:")
print(confusion_matrix(y_test, y_pred_logreg_opt))