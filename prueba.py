from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from textblob import TextBlob
from googletrans import Translator
import re
import nltk
import torch
from nltk.corpus import stopwords
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

app = Flask(__name__)
api = Api(app)

translator = Translator()

# Cargar el tokenizador BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Cargar las groserías tokenizadas desde el archivo
with open('tokenized_groserias.txt', 'r') as f:
    tokenized_groserias = set(f.read().splitlines())

# Cargar el modelo y el tokenizador GPT-2 de Hugging Face
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Función para corregir ortografía usando GPT-2
def corregir_ortografia(texto):
    # Preprocesar el texto para reemplazar números por letras equivalentes
    texto = texto.replace("4", "a").replace("3", "e").replace("1", "i").replace("0", "o")
    
    # Tokenizar el texto de entrada
    inputs = gpt2_tokenizer.encode(texto, return_tensors="pt")
    
    # Generar texto completado
    outputs = gpt2_model.generate(inputs, max_length=len(inputs[0]) + 1, num_return_sequences=1)
    
    # Decodificar la salida generada
    corregido = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Cortar la salida generada para que coincida con la longitud de entrada
    corregido = corregido[:len(texto)]
    
    return corregido

class SentimentAnalysis(Resource):
    def preprocess_text(self, text):
        # Corregir la ortografía antes de cualquier otro procesamiento
        text = corregir_ortografia(text)

        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar puntuaciones, pero mantener signos de exclamación e interrogación
        text = re.sub(r'[^\w\s]', '', text)
        
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        
        # Remover tildes de forma cuidadosa
        text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        
        # Eliminar stopwords
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        # Eliminar espacios vacíos
        text = text.strip()
        
        return text

    def detect_groserias(self, text):
        tokens = bert_tokenizer.tokenize(text)
        for token in tokens:
            if token in tokenized_groserias:
                return True
        return False

    def post(self):
        data = request.get_json()
        text = data['text']
        
        # Preprocesamiento del texto
        cleaned_text = self.preprocess_text(text)

        if self.detect_groserias(cleaned_text):
            return jsonify({
                'error': 'El texto contiene groserías y no puede ser procesado.'
            })

        # Traducción del texto al inglés
        translated_text = translator.translate(cleaned_text, src='es', dest='en').text
        
        # Análisis de sentimiento
        blob = TextBlob(translated_text)
        sentiment = blob.sentiment
        
        return jsonify({
            'original_text': text,
            'cleaned_text': cleaned_text,
            'translated_text': translated_text,
            'sentiment': {
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity
            }
        })

api.add_resource(SentimentAnalysis, '/sentiment')

if __name__ == '__main__':
    app.run(debug=True)
