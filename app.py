from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from textblob import TextBlob
from googletrans import Translator
import re
import nltk
import torch
from nltk.corpus import stopwords
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

app = Flask(__name__)
api = Api(app)

translator = Translator()

# Cargar el tokenizador BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Cargar las groserías tokenizadas desde el archivo
groserias_path = 'tokenized_groserias.txt'
if not os.path.exists(groserias_path):
    raise FileNotFoundError(f"{groserias_path} no se encuentra en el directorio actual.")
with open(groserias_path, 'r', encoding='iso-8859-1') as f:
    tokenized_groserias = set(f.read().splitlines())

# Variables globales para los modelos GPT-2
gpt2_model = None
gpt2_tokenizer = None

# Función para cargar el modelo y el tokenizador GPT-2 si no están cargados
def load_gpt2_model():
    global gpt2_model, gpt2_tokenizer
    if gpt2_model is None or gpt2_tokenizer is None:
        gpt2_model_name = "gpt2"
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Función para corregir ortografía usando GPT-2
def corregir_ortografia(texto):
    load_gpt2_model()  # Cargar el modelo si no está cargado
    texto = texto.replace("4", "a").replace("3", "e").replace("1", "i").replace("0", "o")
    inputs = gpt2_tokenizer.encode(texto, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    outputs = gpt2_model.generate(
        inputs, 
        attention_mask=attention_mask, 
        max_length=len(inputs[0]) + 1, 
        num_return_sequences=1,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    corregido = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    corregido = corregido[:len(texto)]
    return corregido

class SentimentAnalysis(Resource):
    def preprocess_text(self, text):
        text = corregir_ortografia(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        text = ' '.join([word for word in text.split() if word not in stop_words])
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
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No se proporcionó texto para el análisis.'}), 400
        
        try:
            cleaned_text = self.preprocess_text(text)

            if self.detect_groserias(cleaned_text):
                return jsonify({'error': 'El texto contiene groserías y no puede ser procesado.'}), 400

            translated_text = translator.translate(cleaned_text, src='es', dest='en').text
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
        except Exception as e:
            return jsonify({'error': str(e)}), 500

api.add_resource(SentimentAnalysis, '/sentiment')

@app.route('/')
def home():
    return 'Bienvenido a la API de Análisis de Sentimientos'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
