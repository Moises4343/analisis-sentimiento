import pandas as pd
from transformers import BertTokenizer

# Cargar el dataset de groserías desde data.csv
df = pd.read_csv('data.csv')
groserias = df['groseria'].tolist()

# Inicializar el tokenizador de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenizar las groserías
tokenized_groserias = set()
for groseria in groserias:
    tokens = tokenizer.tokenize(groseria)
    tokenized_groserias.update(tokens)

# Guardar las groserías tokenizadas en un archivo para su uso posterior
tokenized_groserias = list(tokenized_groserias)
with open('tokenized_groserias.txt', 'w') as f:
    for token in tokenized_groserias:
        f.write(f"{token}\n")
