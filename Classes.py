class Token:

  def __init__(self, word, lemma, pos):
    self.word = word
    self.lemma = lemma
    self.pos = pos

class Document:
  
# Rappresenta un documento del dataset, contiene testo originale, sequenza di token annotati e label.

  def __init__(self, text, tokens, label):
    self.text = text
    self.tokens = tokens
    self.label = label
    self.features = {}
  
  def get_words(self):
      words = []
      for token in self.tokens:
          words.append(token.word)
      return words
      
  def get_lemmas(self):
      lemmas = []
      for token in self.tokens:
          lemmas.append(token.lemma)
      return lemmas
  
  def get_pos_tags(self):
      pos_tags = []
      for token in self.tokens:
          pos_tags.append(token.pos)
      return pos_tags
  
  def get_tokens(self):
    return self.tokens
  
def get_data_ann(x, nlp):
   
 # annota un dataframe con Stanza

   data_ann = []
   for index, row in x.iterrows():
     row["text"]   
     row["label"]
     doc = nlp(row["text"])
     temp_list = []
     for sent in doc.sentences:
        for word in sent.words:
            temp_list.append(Token(word.text, word.lemma, word.upos))
     d = Document(row["text"],temp_list, row["label"])
     data_ann.append(d)

   return data_ann

from collections import Counter, defaultdict

def extract_word_ngrams(document, item, n):

# estrae n-grams a livello di token da un documento

    if item == "word":
        all_words = document.get_words()
    elif item == "lemma":
        all_words = document.get_lemmas()
    elif item == "pos":
        all_words = document.get_pos_tags()
    else:
        raise ValueError(f"Invalid item: {item}")

    ngrams = defaultdict(int)
    for i in range(0, len(all_words) - n + 1):
        ngram_items = all_words[i : i + n]
        ngram = f"{item}_{n}_" + "_".join(ngram_items)
        ngrams[ngram] += 1

    return dict(ngrams)

def extract_char_ngrams(document, n):

# estrae n-grams di carattere

    all_words = " ".join(document.get_words())

    ngrams = defaultdict(int)
    for i in range(0, len(all_words) - n + 1):
        ngram_chars = all_words[i : i + n]
        ngram = f"CHAR_{n}_" + ngram_chars
        ngrams[ngram] += 1

    return dict(ngrams)

from math import ceil

def normalize_ngrams(ngrams, max_len):

# converte i conteggi degli n-grammi in frequenze relative

    for ngram in ngrams:
        ngrams[ngram] = ngrams[ngram] / float(max_len)
    return ngrams

def filter_features(train_documents, val_documents, test_documents, min_perc=0.0, max_perc=1.0):

 # filtra le feature in base alla document frequency

    min_occ = len(train_documents) * min_perc
    max_occ = len(train_documents) * max_perc

   
    feature_counts = Counter()
    for doc in train_documents:
        for term in doc.features.keys():  
            feature_counts[term] += 1

    filtered_features = {feature for feature, count in feature_counts.items() if min_occ <= count <= max_occ}
    print(f'Rimosse feature che compaiono in meno di {ceil(min_occ)} documenti e in più di {ceil(max_occ)} documenti.\nSi passa da {len(list(feature_counts.keys()))} a {len(list(filtered_features))} features.')

    
    for document in train_documents:
        document.features = {feature: count for feature, count in document.features.items() if feature in filtered_features}
    
    for document in val_documents:
        document.features = {feature: count for feature, count in document.features.items() if feature in filtered_features}
    
    for document in test_documents:
        document.features = {feature: count for feature, count in document.features.items() if feature in filtered_features}
    
    return train_documents, val_documents, test_documents

import re

# normalizzatori di cifre e stringhe

def get_digits(text):
    try:
      val = int(text)
    except:
      text = re.sub(r'\d', '@Dg', text)
      return text
    if val >= 0 and val < 2100:
      return str(val)
    else:
      return "DIGLEN_" + str(len(str(val)))

def normalize_text(word):
    if "http" in word or ("." in word and "/" in word):
      word = str("___URL___")
      return word
    if len(word) > 26:
      return "__LONG-LONG__"
    new_word = get_digits(word)
    if new_word != word:
      word = new_word
    if word[0].isupper():
      word = word.capitalize()
    else:
      word = word.lower()
    return word


# Funzione per caricare gli embeddings del database sqlite in un dizionario

import sqlite3
import numpy as np

def load_embeddings_from_sqlite(db_path):
    embeddings = dict()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for row in c.execute("SELECT * FROM store"):
        word = row[0]
        vector = np.array(row[1:-1])
        embeddings[word] = vector
    conn.close()
    return embeddings
    



     
            
        
    

    



  