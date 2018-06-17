import sys
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import numpy as np
from BeautifulSoup import BeautifulSoup
import re

def get_tokens(text):
    text = BeautifulSoup(text.encode('utf-8').decode('ascii', 'ignore')).text
    return word_tokenize(str(text).lower())

def get_indices(tokens, vocab):
    indices = np.zeros([len(tokens)], dtype=np.int32)
    UNK = "<unk>"
    for i, w in enumerate(tokens):
        try:
            indices[i] = vocab[w]
        except:
            indices[i] = vocab[UNK]
    return indices

def get_similarity(a_indices, b_indices, word_embeddings):
    a_embeddings = [word_embeddings[idx] for idx in a_indices]
    b_embeddings = [word_embeddings[idx] for idx in b_indices]
    avg_a_embedding = np.mean(a_embeddings, axis=0)
    avg_b_embedding = np.mean(b_embeddings, axis=0)
    cosine_similarity = np.dot(avg_a_embedding, avg_b_embedding)/(np.linalg.norm(avg_a_embedding) * np.linalg.norm(avg_b_embedding))
    return cosine_similarity

def remove_urls(text):
    r = re.compile(r"(http://[^ ]+)")
    text = r.sub("", text) #remove urls so that ? is not identified in urls
    r = re.compile(r"(https://[^ ]+)")
    text = r.sub("", text) #remove urls so that ? is not identified in urls
    r = re.compile(r"(http : //[^ ]+)")
    text = r.sub("", text) #remove urls so that ? is not identified in urls
    r = re.compile(r"(https : //[^ ]+)")
    text = r.sub("", text) #remove urls so that ? is not identified in urls
    return text

def is_too_short_or_long(tokens):
    text = ' '.join(tokens)
    r = re.compile('[^a-zA-Z ]+')
    text = r.sub('', text)
    tokens = text.split()
    if len(tokens) < 3 or len(tokens) > 100:
        return True
    return False
