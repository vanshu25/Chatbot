import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#preprocessing

#step 1: tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#step 2: stemming
def stem(word):
    return stemmer.stem(word.lower())

#step 3: bag of words
def bag_of_words(tokenized_sentence, words):
     sentence_words = [stem(word) for word in tokenized_sentence]
     bag = np.zeros(len(words), dtype=np.float32)
     for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

     return bag