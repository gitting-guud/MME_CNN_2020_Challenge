# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:44:49 2020

@author: Houcine's laptop
"""
import re
import unidecode
from nltk.corpus import stopwords


stop_words = set(stopwords.words('french'))

def process_text(text, stem=False):
    """ lowercase, removes stopwords, accents and lemmatizes the tokens if stem=True
    used with the df.apply() to create a new column on a dataframe
    """
    
    text_clean = []
    for sen in text : 
#         sen = unidecode.unidecode(sen.replace("’", " ").replace(","," ").replace("."," ").replace(";"," ").lower())
        sen = unidecode.unidecode(sen.replace("’", " ").replace(","," ").replace(";"," ").lower()) # keep the dots for the date_uniformizer
        sen = sen.replace("/ ","/") #some dates are in DD/ MM/ yyyy format
        tokens = sen.split()
        if stem :
            from nltk.stem.snowball import FrenchStemmer
            stemmer = FrenchStemmer()
            tokens_no_stpwrd = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]
        else :
#             tokens_no_stpwrd = [tok for tok in tokens if (tok not in stop_words) & (tok.isalnum())]
            tokens_no_stpwrd = [tok for tok in tokens if (tok not in stop_words)]

        no_letters = re.sub(' [a-z] ', " ", " ".join(tokens_no_stpwrd))
        
        text_clean.append(no_letters)

    return text_clean