# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:37 2020

@author: Houcine's laptop
"""
import numpy as np
import re

def naive_sex_classifier(text, quantile=0.85, mode=1):
    """ - Assuming that the major information we are looking for is
    contained in big paragraphs (sentences of length more than the q quartile of the
    lengths in the text) ==> Filter out everything else
        - We suppose the name of the victim is anonymized with a capital letter followed by "..." 
        - Naive count of masculine key words and feminine keywords
        - Prediction
        
    mode 1 = Filtering then regex matching
    mode 2 = Regex matching then filtering
    """
    if mode == 1 :

        # Filtering phase
        lens_sentences = []
        for sen in text:
            lens_sentences.append(len(sen))
        filtered_text = [sen for sen in text if len(sen)>=np.quantile(lens_sentences, q=quantile)]

        # Regex matching the capital letter + ...
        filtered_text_2 = []
        for sen in filtered_text :
            if re.match("[A-Z](.*)\.\.\.", sen) is not None :
                filtered_text_2.append(sen)
        if len(filtered_text_2)==0:
            filtered_text_2 = filtered_text
    else :
        # Regex matching the capital letter + ...
        filtered_text = []
        for sen in text :
            if re.match("[A-Z](.*)\.\.\.", sen) is not None :
                filtered_text.append(sen)
        if len(filtered_text)==0:
            filtered_text = text
        # Filtering phase   
        lens_sentences = []
        for sen in filtered_text:
            lens_sentences.append(len(sen))
        filtered_text_2 = [sen for sen in text if len(sen)>=np.quantile(lens_sentences, q=quantile)]
        

    # Naive counting 
    masculine_words = ["Monsieur", "il", "né", "Me", "Mr"]
    feminine_words = ["Madame", "elle", "née", "Mme", "Mademoiselle"]

    homme_count = 0
    femme_count = 0
    for sen in filtered_text_2 :
        for word in masculine_words :
            if word in sen :
                homme_count+=1
        for word in feminine_words :
            if word in sen :
                femme_count+=1
    if homme_count > femme_count :
        return 'homme'
    else :
        return "femme"