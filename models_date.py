# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:21:32 2020

@author: Houcine's laptop
"""

from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import fr_core_news_md
nlp = fr_core_news_md.load()



def date_prediction_classifier_Word_vectors(text, clf, threshold=0.7):
    
    probas_ = []
    sentences_to_test = extract_X_sentences_around_all_dates(text, terms_discarding_the_date)
    for tup in sentences_to_test :
        processed_sentence = tup[0]
        WV_processed_sentence = nlp(processed_sentence).vector
        probas_.append(clf.predict_proba([WV_processed_sentence]))

    probas_ = np.array(probas_)
    if probas_.shape[0] == 0 :
        return "n.c."
    if probas_.shape[0] == 1 :
        probas_ = probas_[0]
    else :
        probas_ = np.squeeze(probas_)[:,1]
    
    if probas_.max() >= threshold :
        return letter_date_to_submission_date(sentences_to_test[np.argmax(probas_)][1])
    else :
        return "n.c."
    
def date_prediction_classifier_tfidf(text, clf, vectorizer, threshold=0.7):
    
    probas_ = []
    sentences_to_test = extract_X_sentences_around_all_dates(text, terms_discarding_the_date)
    for tup in sentences_to_test :
        processed_sentence = tup[0]
        tfidf_encoded_processed_sentence = vectorizer.transform([processed_sentence])
        probas_.append(clf.predict_proba(tfidf_encoded_processed_sentence))

    probas_ = np.array(probas_)
    
    if probas_.shape[0] == 0 :
        return "n.c."
    if probas_.shape[0] == 1 :
        probas_ = probas_[0][0]
        if probas_[1] >= threshold :
            return letter_date_to_submission_date(sentences_to_test[0][1])
        else :
            return "n.c."
        
    else :
        probas_ = np.squeeze(probas_)[:,1]
    
        if probas_.max() >= threshold :
            return letter_date_to_submission_date(sentences_to_test[np.argmax(probas_)][1])
        else :
            return "n.c."
    
def date_prediction_classifier_Word_vectors_multiclass(text, clf, threshold=0.35):
    
    probas_ = []
    sentences_to_test = extract_X_sentences_around_all_dates(text, terms_discarding_the_date)
    for tup in sentences_to_test :
        processed_sentence = tup[0]
        WV_processed_sentence = nlp(processed_sentence).vector
        probas_.append(clf.predict_proba([WV_processed_sentence]))

    probas_ = np.array(probas_)
    if probas_.shape[0] == 0 :
        return "n.c."
    if probas_.shape[0] == 1 :
        probas_ = probas_[0][0]
        if np.argmax(probas_) == 0 :
            return [letter_date_to_submission_date(sentences_to_test[0][1]), "n.c."]
        elif np.argmax(probas_) == 1 :
            return ["n.c.", letter_date_to_submission_date(sentences_to_test[0][1])] 
        else :
            return ["n.c.", "n.c."]
    else :
        probas_ = np.squeeze(probas_)
        
        event = ["accident", "conso"]
        decisions = []
        for row in probas_ :
            if np.argmax(row) == 2 :
                decisions.append("other")
            elif max(row[:2])<= threshold :
                decisions.append("other")
            else :
                decisions.append(event[np.argmax(row[:2])])
        
        ind_accident = None
        ind_conso = None
        prev_prob_acc = 0
        prev_prob_conso = 0
        
        for i in range(probas_.shape[0]):
            prob_acc = probas_[i,0]
            if (decisions[i] == "accident") & (prob_acc >= prev_prob_acc) :
                ind_accident = i
                prev_prob_acc = prob_acc
                
            prob_conso = probas_[i,1]
            if (decisions[i] == "conso") & (prob_conso >= prev_prob_conso):
                ind_conso = i
                prev_prob_conso = prob_conso
                
        if ind_accident is None :
            date_accident = "n.c."
        else :
            date_accident = letter_date_to_submission_date(sentences_to_test[ind_accident][1])
            
        if ind_conso is None :
            date_conso = "n.c."
        else :
            date_conso = letter_date_to_submission_date(sentences_to_test[ind_conso][1])
        
        return [date_accident, date_conso]
    
def date_prediction_classifier_tfidf_multiclass(text, clf, vectorizer, threshold=0.35):
    
    probas_ = []
    sentences_to_test = extract_X_sentences_around_all_dates(text, 
                                                             terms_discarding_the_date=[])
    for tup in sentences_to_test :
        processed_sentence = tup[0]
        tfidf_encoded_processed_sentence = vectorizer.transform([processed_sentence])
        probas_.append(clf.predict_proba(tfidf_encoded_processed_sentence))

    probas_ = np.array(probas_)
    if probas_.shape[0] == 0 :
        return "n.c."
    if probas_.shape[0] == 1 :
        probas_ = probas_[0][0]
        if np.argmax(probas_) == 0 :
            return [letter_date_to_submission_date(sentences_to_test[0][1]), "n.c."]
        elif np.argmax(probas_) == 1 :
            return ["n.c.", letter_date_to_submission_date(sentences_to_test[0][1])] 
        else :
            return ["n.c.", "n.c."]
    else :
        probas_ = np.squeeze(probas_)
        
        event = ["accident", "conso"]
        decisions = []
        for row in probas_ :
            if np.argmax(row) == 2 :
                decisions.append("other")
            elif max(row[:2])<= threshold :
                decisions.append("other")
            else :
                decisions.append(event[np.argmax(row[:2])])
        
        ind_accident = None
        ind_conso = None
        prev_prob_acc = 0
        prev_prob_conso = 0
        
        for i in range(probas_.shape[0]):
            prob_acc = probas_[i,0]
            if (decisions[i] == "accident") & (prob_acc >= prev_prob_acc) :
                ind_accident = i
                prev_prob_acc = prob_acc
                
            prob_conso = probas_[i,1]
            if (decisions[i] == "conso") & (prob_conso >= prev_prob_conso):
                ind_conso = i
                prev_prob_conso = prob_conso
                
        if ind_accident is None :
            date_accident = "n.c."
        else :
            date_accident = letter_date_to_submission_date(sentences_to_test[ind_accident][1])
            
        if ind_conso is None :
            date_conso = "n.c."
        else :
            date_conso = letter_date_to_submission_date(sentences_to_test[ind_conso][1])
        
        return [date_accident, date_conso]