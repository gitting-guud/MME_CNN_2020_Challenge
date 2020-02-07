# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:59:40 2020

@author: Houcine's laptop
"""
import numpy as np
import pandas as pd
import os
import re


PATH_TO_DATA = "./data/"
TRAINING_DIR = "train_folder"
TESTING_DIR = "test_folder"
TRAINING_TXT = "txt_files"
TESTING_TXT = "txt_files"
def read_train_test(sentence_per_row_mode=False):
    if sentence_per_row_mode :
        print("Reading training data ...")
        X_train = pd.read_csv(os.path.join(PATH_TO_DATA, TRAINING_DIR, "x_train_ids.csv"), index_col=0)
        tr_texts = []
        for tr_file in X_train.filename.values :
            f = open(os.path.join(PATH_TO_DATA, TRAINING_DIR, TRAINING_TXT, tr_file), "r", encoding="utf-8")
            text = f.readlines()
            text = [sen.replace('\xa0', '').replace('\n', '') for sen in text]
            tr_texts.append(text)
        X_train["text"] = tr_texts
        y_train = pd.read_csv(os.path.join(PATH_TO_DATA, "Y_train_predilex.csv"), index_col=0)
        print("Reading training data : Done")
        ###############################################
        print("Reading testing data ...")

        X_test = pd.read_csv(os.path.join(PATH_TO_DATA, TESTING_DIR, "x_test_ids.csv"), index_col=0)
        te_texts = []
        for te_file in X_test.filename.values :
            f = open(os.path.join(PATH_TO_DATA, TESTING_DIR, TESTING_TXT, te_file), "r", encoding="utf-8")
            text = f.readlines()
            text = [sen.replace('\xa0', '').replace('\n', '') for sen in text]
            te_texts.append(text)
        X_test["text"] = te_texts
        print("Reading testing data : Done")
        
    else:
        
        print("Reading training data ...")
        X_train = pd.read_csv(os.path.join(PATH_TO_DATA, TRAINING_DIR, "x_train_ids.csv"), index_col=0)
        tr_texts = []
        for tr_file in X_train.filename.values :
            f = open(os.path.join(PATH_TO_DATA, TRAINING_DIR, TRAINING_TXT, tr_file), "r", encoding="utf-8")
            tr_texts.append(f.read().replace('\xa0', ' ').replace('\n', ' '))
        X_train["text"] = tr_texts
        y_train = pd.read_csv(os.path.join(PATH_TO_DATA, "Y_train_predilex.csv"), index_col=0)
        print("Reading training data : Done")
        ###############################################
        print("Reading testing data ...")

        X_test = pd.read_csv(os.path.join(PATH_TO_DATA, TESTING_DIR, "x_test_ids.csv"), index_col=0)
        te_texts = []
        for te_file in X_test.filename.values :
            f = open(os.path.join(PATH_TO_DATA, TESTING_DIR, TESTING_TXT, te_file), "r", encoding="utf-8")
            te_texts.append(f.read().replace('\xa0', ' ').replace('\n', ' '))
        X_test["text"] = te_texts
        print("Reading testing data : Done")
        
    return X_train, y_train, X_test

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