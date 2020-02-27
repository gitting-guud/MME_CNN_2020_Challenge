# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:59:40 2020

@author: Houcine's laptop
"""

import os
import re

import numpy as np
import pandas as pd

from fuzzywuzzy import process



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
        y_train.date_consolidation = y_train.date_consolidation.str.replace("n.a.", "n.c.")
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

def date_parser(date_string):
    """ Transforme une date de YYYY-mm-DD 
        en DD mm (en lettre) YYYY 
        pour la rechercher dans le texte
    """
    
    year_month_day = date_string.split('-')

    list_months=['janvier',"fevrier",
                 'mars','avril','mai','juin','juillet',
                 "aout",'septembre','octobre','novembre',
                 "decembre"]
    
    day = str(int(year_month_day[2])) # sans le 0 qui précède les unités
    month = list_months[int(year_month_day[1])-1]
    year = year_month_day[0]
    
    if day == "1":
        day +="er"
    return day + " " + month + " " + year


def letter_date_to_submission_date(string) :
    """Fonction qui transforme une date DD MM(en lettres) YYYY en une date DD-MM-YYYY pour comparaison avec date_target"""
    dico_months = {"janvier" : "01",
               'fevrier' : "02",
               "mars" : "03",
               "avril" : "04",
               "mai" : "05",
               "juin" : "06",
               "juillet" : "07",
               "aout" : "08",
                "septembre" : "09",
               "octobre": '10',
               "novembre" : "11",
               "decembre":'12'}
    
    strings = string.split()
    
    if len(strings[0]) == 1 :
        day = "0"+strings[0]
    elif strings[0] == "1er":
        day = "01"
    else :
        day = strings[0]
        
    month = process.extract(strings[1], dico_months.keys())[0][0]
    
    return strings[2]+"-"+dico_months[month]+"-"+day

def extract_X_sentences_before_after(text, date, X=2):
    """ Va chercher X phrases avant et apres la date passée en param dans le texte
    peut matcher avec plusieurs dates : renvoit toutes ces phrases sous forme de liste"""
    try :
        non_capitalized_date = date_parser(date)
    except :
        return "Date NC"
    
    L = np.array([non_capitalized_date in sen for sen in text]) + np.array([non_capitalized_date[:-5] in sen for sen in text])
    
    indexes = []
    
    if np.array(L).any() :
        indexes = np.where(L)[0]
    else :
        return "Pas trouvé de Date sous format DD mm (en lettre) YYYY dans le texte"
    
    result = []
    for ind in indexes :
        result.append(" ".join(text[ind-X : ind+X+1]))
    
    result = [e.replace(non_capitalized_date, "").replace(non_capitalized_date[:-5], '') for e in result]
    return result

terms_discarding_the_date = ["loi",
                             "jugement",
                             "audience",
                             "publique",
                             "tribubal",
                             "decision",
                             "greffe",
                             "ordonnance",
                             ]
def extract_X_sentences_around_all_dates(text, terms_discarding_the_date=terms_discarding_the_date, 
                                         use_date_forcing=False, date_to_look_for=None,
                                         X=1):
    """
    The text in structured format ( like in the column text of X_train)
    
    - if use_date_forcing=False : Looks for all the dates in the text
    - if use_date_forcing=True : Looks for date_to_look_for in the text (using date_parser)
    
    - for each date, extract X sentences  before and after this date (sentence meaning rows of the original doc)
    - if a row has multiple dates, separate each date in a row (the dates will have the same context) ==> PROBLEM
    
    
    - remove the contexts that contain the words in terms_discarding_the_date (to lower the number of contexts to score with the model later)
    - Return a list of tuples (context in lower + no stopwords + clean from punct + clan from date)
    - this context can be passed to Spacy avg vectorizer to get the avg Word embedding of the sentence or tfidf vectorizer
    """
    if use_date_forcing :
        assert date_to_look_for is not None
        try :
            date = date_parser(date_to_look_for)
        except :
            return "Date NC"
        
        l = [re.findall(date, STRING) if (re.findall(date, STRING)!=[]) else re.findall(date[:-5], STRING) for STRING in text ] 
        indexes = [i for i in range(len(l)) if len(l[i])!=0]
        ll = [(" ".join(text[i-X+1 : i+X]), l[i]) for i in indexes]
        
        lll = []
        for i in range(len(ll)) :
            if len([word for word in terms_discarding_the_date if word in ll[i][0]]) == 0 :
                if len(ll[i][1]) == 1 :
                    date = ll[i][1][0]
                    context_date_removed = ll[i][0].replace(date, '')
                    context_date_removed_cleaned = ""
                    for element in context_date_removed :
                        if not element.isalnum():
                            if element == " " :
                                context_date_removed_cleaned += element 
                        else :
                            context_date_removed_cleaned += element 
                    lll.append((context_date_removed_cleaned, date))
                else :
                    for j in range(len(ll[i][1])) :
                        date = ll[i][1][j]
                        context_date_removed = ll[i][0].replace(date, '')
                        context_date_removed_cleaned = ''
                        for element in context_date_removed :
                            if not element.isalnum():
                                if element == " " :
                                    context_date_removed_cleaned += element 
                            else :
                                context_date_removed_cleaned += element 
                        lll.append((context_date_removed_cleaned, date))
    else :
        l = [re.findall("\d{1,2} [a-zéû]{3,9} \d{4}", STRING) for STRING in text]
        indexes = [i for i in range(len(l)) if len(l[i])!=0]
        ll = [(" ".join(text[i-X+1 : i+X]), l[i]) for i in indexes]
        lll = []
        for i in range(len(ll)) :
            if len([word for word in terms_discarding_the_date if word in ll[i][0]]) == 0 :
                if len(ll[i][1]) == 1 :
                    date = ll[i][1][0]
                    context_date_removed = ll[i][0].replace(date, '')
                    if context_date_removed != "": 
                        lll.append((context_date_removed, date))
                else :
                    for j in range(len(ll[i][1])) :
                        date = ll[i][1][j]
                        context_date_removed = ll[i][0].replace(date, '')
                        if context_date_removed != "": 
                            lll.append((context_date_removed, date))
    return lll
    

terms_discarding_the_date = ["loi",
                             "jugement",
                             "audience",
                             "publique",
#                              "juge",
                             "tribubal",
                             "decision",
                             "greffe",
                             "conclusion",
                             "ordonnance",
                             ]
def extract_X_sentences_around_all_dates_other_dates(text, terms_discarding_the_date= terms_discarding_the_date, 
                                                     X=1):
    """
    The text in structured format ( like in the column text of X_train)
    - Looks for all the dates in the text
    - for each date, extract X sentences  before and after this date (sentence meaning rows of the original doc)
    - if a row has multiple dates, separate each date in a row (the dates will have the same context) ==> PROBLEM7
    - remove the contexts taht contain the words in terms_discarding_the_date (meaning that those are probably other dates) ==> Build a classifier 3 classes after
    - Return a list of tuples (context in lower + no stopwords + clean from punct , date)
    - this context can be passed to Spacy avg vectorizer to get the avg Word embedding of the sentence
    """

    l = [re.findall("\d{1,2} [a-zéû]{3,9} \d{4}", STRING) for STRING in text]
    indexes = [i for i in range(len(l)) if len(l[i])!=0]
    ll = [(" ".join(text[i-X+1 : i+X]), l[i]) for i in indexes]

    lll = []
    for i in range(len(ll)) :
        if len([word for word in terms_discarding_the_date if word in ll[i][0]]) > 0 :
            if len(ll[i][1]) == 1 :
                date = ll[i][1][0]
                context_date_removed = ll[i][0].replace(date, '')
                lll.append((context_date_removed, date))
    ind = np.random.choice(range(len(lll))) # je ne prends qu'un contexte aleatoirement pour commencer
    return lll[ind][0]
