# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:01:21 2020

@author: Houcine's laptop
"""

import re
import numpy as np

regex_year = "\d{4}"
regex_month_year = "[a-zéû]{3,9} \d{4}"

regex_standard_date = "\d{1,2} [a-zéû]{3,9} \d{4}"
# regex_standard_date = "\d{1,2} [(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)] \d{4}"
regex_slash_slash_slash = "\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}"  # Regex to match dates in DD/MM/YYYY format
regex_dot_dot_dot = "\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}"

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
dico_months_inv = {v:k for k, v in dico_months.items()}

def extract_years_row(row, dico_replacement) :
    list_year = re.findall(regex_year, row)
    for year in list_year :
        if 1900 <int(year)<2020 :
            dico_replacement[year] = year_approximation(year)
    return dico_replacement

def year_approximation(year) :
    return "1er janvier "+year

def extract_months_years_row(row, dico_replacement) :
    list_month_year = re.findall(regex_month_year, row)
    for month_year in list_month_year :
        if 1900<int(month_year.strip()[-4:])<2020 :
            if month_year.strip()[:-5] in ['janvier',"fevrier",
                 'mars','avril','mai','juin','juillet',
                 "aout",'septembre','octobre','novembre',
                 "decembre"]:
                dico_replacement[month_year] = month_year_approximation(month_year)
    return dico_replacement

def month_year_approximation(month_year) :
    return "1er "+ month_year

def extract_exact_date_row(row, dico_replacement) :
    list_exact_date = (re.findall(regex_standard_date, row) 
                       + re.findall(regex_slash_slash_slash, row) 
                       + re.findall(regex_dot_dot_dot, row))
    for exact_date in list_exact_date :
        if "/" in exact_date :
            dico_replacement[exact_date] = slash_date_parsing(exact_date)
        elif "." in exact_date :
            dico_replacement[exact_date] = dot_date_parsing(exact_date)
        else :
            tokens = exact_date.split()
            if int(tokens[0])==1 :
                tokens[0] = "1er"
            dico_replacement[exact_date] = " ".join(tokens)
    return dico_replacement

def dot_date_parsing(exact_date_dot) :
    strings = exact_date_dot.split(".")
    if int(strings[2]) < 100 :
        strings[2] = '19'+strings[2]
    if int(strings[0])==1 :
        strings[0] = "1er"
        return " ".join([strings[0], dico_months_inv[strings[1].strip()], strings[2]])
    else :
        return " ".join([str(int(strings[0])), dico_months_inv[strings[1].strip()], strings[2]])
    
def slash_date_parsing(exact_date_slash) :
    
    strings = exact_date_slash.split("/")
    if int(strings[2]) < 100 :
        strings[2] = '19'+strings[2]
    if int(strings[0])==1 :
        strings[0] = "1er"
        return " ".join([strings[0], dico_months_inv[strings[1].strip()], strings[2]])
    else :
        return " ".join([str(int(strings[0])), dico_months_inv[strings[1].strip()], strings[2]])

def build_final_dico_replacement(dico_replacement):
    new_dico_replacement = {}
    for i, e in enumerate(list(dico_replacement.keys())):
        flags = [e in element for element in list(dico_replacement.keys())[i+1:]]
        if not np.array(flags).any() :
            new_dico_replacement[e] = dico_replacement[e]
            
    return new_dico_replacement

def uniform_dates(row) :
    try :
        dico_replacement = extract_years_row(row, {})
        dico_replacement = extract_months_years_row(row, dico_replacement)    
        dico_replacement = extract_exact_date_row(row, dico_replacement)    
        new_dico_replacement = build_final_dico_replacement(dico_replacement)
        new_row = row
        for k, v in new_dico_replacement.items():
            new_row = re.sub(k, v, new_row)
        return new_row
    except:
        return row

def uniform_dates_text(text):
    return [uniform_dates(row) for row in text]