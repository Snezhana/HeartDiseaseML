import pandas as pd
import numpy as np


def getAgeGroup(row):
    if row.age < 36:
        return 1
    elif row.age >= 36 and row.age< 56:
        return 2
    elif row.age >= 56:
        return 3
    

def get_resting_blood_pressure_categories(row):
    if row.resting_blood_pressure < 100:
        return 1
    elif row.resting_blood_pressure >= 100 and row.resting_blood_pressure< 120:
        return 2
    elif row.resting_blood_pressure >= 120 and row.resting_blood_pressure< 140:
        return 3
    elif row.resting_blood_pressure >= 140 and row.resting_blood_pressure< 160:
        return 4
    elif row.resting_blood_pressure >= 160:
        return 5
    
def get_serum_cholesterol_mg_per_dl_categ(row):
    if row.serum_cholesterol_mg_per_dl < 200:
        return 1
    elif row.serum_cholesterol_mg_per_dl >= 200 and row.serum_cholesterol_mg_per_dl< 240:
        return 2
    elif row.serum_cholesterol_mg_per_dl >= 240: 
        return 3


def get_oldpeak_eq_st_depression_categ(row):
    if row.oldpeak_eq_st_depression < 0.8:
        return 1
    elif row.oldpeak_eq_st_depression >= 0.8 and row.oldpeak_eq_st_depression< 1.7:
        return 2
    elif row.oldpeak_eq_st_depression >= 1.7 and row.oldpeak_eq_st_depression< 2.1:
        return 3
    elif row.oldpeak_eq_st_depression >= 2.1 and row.oldpeak_eq_st_depression< 2.5:
        return 4
    elif row.oldpeak_eq_st_depression >= 2.5:
        return 5
    
def get_max_heart_rate_achieved_categories(row):
    if row.max_heart_rate_achieved < 118:
        return 1
    elif row.max_heart_rate_achieved >= 118 and row.max_heart_rate_achieved< 133:
        return 2
    elif row.max_heart_rate_achieved >= 133 and row.max_heart_rate_achieved< 148:
        return 3
    elif row.max_heart_rate_achieved >= 148 and row.max_heart_rate_achieved< 167:
        return 4
    elif row.max_heart_rate_achieved >= 167:
        return 5

def get_disc_HR_adj_exerc_ind_ST_seg_categories(row):
    if row.HR_adj_exerc_ind_ST_seg_depr < 0.01:
        return 1
    elif row.HR_adj_exerc_ind_ST_seg_depr >= 0.01 and row.HR_adj_exerc_ind_ST_seg_depr< 0.02:
        return 2
    elif row.HR_adj_exerc_ind_ST_seg_depr >= 0.02: 
        return 3
                                                
def logLoss(df, true, pred):
    logloss = -(df[true] * np.log(df[pred]) + ((1-df[true]) * np.log(1-df[pred]))).sum()/len(df)
    return logloss
