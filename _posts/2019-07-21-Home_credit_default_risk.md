---
title: "Kaggle Challenge: Home Credit Default Risk"
date: 2019-07-19
categories:
  - Data Science
tags: [Kaggle Competitions]
header:
  image: "/images/2019-07-21-Home_credit_default_risk/banner.png"
excerpt: "How capable each applicant is of repaying a loan? Various machine learning models will answer that question"
mathjax: "true"
---

Predict how capable each applicant is of repaying a loan.

Banner photo [Breno Assis](https://unsplash.com/@brenoassis)

## Context

This [challenge](https://www.kaggle.com/c/home-credit-default-risk) was proposed by __Home Credit Group__.

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Goal

Use historical loan application data to **predict whether or not an applicant will be able to repay a loan**. This is a standard **supervised classification task**.

Submissions are evaluated on **area under the ROC curve** between the predicted probability and the observed target.

## Type of ML
* Supervised: Labels are included in the training data
* Binary classification: target has only two values 0 (will repay loan on time), 1 (will have difficulty repaying loan)

## Guidelines 

* Download and load the data
* Sample the data in order to work on a smaller subset at first
* Explore the data, creating functions for cleaning it
* Split your data into features & labels ; training & testing
* Train different models and compare performance
* Train on your entire dataset, run predictions on your entire dataset and submit your results!
* Iterate

---

# Data

## First insight


```python
# usual data science stack in python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
```

    ['application_train.csv', 'home_credit.png', 'ROC-curve.png', 'HomeCredit_columns_description_perso.csv', 'breno-assis-517356-unsplash.jpg', 'application_test.csv', 'home-credit-default-risk.zip']
    


```python
# imports of need modules in sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
```


```python
import lightgbm as lgb
import xgboost as xgb
```


```python
# set options in this notebook
pd.set_option('display.max_columns', 300)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
path_train = os.path.join('..', 'input', 'application_train.csv')
path_test = os.path.join('..', 'input', 'application_test.csv')
```


```python
# load main datasets
app_train, app_test = pd.read_csv(path_train), pd.read_csv(path_test)
```


```python
# 1st insight
app_train.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>ORGANIZATION_TYPE</th>
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>APARTMENTS_AVG</th>
      <th>BASEMENTAREA_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BUILD_AVG</th>
      <th>COMMONAREA_AVG</th>
      <th>ELEVATORS_AVG</th>
      <th>ENTRANCES_AVG</th>
      <th>FLOORSMAX_AVG</th>
      <th>FLOORSMIN_AVG</th>
      <th>LANDAREA_AVG</th>
      <th>LIVINGAPARTMENTS_AVG</th>
      <th>LIVINGAREA_AVG</th>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <th>NONLIVINGAREA_AVG</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_MODE</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_MODE</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_MODE</th>
      <th>ENTRANCES_MODE</th>
      <th>FLOORSMAX_MODE</th>
      <th>FLOORSMIN_MODE</th>
      <th>LANDAREA_MODE</th>
      <th>LIVINGAPARTMENTS_MODE</th>
      <th>LIVINGAREA_MODE</th>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <th>NONLIVINGAREA_MODE</th>
      <th>APARTMENTS_MEDI</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>COMMONAREA_MEDI</th>
      <th>ELEVATORS_MEDI</th>
      <th>ENTRANCES_MEDI</th>
      <th>FLOORSMAX_MEDI</th>
      <th>FLOORSMIN_MEDI</th>
      <th>LANDAREA_MEDI</th>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>307506</th>
      <td>456251</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>254700.0</td>
      <td>27558.0</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Separated</td>
      <td>With parents</td>
      <td>0.032561</td>
      <td>-9327</td>
      <td>-236</td>
      <td>-8456.0</td>
      <td>-1982</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>THURSDAY</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Services</td>
      <td>0.145570</td>
      <td>0.681632</td>
      <td>NaN</td>
      <td>0.2021</td>
      <td>0.0887</td>
      <td>0.9876</td>
      <td>0.8300</td>
      <td>0.0202</td>
      <td>0.22</td>
      <td>0.1034</td>
      <td>0.6042</td>
      <td>0.2708</td>
      <td>0.0594</td>
      <td>0.1484</td>
      <td>0.1965</td>
      <td>0.0753</td>
      <td>0.1095</td>
      <td>0.1008</td>
      <td>0.0172</td>
      <td>0.9782</td>
      <td>0.7125</td>
      <td>0.0172</td>
      <td>0.0806</td>
      <td>0.0345</td>
      <td>0.4583</td>
      <td>0.0417</td>
      <td>0.0094</td>
      <td>0.0882</td>
      <td>0.0853</td>
      <td>0.0</td>
      <td>0.0125</td>
      <td>0.2040</td>
      <td>0.0887</td>
      <td>0.9876</td>
      <td>0.8323</td>
      <td>0.0203</td>
      <td>0.22</td>
      <td>0.1034</td>
      <td>0.6042</td>
      <td>0.2708</td>
      <td>0.0605</td>
      <td>0.1509</td>
      <td>0.2001</td>
      <td>0.0757</td>
      <td>0.1118</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.2898</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-273.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>456252</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>72000.0</td>
      <td>269550.0</td>
      <td>12001.5</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-20775</td>
      <td>365243</td>
      <td>-4388.0</td>
      <td>-4090</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.115992</td>
      <td>NaN</td>
      <td>0.0247</td>
      <td>0.0435</td>
      <td>0.9727</td>
      <td>0.6260</td>
      <td>0.0022</td>
      <td>0.00</td>
      <td>0.1034</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0579</td>
      <td>0.0202</td>
      <td>0.0257</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0252</td>
      <td>0.0451</td>
      <td>0.9727</td>
      <td>0.6406</td>
      <td>0.0022</td>
      <td>0.0000</td>
      <td>0.1034</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0592</td>
      <td>0.0220</td>
      <td>0.0267</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0250</td>
      <td>0.0435</td>
      <td>0.9727</td>
      <td>0.6310</td>
      <td>0.0022</td>
      <td>0.00</td>
      <td>0.1034</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0589</td>
      <td>0.0205</td>
      <td>0.0261</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0214</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>456253</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>153000.0</td>
      <td>677664.0</td>
      <td>29979.0</td>
      <td>585000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Separated</td>
      <td>House / apartment</td>
      <td>0.005002</td>
      <td>-14966</td>
      <td>-7921</td>
      <td>-6737.0</td>
      <td>-5150</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Managers</td>
      <td>1.0</td>
      <td>3</td>
      <td>3</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>School</td>
      <td>0.744026</td>
      <td>0.535722</td>
      <td>0.218859</td>
      <td>0.1031</td>
      <td>0.0862</td>
      <td>0.9816</td>
      <td>0.7484</td>
      <td>0.0123</td>
      <td>0.00</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>NaN</td>
      <td>0.0841</td>
      <td>0.9279</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1050</td>
      <td>0.0894</td>
      <td>0.9816</td>
      <td>0.7583</td>
      <td>0.0124</td>
      <td>0.0000</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>NaN</td>
      <td>0.0918</td>
      <td>0.9667</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.1041</td>
      <td>0.0862</td>
      <td>0.9816</td>
      <td>0.7518</td>
      <td>0.0124</td>
      <td>0.00</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>NaN</td>
      <td>0.0855</td>
      <td>0.9445</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.7970</td>
      <td>Panel</td>
      <td>No</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>-1909.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>456254</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>171000.0</td>
      <td>370107.0</td>
      <td>20205.0</td>
      <td>319500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.005313</td>
      <td>-11961</td>
      <td>-4786</td>
      <td>-2562.0</td>
      <td>-931</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Business Entity Type 1</td>
      <td>NaN</td>
      <td>0.514163</td>
      <td>0.661024</td>
      <td>0.0124</td>
      <td>NaN</td>
      <td>0.9771</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0690</td>
      <td>0.0417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0061</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0126</td>
      <td>NaN</td>
      <td>0.9772</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0690</td>
      <td>0.0417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0063</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0125</td>
      <td>NaN</td>
      <td>0.9771</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0690</td>
      <td>0.0417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0062</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>block of flats</td>
      <td>0.0086</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-322.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>456255</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>675000.0</td>
      <td>49117.5</td>
      <td>675000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.046220</td>
      <td>-16856</td>
      <td>-1262</td>
      <td>-5128.0</td>
      <td>-410</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>THURSDAY</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Business Entity Type 3</td>
      <td>0.734460</td>
      <td>0.708569</td>
      <td>0.113922</td>
      <td>0.0742</td>
      <td>0.0526</td>
      <td>0.9881</td>
      <td>NaN</td>
      <td>0.0176</td>
      <td>0.08</td>
      <td>0.0690</td>
      <td>0.3750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0791</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0756</td>
      <td>0.0546</td>
      <td>0.9881</td>
      <td>NaN</td>
      <td>0.0178</td>
      <td>0.0806</td>
      <td>0.0690</td>
      <td>0.3750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0824</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0749</td>
      <td>0.0526</td>
      <td>0.9881</td>
      <td>NaN</td>
      <td>0.0177</td>
      <td>0.08</td>
      <td>0.0690</td>
      <td>0.3750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0805</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>block of flats</td>
      <td>0.0718</td>
      <td>Panel</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-787.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
app_train.shape, app_test.shape
```




    ((307511, 122), (48744, 121))



Of course, the column named 'TARGET' is not in the test dataset.

## Content of each table and links between tables

* __application_{train|test}.csv__
    * This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
    * Static data for all applications. One row represents one loan in our data sample.
* __bureau.csv__
    * All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
    * For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
* __bureau_balance.csv__
    * Monthly balances of previous credits in Credit Bureau.
    * This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
* __POS_CASH_balance.csv__
    * Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
    * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
* __credit_card_balance.csv__
    * Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
    * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
* __previous_application.csv__
    * All previous applications for Home Credit loans of clients who have loans in our sample.
    * There is one row for each previous application related to loans in our data sample.
* __installments_payments.csv__
    * Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
    * There is a) one row for every payment that was made plus b) one row each for missed payment.
    * One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
* __HomeCredit_columns_description.csv__
    * This file contains descriptions for the columns in the various data files.

![png](/images/2019-07-21-Home_credit_default_risk/home_credit.png)

---

# Exploratory Data Analysis¶

## Distribution of the Target Column


```python
app_train.TARGET.value_counts()
```




    0    282686
    1     24825
    Name: TARGET, dtype: int64




```python
print(f'percentage of clients with payment difficulties: {app_train.TARGET.sum() / app_train.shape[0] * 100 :.2f}%')
```

    percentage of clients with payment difficulties: 8.07%
    


```python
plt.title('Distribution of the Target Column - 1 - client with payment difficulties / 0 - all other cases')
sns.countplot(x=app_train.TARGET, data=app_train)
plt.show()
```


![png](/images/2019-07-21-Home_credit_default_risk/output_27_0.png)


This is an [imbalanced class problem](http://www.chioka.in/class-imbalance-problem/). There are far more repaid loans than loans that were not repaid. It is important to weight the classes by their representation in the data to reflect this imbalance.

## Column Types


```python
app_train.dtypes.value_counts()
```




    float64    65
    int64      41
    object     16
    dtype: int64



int64 and float64 are numeric variables which can correspond to discrete or continuous features. 
Whereas object columns contain strings and are categorical features.


```python
# Number of unique classes in each object column
app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
```




    NAME_CONTRACT_TYPE             2
    CODE_GENDER                    3
    FLAG_OWN_CAR                   2
    FLAG_OWN_REALTY                2
    NAME_TYPE_SUITE                7
    NAME_INCOME_TYPE               8
    NAME_EDUCATION_TYPE            5
    NAME_FAMILY_STATUS             6
    NAME_HOUSING_TYPE              6
    OCCUPATION_TYPE               18
    WEEKDAY_APPR_PROCESS_START     7
    ORGANIZATION_TYPE             58
    FONDKAPREMONT_MODE             4
    HOUSETYPE_MODE                 3
    WALLSMATERIAL_MODE             7
    EMERGENCYSTATE_MODE            2
    dtype: int64



## Missing Values


```python
# Function to calculate missing values by column# Funct // credits Will Koehrsen
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```


```python
# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(10)
```

    Your selected dataframe has 122 columns.
    There are 67 columns that have missing values.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COMMONAREA_MEDI</th>
      <td>214865</td>
      <td>69.9</td>
    </tr>
    <tr>
      <th>COMMONAREA_AVG</th>
      <td>214865</td>
      <td>69.9</td>
    </tr>
    <tr>
      <th>COMMONAREA_MODE</th>
      <td>214865</td>
      <td>69.9</td>
    </tr>
    <tr>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <td>213514</td>
      <td>69.4</td>
    </tr>
    <tr>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <td>213514</td>
      <td>69.4</td>
    </tr>
    <tr>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <td>213514</td>
      <td>69.4</td>
    </tr>
    <tr>
      <th>FONDKAPREMONT_MODE</th>
      <td>210295</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>LIVINGAPARTMENTS_MODE</th>
      <td>210199</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <td>210199</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>LIVINGAPARTMENTS_AVG</th>
      <td>210199</td>
      <td>68.4</td>
    </tr>
  </tbody>
</table>
</div>



From here we have 2 options :
    * Use models such as XGBoost that can handle missing values
    * Or drop columns with a high percentage of missing values, and fill in other columns with a low percentage
It is not possible to know ahead of time if these columns will be helpful or not. My choice here is to drop them. Later if we need a more accurate score, we'll change the way to proceed.

### Dropping columns with a high ratio of missing values


```python
# cols_to_drop = list((app_train.isnull().sum() > 75000).index)
cols_to_drop = [c for c in app_train.columns if app_train[c].isnull().sum() > 75000]
```


```python
app_train, app_test = app_train.drop(cols_to_drop, axis=1), app_test.drop(cols_to_drop, axis=1)
app_test.isnull().sum().sort_values(ascending=False).head(10)
```




    EXT_SOURCE_3                  8668
    AMT_REQ_CREDIT_BUREAU_YEAR    6049
    AMT_REQ_CREDIT_BUREAU_MON     6049
    AMT_REQ_CREDIT_BUREAU_WEEK    6049
    AMT_REQ_CREDIT_BUREAU_DAY     6049
    AMT_REQ_CREDIT_BUREAU_HOUR    6049
    AMT_REQ_CREDIT_BUREAU_QRT     6049
    NAME_TYPE_SUITE                911
    DEF_60_CNT_SOCIAL_CIRCLE        29
    OBS_60_CNT_SOCIAL_CIRCLE        29
    dtype: int64



### Filling other missing values


```python
obj_cols = app_train.select_dtypes('object').columns
obj_cols
```




    Index(['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
           'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
           'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'ORGANIZATION_TYPE'],
          dtype='object')




```python
# filling string cols with 'Not specified' 
app_train[obj_cols] = app_train[obj_cols].fillna('Not specified')
app_test[obj_cols] = app_test[obj_cols].fillna('Not specified')
```


```python
float_cols = app_train.select_dtypes('float').columns
float_cols
```




    Index(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
           'REGION_POPULATION_RELATIVE', 'DAYS_REGISTRATION', 'CNT_FAM_MEMBERS',
           'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE',
           'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
           'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
           'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
           'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'],
          dtype='object')




```python
# filling float values with median of train (not test)
app_train[float_cols] = app_train[float_cols].fillna(app_train[float_cols].median())
app_test[float_cols] = app_test[float_cols].fillna(app_test[float_cols].median())
```


```python
app_train.shape, app_test.shape
```




    ((307511, 72), (48744, 71))



Let's check if there is still NaNs


```python
app_train.isnull().sum().sort_values(ascending=False).head()
```




    AMT_REQ_CREDIT_BUREAU_YEAR    0
    AMT_REQ_CREDIT_BUREAU_QRT     0
    DAYS_REGISTRATION             0
    DAYS_ID_PUBLISH               0
    FLAG_MOBIL                    0
    dtype: int64




```python
app_test.isnull().sum().sort_values(ascending=False).head()
```




    AMT_REQ_CREDIT_BUREAU_YEAR    0
    FLAG_EMAIL                    0
    DAYS_ID_PUBLISH               0
    FLAG_MOBIL                    0
    FLAG_EMP_PHONE                0
    dtype: int64




```python
# Is there any duplicated rows ?
```


```python
app_train.duplicated().sum()
```




    0




```python
app_test.duplicated().sum()
```




    0



## Categorical columns (type object)


```python
# Number of unique classes in each object column
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
```




    NAME_CONTRACT_TYPE             2
    CODE_GENDER                    3
    FLAG_OWN_CAR                   2
    FLAG_OWN_REALTY                2
    NAME_TYPE_SUITE                8
    NAME_INCOME_TYPE               8
    NAME_EDUCATION_TYPE            5
    NAME_FAMILY_STATUS             6
    NAME_HOUSING_TYPE              6
    WEEKDAY_APPR_PROCESS_START     7
    ORGANIZATION_TYPE             58
    dtype: int64



## Dealing with anomalies


```python
app_train['DAYS_EMPLOYED'].describe()
```




    count    307511.000000
    mean      63815.045904
    std      141275.766519
    min      -17912.000000
    25%       -2760.000000
    50%       -1213.000000
    75%        -289.000000
    max      365243.000000
    Name: DAYS_EMPLOYED, dtype: float64



The maximum value is abnormal (besides being positive). It corresponds to 1000 years...


```python
sns.distplot(app_train['DAYS_EMPLOYED'], kde=False);
plt.show()
```


![png](/images/2019-07-21-Home_credit_default_risk/output_57_0.png)



```python
print('The non-anomalies default on %0.2f%% of loans' % (100 * app_train[app_train['DAYS_EMPLOYED'] != 365243]['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * app_train[app_train['DAYS_EMPLOYED'] == 365243]['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(app_train[app_train['DAYS_EMPLOYED'] == 365243]))
```

    The non-anomalies default on 8.66% of loans
    The anomalies default on 5.40% of loans
    There are 55374 anomalous days of employment
    

It turns out that the anomalies have a lower rate of default.

The anomalous values seem to have some importance. Let's fill in the anomalous values with not a np.nan and then create a new boolean column indicating whether or not the value was anomalous.


```python
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

sns.distplot(app_train['DAYS_EMPLOYED'].dropna(), kde=False);
```


![png](/images/2019-07-21-Home_credit_default_risk/output_60_0.png)



```python
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
```

    There are 9274 anomalies in the test data out of 48744 entries
    


```python
# refilling float values with median of train (not test)

app_train[float_cols] = app_train[float_cols].apply(pd.to_numeric, errors='coerce')
app_train = app_train.fillna(app_train.median())

app_test[float_cols] = app_test[float_cols].apply(pd.to_numeric, errors='coerce')
app_test = app_train.fillna(app_test.median())
```

## Correlations

The correlation coefficient is not the best method to represent "relevance" of a feature, but it gives us an idea of possible relationships within the data. Some general interpretations of the absolute value of the correlation coefficent are:

* 00-.19 “very weak”
* 20-.39 “weak”
* 40-.59 “moderate”
* 60-.79 “strong”
* 80-1.0 “very strong”


```python
correlations = app_train.corr()['TARGET'].sort_values()

print('Most Positive Correlations:\n', correlations.tail(10))
print('\n\nMost Negative Correlations:\n', correlations.head(10))
```

    Most Positive Correlations:
     REG_CITY_NOT_LIVE_CITY         0.044395
    FLAG_EMP_PHONE                 0.045982
    REG_CITY_NOT_WORK_CITY         0.050994
    DAYS_ID_PUBLISH                0.051457
    DAYS_LAST_PHONE_CHANGE         0.055218
    REGION_RATING_CLIENT           0.058899
    REGION_RATING_CLIENT_W_CITY    0.060893
    DAYS_EMPLOYED                  0.063368
    DAYS_BIRTH                     0.078239
    TARGET                         1.000000
    Name: TARGET, dtype: float64
    
    
    Most Negative Correlations:
     EXT_SOURCE_2                 -0.160295
    EXT_SOURCE_3                 -0.155892
    DAYS_EMPLOYED_ANOM           -0.045987
    AMT_GOODS_PRICE              -0.039623
    REGION_POPULATION_RELATIVE   -0.037227
    AMT_CREDIT                   -0.030369
    FLAG_DOCUMENT_6              -0.028602
    HOUR_APPR_PROCESS_START      -0.024166
    FLAG_PHONE                   -0.023806
    AMT_REQ_CREDIT_BUREAU_MON    -0.014794
    Name: TARGET, dtype: float64
    


```python
# Compute the correlation matrix
corr = app_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(21, 19))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7b866f50f0>




![png](/images/2019-07-21-Home_credit_default_risk/output_66_1.png)


### Effect of Age on Repayment


```python
# Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
```




    -0.07823930830982712



There isn't any correlation between age and repayment


```python
plt.figure(figsize = (12, 6))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
```


![png](/images/2019-07-21-Home_credit_default_risk/output_70_0.png)



```python
# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)
```

    /home/sunflowa/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /home/sunflowa/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TARGET</th>
      <th>DAYS_BIRTH</th>
      <th>YEARS_BIRTH</th>
      <th>YEARS_BINNED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9461</td>
      <td>25.920548</td>
      <td>(25.0, 30.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>16765</td>
      <td>45.931507</td>
      <td>(45.0, 50.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>19046</td>
      <td>52.180822</td>
      <td>(50.0, 55.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>19005</td>
      <td>52.068493</td>
      <td>(50.0, 55.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>19932</td>
      <td>54.608219</td>
      <td>(50.0, 55.0]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>16941</td>
      <td>46.413699</td>
      <td>(45.0, 50.0]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>13778</td>
      <td>37.747945</td>
      <td>(35.0, 40.0]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>18850</td>
      <td>51.643836</td>
      <td>(50.0, 55.0]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>20099</td>
      <td>55.065753</td>
      <td>(55.0, 60.0]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>14469</td>
      <td>39.641096</td>
      <td>(35.0, 40.0]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TARGET</th>
      <th>DAYS_BIRTH</th>
      <th>YEARS_BIRTH</th>
    </tr>
    <tr>
      <th>YEARS_BINNED</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(20.0, 25.0]</th>
      <td>0.123036</td>
      <td>8532.795625</td>
      <td>23.377522</td>
    </tr>
    <tr>
      <th>(25.0, 30.0]</th>
      <td>0.111436</td>
      <td>10155.219250</td>
      <td>27.822518</td>
    </tr>
    <tr>
      <th>(30.0, 35.0]</th>
      <td>0.102814</td>
      <td>11854.848377</td>
      <td>32.479037</td>
    </tr>
    <tr>
      <th>(35.0, 40.0]</th>
      <td>0.089414</td>
      <td>13707.908253</td>
      <td>37.555913</td>
    </tr>
    <tr>
      <th>(40.0, 45.0]</th>
      <td>0.078491</td>
      <td>15497.661233</td>
      <td>42.459346</td>
    </tr>
    <tr>
      <th>(45.0, 50.0]</th>
      <td>0.074171</td>
      <td>17323.900441</td>
      <td>47.462741</td>
    </tr>
    <tr>
      <th>(50.0, 55.0]</th>
      <td>0.066968</td>
      <td>19196.494791</td>
      <td>52.593136</td>
    </tr>
    <tr>
      <th>(55.0, 60.0]</th>
      <td>0.055314</td>
      <td>20984.262742</td>
      <td>57.491131</td>
    </tr>
    <tr>
      <th>(60.0, 65.0]</th>
      <td>0.052737</td>
      <td>22780.547460</td>
      <td>62.412459</td>
    </tr>
    <tr>
      <th>(65.0, 70.0]</th>
      <td>0.037270</td>
      <td>24292.614340</td>
      <td>66.555108</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (8, 6))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
```


![png](/images/2019-07-21-Home_credit_default_risk/output_73_0.png)


Younger applicants are more likely to not repay the loan.

# Preparing data

## Encoding Categorical Variables

A ML model can't deal with categorical features (except for some models such as LightGBM). 
One have to find a way to encode (represent) these variables as numbers. There are two main ways :

* Label encoding: assign each unique category in a categorical variable with an integer. No new columns are created. The problem with label encoding is that it gives the categories an arbitrary ordering.
* One-hot encoding: create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.  


```python
app_train = pd.get_dummies(data=app_train, columns=obj_cols)
app_test = pd.get_dummies(data=app_test, columns=obj_cols)
```

## Aligning Training and Testing Data

Both the training and testing data should have the same features (columns). One-hot encoding can more columns in the one dataset because there were some categorical variables with categories not represented in the other dataset. In order to remove the columns in the training data that are not in the testing data, one need to align the dataframes.


```python
# back up of the target /  need to keep this information
y = app_train.TARGET
app_train = app_train.drop(columns=['TARGET'])
```


```python
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
```


```python
app_train.shape, app_test.shape
```




    ((307511, 168), (307511, 168))



## Scaling values


```python
feat_to_scale = list(float_cols).copy()
feat_to_scale.extend(['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'HOUR_APPR_PROCESS_START'])
feat_to_scale
```




    ['AMT_INCOME_TOTAL',
     'AMT_CREDIT',
     'AMT_ANNUITY',
     'AMT_GOODS_PRICE',
     'REGION_POPULATION_RELATIVE',
     'DAYS_REGISTRATION',
     'CNT_FAM_MEMBERS',
     'EXT_SOURCE_2',
     'EXT_SOURCE_3',
     'OBS_30_CNT_SOCIAL_CIRCLE',
     'DEF_30_CNT_SOCIAL_CIRCLE',
     'OBS_60_CNT_SOCIAL_CIRCLE',
     'DEF_60_CNT_SOCIAL_CIRCLE',
     'DAYS_LAST_PHONE_CHANGE',
     'AMT_REQ_CREDIT_BUREAU_HOUR',
     'AMT_REQ_CREDIT_BUREAU_DAY',
     'AMT_REQ_CREDIT_BUREAU_WEEK',
     'AMT_REQ_CREDIT_BUREAU_MON',
     'AMT_REQ_CREDIT_BUREAU_QRT',
     'AMT_REQ_CREDIT_BUREAU_YEAR',
     'CNT_CHILDREN',
     'DAYS_BIRTH',
     'DAYS_EMPLOYED',
     'DAYS_ID_PUBLISH',
     'HOUR_APPR_PROCESS_START']




```python
scaler = StandardScaler()
app_train[feat_to_scale] = scaler.fit_transform(app_train[feat_to_scale])
app_test[feat_to_scale] = scaler.fit_transform(app_test[feat_to_scale])
app_train.head()
```

    /home/sunflowa/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /home/sunflowa/anaconda3/lib/python3.7/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    /home/sunflowa/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /home/sunflowa/anaconda3/lib/python3.7/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
      <th>DAYS_EMPLOYED_ANOM</th>
      <th>NAME_CONTRACT_TYPE_Cash loans</th>
      <th>NAME_CONTRACT_TYPE_Revolving loans</th>
      <th>CODE_GENDER_F</th>
      <th>CODE_GENDER_M</th>
      <th>CODE_GENDER_XNA</th>
      <th>FLAG_OWN_CAR_N</th>
      <th>FLAG_OWN_CAR_Y</th>
      <th>FLAG_OWN_REALTY_N</th>
      <th>FLAG_OWN_REALTY_Y</th>
      <th>NAME_TYPE_SUITE_Children</th>
      <th>NAME_TYPE_SUITE_Family</th>
      <th>NAME_TYPE_SUITE_Group of people</th>
      <th>NAME_TYPE_SUITE_Not specified</th>
      <th>NAME_TYPE_SUITE_Other_A</th>
      <th>NAME_TYPE_SUITE_Other_B</th>
      <th>NAME_TYPE_SUITE_Spouse, partner</th>
      <th>NAME_TYPE_SUITE_Unaccompanied</th>
      <th>NAME_INCOME_TYPE_Businessman</th>
      <th>NAME_INCOME_TYPE_Commercial associate</th>
      <th>NAME_INCOME_TYPE_Maternity leave</th>
      <th>NAME_INCOME_TYPE_Pensioner</th>
      <th>NAME_INCOME_TYPE_State servant</th>
      <th>NAME_INCOME_TYPE_Student</th>
      <th>NAME_INCOME_TYPE_Unemployed</th>
      <th>NAME_INCOME_TYPE_Working</th>
      <th>NAME_EDUCATION_TYPE_Academic degree</th>
      <th>NAME_EDUCATION_TYPE_Higher education</th>
      <th>NAME_EDUCATION_TYPE_Incomplete higher</th>
      <th>NAME_EDUCATION_TYPE_Lower secondary</th>
      <th>NAME_EDUCATION_TYPE_Secondary / secondary special</th>
      <th>NAME_FAMILY_STATUS_Civil marriage</th>
      <th>NAME_FAMILY_STATUS_Married</th>
      <th>NAME_FAMILY_STATUS_Separated</th>
      <th>NAME_FAMILY_STATUS_Single / not married</th>
      <th>NAME_FAMILY_STATUS_Unknown</th>
      <th>NAME_FAMILY_STATUS_Widow</th>
      <th>NAME_HOUSING_TYPE_Co-op apartment</th>
      <th>NAME_HOUSING_TYPE_House / apartment</th>
      <th>NAME_HOUSING_TYPE_Municipal apartment</th>
      <th>NAME_HOUSING_TYPE_Office apartment</th>
      <th>NAME_HOUSING_TYPE_Rented apartment</th>
      <th>NAME_HOUSING_TYPE_With parents</th>
      <th>WEEKDAY_APPR_PROCESS_START_FRIDAY</th>
      <th>WEEKDAY_APPR_PROCESS_START_MONDAY</th>
      <th>WEEKDAY_APPR_PROCESS_START_SATURDAY</th>
      <th>WEEKDAY_APPR_PROCESS_START_SUNDAY</th>
      <th>WEEKDAY_APPR_PROCESS_START_THURSDAY</th>
      <th>WEEKDAY_APPR_PROCESS_START_TUESDAY</th>
      <th>WEEKDAY_APPR_PROCESS_START_WEDNESDAY</th>
      <th>ORGANIZATION_TYPE_Advertising</th>
      <th>ORGANIZATION_TYPE_Agriculture</th>
      <th>ORGANIZATION_TYPE_Bank</th>
      <th>ORGANIZATION_TYPE_Business Entity Type 1</th>
      <th>ORGANIZATION_TYPE_Business Entity Type 2</th>
      <th>ORGANIZATION_TYPE_Business Entity Type 3</th>
      <th>ORGANIZATION_TYPE_Cleaning</th>
      <th>ORGANIZATION_TYPE_Construction</th>
      <th>ORGANIZATION_TYPE_Culture</th>
      <th>ORGANIZATION_TYPE_Electricity</th>
      <th>ORGANIZATION_TYPE_Emergency</th>
      <th>ORGANIZATION_TYPE_Government</th>
      <th>ORGANIZATION_TYPE_Hotel</th>
      <th>ORGANIZATION_TYPE_Housing</th>
      <th>ORGANIZATION_TYPE_Industry: type 1</th>
      <th>ORGANIZATION_TYPE_Industry: type 10</th>
      <th>ORGANIZATION_TYPE_Industry: type 11</th>
      <th>ORGANIZATION_TYPE_Industry: type 12</th>
      <th>ORGANIZATION_TYPE_Industry: type 13</th>
      <th>ORGANIZATION_TYPE_Industry: type 2</th>
      <th>ORGANIZATION_TYPE_Industry: type 3</th>
      <th>ORGANIZATION_TYPE_Industry: type 4</th>
      <th>ORGANIZATION_TYPE_Industry: type 5</th>
      <th>ORGANIZATION_TYPE_Industry: type 6</th>
      <th>ORGANIZATION_TYPE_Industry: type 7</th>
      <th>ORGANIZATION_TYPE_Industry: type 8</th>
      <th>ORGANIZATION_TYPE_Industry: type 9</th>
      <th>ORGANIZATION_TYPE_Insurance</th>
      <th>ORGANIZATION_TYPE_Kindergarten</th>
      <th>ORGANIZATION_TYPE_Legal Services</th>
      <th>ORGANIZATION_TYPE_Medicine</th>
      <th>ORGANIZATION_TYPE_Military</th>
      <th>ORGANIZATION_TYPE_Mobile</th>
      <th>ORGANIZATION_TYPE_Other</th>
      <th>ORGANIZATION_TYPE_Police</th>
      <th>ORGANIZATION_TYPE_Postal</th>
      <th>ORGANIZATION_TYPE_Realtor</th>
      <th>ORGANIZATION_TYPE_Religion</th>
      <th>ORGANIZATION_TYPE_Restaurant</th>
      <th>ORGANIZATION_TYPE_School</th>
      <th>ORGANIZATION_TYPE_Security</th>
      <th>ORGANIZATION_TYPE_Security Ministries</th>
      <th>ORGANIZATION_TYPE_Self-employed</th>
      <th>ORGANIZATION_TYPE_Services</th>
      <th>ORGANIZATION_TYPE_Telecom</th>
      <th>ORGANIZATION_TYPE_Trade: type 1</th>
      <th>ORGANIZATION_TYPE_Trade: type 2</th>
      <th>ORGANIZATION_TYPE_Trade: type 3</th>
      <th>ORGANIZATION_TYPE_Trade: type 4</th>
      <th>ORGANIZATION_TYPE_Trade: type 5</th>
      <th>ORGANIZATION_TYPE_Trade: type 6</th>
      <th>ORGANIZATION_TYPE_Trade: type 7</th>
      <th>ORGANIZATION_TYPE_Transport: type 1</th>
      <th>ORGANIZATION_TYPE_Transport: type 2</th>
      <th>ORGANIZATION_TYPE_Transport: type 3</th>
      <th>ORGANIZATION_TYPE_Transport: type 4</th>
      <th>ORGANIZATION_TYPE_University</th>
      <th>ORGANIZATION_TYPE_XNA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>-0.577538</td>
      <td>0.142129</td>
      <td>-0.478095</td>
      <td>-0.166143</td>
      <td>-0.507236</td>
      <td>-0.149452</td>
      <td>-1.506880</td>
      <td>0.755835</td>
      <td>0.379837</td>
      <td>0.579154</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1.265722</td>
      <td>2</td>
      <td>2</td>
      <td>-0.631821</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.317940</td>
      <td>-2.153651</td>
      <td>0.242861</td>
      <td>4.163504</td>
      <td>0.252132</td>
      <td>5.253260</td>
      <td>-0.206992</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.070987</td>
      <td>-0.058766</td>
      <td>-0.155837</td>
      <td>-0.269947</td>
      <td>-0.30862</td>
      <td>-0.440926</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>-0.577538</td>
      <td>0.426792</td>
      <td>1.725450</td>
      <td>0.592683</td>
      <td>1.600873</td>
      <td>-1.252750</td>
      <td>0.166821</td>
      <td>0.497899</td>
      <td>1.078697</td>
      <td>1.790855</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-0.167638</td>
      <td>1</td>
      <td>1</td>
      <td>-0.325620</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.564482</td>
      <td>0.112063</td>
      <td>-0.174085</td>
      <td>-0.320480</td>
      <td>-0.168527</td>
      <td>-0.275663</td>
      <td>0.163107</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.070987</td>
      <td>-0.058766</td>
      <td>-0.155837</td>
      <td>-0.269947</td>
      <td>-0.30862</td>
      <td>-1.007331</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>-0.577538</td>
      <td>-0.427196</td>
      <td>-1.152888</td>
      <td>-1.404669</td>
      <td>-1.092145</td>
      <td>-0.783451</td>
      <td>0.689509</td>
      <td>0.948701</td>
      <td>0.206116</td>
      <td>0.306869</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1.265722</td>
      <td>2</td>
      <td>2</td>
      <td>-0.938022</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.216948</td>
      <td>1.223975</td>
      <td>-0.591031</td>
      <td>-0.320480</td>
      <td>-0.589187</td>
      <td>-0.275663</td>
      <td>0.178831</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.070987</td>
      <td>-0.058766</td>
      <td>-0.155837</td>
      <td>-0.269947</td>
      <td>-0.30862</td>
      <td>-1.007331</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>-0.577538</td>
      <td>-0.142533</td>
      <td>-0.711430</td>
      <td>0.177874</td>
      <td>-0.653463</td>
      <td>-0.928991</td>
      <td>0.680114</td>
      <td>-0.368597</td>
      <td>-1.375829</td>
      <td>0.369143</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.167638</td>
      <td>2</td>
      <td>2</td>
      <td>1.511587</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.712205</td>
      <td>0.112063</td>
      <td>0.242861</td>
      <td>-0.320480</td>
      <td>0.252132</td>
      <td>-0.275663</td>
      <td>0.418306</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.070987</td>
      <td>-0.058766</td>
      <td>-0.155837</td>
      <td>-0.269947</td>
      <td>-0.30862</td>
      <td>-0.440926</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>-0.577538</td>
      <td>-0.199466</td>
      <td>-0.213734</td>
      <td>-0.361749</td>
      <td>-0.068554</td>
      <td>0.563570</td>
      <td>0.892535</td>
      <td>-0.368129</td>
      <td>0.191639</td>
      <td>-0.307263</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1.265722</td>
      <td>2</td>
      <td>2</td>
      <td>-0.325620</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1.004691</td>
      <td>0.112063</td>
      <td>-0.591031</td>
      <td>-0.320480</td>
      <td>-0.589187</td>
      <td>-0.275663</td>
      <td>-0.173126</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.070987</td>
      <td>-0.058766</td>
      <td>-0.155837</td>
      <td>-0.269947</td>
      <td>-0.30862</td>
      <td>-1.007331</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Splitting training / test datasets 
from app_train in order to make few predictions before submission & select models


```python
X_train, X_test, y_train, y_test = train_test_split(app_train, y, test_size=0.2)
```

---

# Base line

## Metric: ROC AUC

more infos on [the Receiver Operating Characteristic Area Under the Curve (ROC AUC, also sometimes called AUROC)](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it).

The [Reciever Operating Characteristic (ROC) curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) graphs the true positive rate versus the false positive rate:

![png](/images/2019-07-21-Home_credit_default_risk/ROC-curve.png)

A single line on the graph indicates the curve for a single model, and movement along a line indicates changing the threshold used for classifying a positive instance. The threshold starts at 0 in the upper right to and goes to 1 in the lower left. A curve that is to the left and above another curve indicates a better model. For example, the blue model is better than the red one (which is better than the black diagonal line which indicates a naive random guessing model).

The Area Under the Curve (AUC) is the integral of the curve. This metric is between 0 and 1 with a better model scoring higher. A model that simply guesses at random will have an ROC AUC of 0.5.

When we measure a classifier according to the ROC AUC, we do not generate 0 or 1 predictions, but rather a probability between 0 and 1. 

When we get into problems with inbalanced classes, accuracy is not the best metric. A model with a high ROC AUC will also have a high accuracy, but the ROC AUC is a better representation of model performance.

## Random forrest


```python
# a simple RandomForrest Classifier without CV
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
roc_auc_score(y_test, y_pred)
```




    0.5011674887648608



The predictions must be in the format shown in the sample_submission.csv file, where there are only two columns: SK_ID_CURR and TARGET. Let's create a dataframe in this format from the test set and the predictions called submit.


```python
def submit(model, csv_name):
    
    # fit on the whole dataset of train
    model.fit(app_train, y)
    
    # Make predictions & make sure to select the second column only
    result = model.predict_proba(app_test)[:, 1]

    submit = app_test[['SK_ID_CURR']]
    submit['TARGET'] = result

    # Save the submission to a csv file
    submit.to_csv(csv_name, index = False)
```


```python
# submit(rf, 'random_forrest_clf.csv')
```

The random forrest model should score around 0.58329 when submitted which is not really good, because just above 0.5 i.e a random classifier...

## Feature Importances


```python
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(app_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
```

    Feature ranking:
    1. feature 27 (0.064841)
    2. feature 28 (0.059712)
    3. feature 7 (0.046094)
    4. feature 10 (0.045429)
    5. feature 9 (0.045176)
    6. feature 0 (0.044173)
    7. feature 4 (0.041803)
    8. feature 8 (0.041481)
    9. feature 33 (0.041030)
    10. feature 3 (0.039948)
    11. feature 2 (0.035454)
    12. feature 6 (0.035247)
    13. feature 5 (0.033751)
    14. feature 20 (0.031268)
    15. feature 59 (0.022036)
    16. feature 31 (0.017715)
    17. feature 29 (0.017527)
    18. feature 17 (0.013756)
    19. feature 1 (0.010035)
    20. feature 58 (0.008328)
    21. feature 57 (0.007471)
    22. feature 18 (0.006926)
    23. feature 69 (0.006884)
    24. feature 19 (0.006881)
    25. feature 15 (0.006740)
    26. feature 92 (0.006712)
    27. feature 68 (0.006608)
    28. feature 30 (0.006549)
    29. feature 115 (0.006527)
    30. feature 108 (0.006369)
    31. feature 13 (0.006358)
    32. feature 103 (0.006115)
    33. feature 107 (0.006110)
    34. feature 109 (0.006067)
    35. feature 104 (0.005811)
    36. feature 152 (0.005601)
    37. feature 77 (0.005453)
    38. feature 25 (0.005225)
    39. feature 35 (0.005202)
    40. feature 32 (0.005143)
    41. feature 85 (0.005102)
    42. feature 66 (0.004996)
    43. feature 67 (0.004896)
    44. feature 94 (0.004894)
    45. feature 105 (0.004844)
    46. feature 26 (0.004821)
    47. feature 71 (0.004800)
    48. feature 91 (0.004709)
    49. feature 90 (0.004508)
    50. feature 79 (0.004343)
    51. feature 98 (0.004327)
    52. feature 24 (0.004303)
    53. feature 64 (0.004122)
    54. feature 63 (0.004009)
    55. feature 87 (0.003993)
    56. feature 93 (0.003624)
    57. feature 106 (0.003384)
    58. feature 16 (0.003367)
    59. feature 143 (0.003299)
    60. feature 102 (0.003096)
    61. feature 40 (0.002746)
    62. feature 114 (0.002712)
    63. feature 117 (0.002707)
    64. feature 76 (0.002608)
    65. feature 161 (0.002586)
    66. feature 56 (0.002539)
    67. feature 22 (0.002522)
    68. feature 99 (0.002518)
    69. feature 121 (0.002230)
    70. feature 96 (0.002213)
    71. feature 82 (0.002195)
    72. feature 140 (0.002136)
    73. feature 23 (0.002091)
    74. feature 88 (0.002060)
    75. feature 101 (0.001985)
    76. feature 61 (0.001886)
    77. feature 113 (0.001883)
    78. feature 165 (0.001881)
    79. feature 38 (0.001878)
    80. feature 62 (0.001859)
    81. feature 149 (0.001758)
    82. feature 130 (0.001690)
    83. feature 138 (0.001645)
    84. feature 89 (0.001614)
    85. feature 157 (0.001587)
    86. feature 37 (0.001502)
    87. feature 150 (0.001492)
    88. feature 111 (0.001400)
    89. feature 123 (0.001298)
    90. feature 126 (0.001292)
    91. feature 21 (0.001287)
    92. feature 70 (0.001248)
    93. feature 164 (0.001238)
    94. feature 148 (0.001203)
    95. feature 75 (0.001104)
    96. feature 145 (0.001089)
    97. feature 167 (0.001063)
    98. feature 136 (0.001062)
    99. feature 55 (0.001006)
    100. feature 81 (0.001001)
    101. feature 163 (0.000992)
    102. feature 54 (0.000963)
    103. feature 12 (0.000949)
    104. feature 60 (0.000938)
    105. feature 100 (0.000927)
    106. feature 124 (0.000895)
    107. feature 134 (0.000826)
    108. feature 131 (0.000789)
    109. feature 153 (0.000724)
    110. feature 48 (0.000719)
    111. feature 141 (0.000694)
    112. feature 112 (0.000693)
    113. feature 50 (0.000687)
    114. feature 144 (0.000681)
    115. feature 156 (0.000631)
    116. feature 74 (0.000601)
    117. feature 97 (0.000592)
    118. feature 151 (0.000589)
    119. feature 73 (0.000482)
    120. feature 166 (0.000461)
    121. feature 119 (0.000460)
    122. feature 122 (0.000450)
    123. feature 146 (0.000447)
    124. feature 41 (0.000426)
    125. feature 129 (0.000396)
    126. feature 154 (0.000395)
    127. feature 14 (0.000376)
    128. feature 155 (0.000343)
    129. feature 132 (0.000341)
    130. feature 110 (0.000311)
    131. feature 43 (0.000310)
    132. feature 116 (0.000301)
    133. feature 120 (0.000282)
    134. feature 142 (0.000280)
    135. feature 137 (0.000279)
    136. feature 72 (0.000267)
    137. feature 139 (0.000261)
    138. feature 160 (0.000257)
    139. feature 46 (0.000214)
    140. feature 118 (0.000208)
    141. feature 45 (0.000186)
    142. feature 127 (0.000184)
    143. feature 52 (0.000161)
    144. feature 51 (0.000144)
    145. feature 162 (0.000106)
    146. feature 47 (0.000101)
    147. feature 133 (0.000101)
    148. feature 84 (0.000093)
    149. feature 53 (0.000089)
    150. feature 147 (0.000088)
    151. feature 128 (0.000085)
    152. feature 125 (0.000069)
    153. feature 159 (0.000063)
    154. feature 34 (0.000057)
    155. feature 135 (0.000053)
    156. feature 49 (0.000051)
    157. feature 86 (0.000044)
    158. feature 158 (0.000037)
    159. feature 39 (0.000020)
    160. feature 80 (0.000020)
    161. feature 83 (0.000001)
    162. feature 65 (0.000001)
    163. feature 11 (0.000000)
    164. feature 42 (0.000000)
    165. feature 36 (0.000000)
    166. feature 44 (0.000000)
    167. feature 95 (0.000000)
    168. feature 78 (0.000000)
    


```python
# Plot the feature importances of the rf
plt.figure(figsize=(16, 8))
plt.title("Feature importances")
plt.bar(range(app_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(app_train.shape[1]), indices)
plt.xlim([-1, app_train.shape[1]])
plt.show()
```


![png](/images/2019-07-21-Home_credit_default_risk/output_101_0.png)



```python
(pd.Series(rf.feature_importances_, index=app_train.columns)
   .nlargest(15)
   .plot(kind='barh'))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7b8d800be0>




![png](/images/2019-07-21-Home_credit_default_risk/output_102_1.png)


## Random forrest with a cross validation


```python
rf_cv = RandomForestClassifier()
scores = cross_val_score(rf_cv, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
scores
```




    array([0.63219173, 0.63231896, 0.62319801, 0.62533242, 0.62668538])




```python
rf_cv.fit(X_train, y_train)
roc_auc_score(y_test, rf_cv.predict(X_test))
```




    0.5036931060869384




```python
#!pip install kaggle
```


```python
#!kaggle competitions submit -c home-credit-default-risk -f randomforest_baseline.csv -m "My 1st submission - the baseline"
```

---

# More advanced models

## LightGBM 


```python
lgbm = lgb.LGBMClassifier(random_state = 50)
lgbm.fit(X_train, y_train, eval_metric = 'auc')
roc_auc_score(y_train, lgbm.predict(X_train))
```




    0.5108945200660795




```python
roc_auc_score(y_test, lgbm.predict(X_test))
```




    0.5069964776833696



Different tests on hyperparameters and results:

* underfitting / high biais -> let's try to complified the model
* max_depth = 7/11 or objective = 'binary' -> scores 0.508 / 0.508
* n_estimators=1000 -> scores 0.57 / 0.511
* class_weight = 'balanced' -> scores 0.71 / 0.68
* reg_alpha = 0.1, reg_lambda = 0.1 -> no influence


```python
lgbm = lgb.LGBMClassifier(random_state = 50, n_jobs = -1, class_weight = 'balanced')
lgbm.fit(X_train, y_train, eval_metric = 'auc')
roc_auc_score(y_train, lgbm.predict(X_train))
```




    0.7121220817520526




```python
roc_auc_score(y_test, lgbm.predict(X_test))
```




    0.6846561563080866




```python
def submit_func(model, X_Test, file_name):
    model.fit(app_train, y)
    result = model.predict_proba(app_test)[:, 1]
    submit = app_test[['SK_ID_CURR']]
    submit['TARGET'] = result
    print(submit.head())
    print(submit.shape)
    submit.to_csv(file_name + '.csv', index=False)
```


```python
submit_func(lgbm, app_test, 'lgbm_submission')
```

    /home/sunflowa/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    

       SK_ID_CURR    TARGET
    0      100002  0.876184
    1      100003  0.237339
    2      100004  0.293612
    3      100006  0.427514
    4      100007  0.634639
    (307511, 2)
    

__submission -> 0.72057__

--- 

# Using XGBoost and weighted classes

As said earlier, there are far more 0 than 1 in the target column. This is an [imbalanced class problem].(http://www.chioka.in/class-imbalance-problem/).
    
It's a common problem affecting ML due to having disproportionate number of class instances in practice.
This is why the ROC AUC metric suits our needs here. There are 2 class of approaches out there to deal with this problem:

1) sampling based, that can be broken into three major categories: 

    a) over sampling 
    
    b) under sampling 
    
    c) hybrid of oversampling and undersampling.

2) cost function based. 

With default or few changes in hyperparameters

* base score : 0.50 / 0.709
* max_delta_step=2 -> unchanged
* with ratio : 0.68 / 0.71


```python
y.shape[0], y.sum()
```




    (307511, 24825)




```python
ratio = (y.shape[0] - y.sum()) / y.sum()
ratio
```




    11.387150050352467




```python
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=50, eval_metric='auc', 
                              max_delta_step=2, scale_pos_weight=20)
xgb_model.fit(X_train, y_train)
roc_auc_score(y_train, xgb_model.predict(X_train))
```




    0.656946528564419




```python
roc_auc_score(y_test, xgb_model.predict(X_test))
```




    0.6488130660302404



For common cases when the dataset is extremely imbalanced, this can affect the training of XGBoost model, and there are two ways to improve it.

If you care only about the overall performance metric (AUC) of your prediction
Balance the positive and negative weights via scale_pos_weight
Use AUC for evaluation

If you care about predicting the right probability
In such a case, you cannot re-balance the dataset
Set parameter max_delta_step to a finite number (say 1) to help convergence


```python
submit_func(xgb_model, app_test, 'xgb_submission')
```

    /home/sunflowa/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    

       SK_ID_CURR    TARGET
    0      100002  0.908168
    1      100003  0.408862
    2      100004  0.452180
    3      100006  0.607726
    4      100007  0.763639
    (307511, 2)
    

submission -> 0.72340

# Credits / side notes
[Will Koehrsen](https://www.kaggle.com/willkoehrsen/) for many interesting tips in his kernel !

This notebook is intended to be an introduction to machine learning. So many things are missing or can be done better, such as :

* Using function to clean / prepare the data
* Exploring the other tables and select other columns that can be relevant
* Doing more feature engineering, this will lead to a better score !
