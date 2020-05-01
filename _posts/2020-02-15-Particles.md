---
title: "Cirta - Particle Type Classification"
date: 2020-02-15
categories:
  - Data Science
tags: [Zindi Competitions]
header:
    image: "/images/2020-02-15-Particles/banner.png"
excerpt: "Build a machine learning model to help physicists identify particles in images"
mathjax: "true"
---

Banner made from a photo by [Moritz Kindler](https://unsplash.com/@moritz_photography) on Unsplash

__Build a machine learning model to help physicists identify particles in images__

---

# Short Introduction

__If you're in a hurry...__

Proposed by [Zindi](https://zindi.africa/competitions/tic-heap-cirta-particle-classification-challenge)

In this challenge we want to build a machine learning model to help us recognize particles. Particles are the tiny constituant of matter generated in a collision between proton bunches at the Large Hadron Collider at CERN. 

Particles are of course of different types and identifying which particle was produced in an extremly important task for particle physicists. 

Our dataset comprises 350 independent simulated events, where each event contains labelled images of particle trajectories. 

A good model assigns the correct particle type to any particle, even the least frequent ones.

Read throught this notebook to discover more about the particles.

---

# Longer Introduction

__With in depth explanations if you have more time :)__

This challenge is part of an effort to explore the use of machine learning to assist high energy physicists in discovering and characterizing new particles.

Particles are the tiny constituents of matter generated in a collision between proton bunches. Physicists at CERN study particles using [particle accelerators](https://home.cern/science/accelerators/how-accelerator-works). [The Large Hadron Collider (LHC) at CERN](https://home.cern/science/accelerators/large-hadron-collider) is the world’s largest and most powerful particle accelerator and is used to accelerate and collide protons as well as heavy lead ions. The LHC consists of a 27-kilometre ring of superconducting magnets with a number of accelerating structures to boost the energy of the particles along the way.

In the LHC, proton bunches (beams) circulates and collide at high energy. Each beam collision (also called an event) produces a firework of new particles. To identify the types of these particles, a complex apparatus, the detector records the small energy deposited by the particles when they impact well-defined locations in the detector.

Particle Identification (PID) is fundamental to particle physics experiments. Currently no machine learning solution exists for PID.

The goal of this challenge is to build a machine learning model to read images of particles and identify their type.

This challenge was provided by Sabrina Amrouche and Dalila Salamani who are researchers at CERN and are hosting a session on machine learning at the 10th Conference on High Energy and Astro Particles.

__About the Tenth International Conference on High Energy and Astro Particles ([event](https://indico.cern.ch/event/776520/))__

This Tenth edition of the International Conference on High Energy and Astroparticle Physics (TIC-HEAP) will be held at Mentouri University, Constantine in Algeria during the period of 19th-21st October 2019. Held in close coordination with the DGRSDT (The Algerian General Direction of Scientific Research), it will focus on discussing the latest development on particle physics, astroparticle and cosmology, as well as strategically planning for Algeria to become an active participating member of the CERN.

---

# The Data Set

The training dataset comprises of 350 independent simulated events (collisions). Where each event contains approximately 3,000 labeled images of different particle trajectories passing through many detectors resulting from the collision. The events were simulated with [ACTS in the context of the TRACKML challenge](https://hal.inria.fr/hal-01745714/document) and were modified to target not particle tracking but rather particle identification.

If you are curious to learn about the original format of the dataset (which has also geometry and clusters information), checkout the dataset description and files here (you have to sign in) : https://competitions.codalab.org/competitions/20112#participate-get-data

This is the multiclass classification computer vision problem to identify particles by five types, labeled as follows:

- 11: "electron"
- 13: "muon"
- 211: "pion"
- 321: "kaon"
- 2212: "proton"

![title](/images/2020-02-15-Particles/fig1.png)
Fig 1 Transverse plane of the TrackML detector with the particle in red

![title](/images/2020-02-15-Particles/fig2.png)
Fig 2 Translated particle with RZ binning

__Files available :__

- __The training data__ consists of 350 .pkl files, each representing a unique event (or collision). Each .pkl file contains two columns: Column 1 is a list of 10x10 images. Column 2 is the particle type associated to the image (int). Training data can be downloaded at: https://cernbox.cern.ch/index.php/s/OH9tOo8VHYpHJDl. This data is open-source data.
- __SampleSubmission.csv__ - is an example of what your submission file should look like.
- __Data_test_file.pkl__ is the test set which contains about 4,000 images of particles (not associated with any specific event)
- __cirtaChallenge.ipynb__: is a starter python notebook. It shows you how to open and view a .pkl file and starts you off with a simple classifier.


__Note that the training set is highly imbalanced, but the test set has been designed to be balanced.__

---

# First insight


```python
#Import libraries to load and process data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import glob
import random
import os

import warnings
warnings.filterwarnings('ignore')
```


```python
# code to particle name dictionary : 
dic_types = {11: "electron", 13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}
```


```python
# load a pickle file
event = pickle.load(open('../../Desktop/particle_train_data/event1.pkl', 'rb'))

# get the data and target
data, target = event[0], event[1]
target = target.astype('int')
```


```python
event.shape, data.shape, target.shape, data[0].shape, target[0].shape
```




    ((2, 3598), (3598,), (3598,), (10, 10), ())




```python
data[3597]
```




    array([[2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 3., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 2., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])




```python
target[3597]
```




    2212




```python
target.dtype, data.dtype
```




    (dtype('int32'), dtype('O'))




```python
target[0].dtype, data[0].dtype
```




    (dtype('int32'), dtype('float64'))



Distribution of particles in an event


```python
from collections import Counter

plt.bar(range(len(dic_types)),list(Counter(target).values()))
plt.xticks(range(len(dic_types)), [dic_types[i] for i in list(Counter(target).keys())])
plt.show()
```


![png](/images/2020-02-15-Particles/output_12_0.png)



```python
# display 10 randomly choosen images for each collision
for j in [11, 13, 211, 321, 2212]:
    plt.figure(figsize=(12, 10))
    data_tmp = data[np.where(target==j)]
    for i in range(1, 6):
        plt.subplot(1, 5, i)
        num = random.randint(0, data_tmp.shape[0]-1)
        plt.axis('off')
        plt.title(dic_types[j])
        plt.imshow(data_tmp[num])
    plt.show()
```


![png](/images/2020-02-15-Particles/output_13_0.png)



![png](/images/2020-02-15-Particles/output_13_1.png)



![png](/images/2020-02-15-Particles/output_13_2.png)



![png](/images/2020-02-15-Particles/output_13_3.png)



![png](/images/2020-02-15-Particles/output_13_4.png)


---
# Load all the data & preparation

Transform all pickles files into two numpy arrays (data & target)


```python
pkls = glob.glob('../../Desktop/particle_train_data/*.pkl')


def check_consistency():
    """Check consistency of types and shapes of the data"""
    for pk in pkls:
        event = pickle.load(open(pk, 'rb'))
        if event.shape[0] != 2:
            print("shape different for :", pk[39:-4], event.shape)
        data, target = event[0], event[1]
        target = target.astype('int')
        if data.shape != target.shape:
            print("inconsistent shapes between data & target for ", pk[39:-4], data.shape, event.shape)
        i = 0
        for d, t in zip(data, target):
            if d.dtype != 'float64' or t.dtype != 'int64':
                print("pb of type", i)
            if d.shape != (10, 10) or t.shape != ():
                print("pb of size")
            i += 1
            

def seperate_d_t(pkl):
    "load the pickle file & return flatten data & target"
    event = pickle.load(open('../../Desktop/particle_train_data/event1.pkl', 'rb'))
    data, target = event[0], event[1].astype('int')
    data = np.array([d.reshape(100) for d in data])
    return data, target


def prepare_data():
    """Merge all the pickle files"""
    # initialize the first element
    data, target = seperate_d_t('../../Desktop/particle_train_data/event1.pkl')


    # loop to concatenate the 1st with all other elts
    pkls.remove('../../Desktop/particle_train_data/event1.pkl')
    for pk in pkls:
        try:
            data_tmp, target_tmp = seperate_d_t(pk)
            data, target = np.concatenate((data, data_tmp), axis=0), np.concatenate((target, target_tmp), axis=0)
        except:
            print("pb for pk: ", pkpk[39:-4])
    return data, target
 

X_path, y_path = '../../Desktop/particle_train_data/X.csv', '../../Desktop/particle_train_data/y.csv'
        
        
# if the file are present load it instead of repreparing data        
if os.path.isfile(X_path) and os.path.isfile(y_path):
    data = np.array(pd.read_csv(X_path))
    target = np.array(pd.read_csv(y_path))
else :
    check_consistency()
    data, target = prepare_data()
    pd.DataFrame(data).to_csv(X_path, index=False)
    pd.DataFrame(target).to_csv(y_path, index=False)
    
         
data.shape, target.shape, "//", data[0].shape, target[0].shape, "//", \
data[data.shape[0]-1].shape, target[target.shape[0]-1].shape
```




    ((1259300, 100), (1259300, 1), '//', (100,), (1,), '//', (100,), (1,))




```python
data[:1]
```




    array([[3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1.]])




```python
target[:1]
```




    array([[211]], dtype=int64)




```python
freq = np.unique(target, return_counts=True)
freq
```




    (array([  11,   13,  211,  321, 2212], dtype=int64),
     array([  3150,    700, 981050, 160300, 114100], dtype=int64))




```python
print('ratio:', np.unique(target, return_counts=True)[1][1] / np.unique(target, return_counts=True)[1][3] * 100, '%')
```

    ratio: 0.43668122270742354 %
    

---
# Basic Machine Learning Models & Predictions

why stratify is important ?????????????????????????????


```python
from sklearn.model_selection import train_test_split

# preprocess dataset, split into training and test part
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42, stratify=target)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((1007440, 100), (251860, 100), (1007440, 1), (251860, 1))



Models à éviter : 
- KNeighborsClassifier
- SVC(gamma='auto')
- xgboost


```python
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

import lightgbm as lgbm
```


```python
model_list = [RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), RidgeClassifier(), 
             AdaBoostClassifier(), lgbm.LGBMClassifier(n_jobs = -1)
             ]
model_names = []
train_score, test_score, f1_train, f1_test = [], [], [], []

   
def train_all_models(prefix):
    """Train all the models in the list, & fill the different lists with the predictions' scores"""
    
    # creation of list of names and scores for the train / test
    model_names_tmp = [str(m)[:str(m).index('(')] for m in model_list]
    model_names.extend([(prefix + name) for name in model_names_tmp])

    # iterate over classifiers
    for clf, name in zip(model_list, model_names_tmp):
        pickle_filename = prefix + name
        if os.path.isfile(pickle_filename):          # if model alreday been trained load it
            with open(pickle_filename, 'rb') as f:
                clf = pickle.load(f)
        else:                                        # otherwise fit model and serialize it
            clf.fit(X_train, y_train)
            with open(pickle_filename, 'wb') as f:
                pickle.dump(clf, f)          
            
        train_score.append(clf.score(X_train, y_train))
        test_score.append(clf.score(X_test, y_test))
        #print(name, f"train_score : {train_score[-1]:.2f}, test_score : {test_score[-1]:.2f}")
        
        #f1_train.append(f1_score(y_train, clf.predict(X_train)))
        #f1_test.append(f1_score(y_test, clf.predict(X_test)))
        #f1_train : {f1_train[-1]:.2f}, f1_test : {f1_test[-1]:.2f}, ")
        
        
train_all_models("1_original_data_")


def display_scores():
    """Print table & plot of all the models' scores"""
    df_score = pd.DataFrame({'model_names' : model_names,
                             'train_score' : train_score,
                             'test_score' : test_score})
    df_score = pd.melt(df_score, id_vars=['model_names'], value_vars=['train_score', 'test_score'])
    print(df_score.head(50))
    plt.figure(figsize=(12, len(model_names)/2))
    sns.barplot(y="model_names", x="value", hue="variable", data=df_score)

display_scores()
```

                                   model_names     variable     value
    0   1_original_data_RandomForestClassifier  train_score  0.902241
    1   1_original_data_DecisionTreeClassifier  train_score  0.902326
    2      1_original_data_ExtraTreeClassifier  train_score  0.902326
    3          1_original_data_RidgeClassifier  train_score  0.783943
    4       1_original_data_AdaBoostClassifier  train_score  0.577456
    5           1_original_data_LGBMClassifier  train_score  0.828249
    6   1_original_data_RandomForestClassifier   test_score  0.900484
    7   1_original_data_DecisionTreeClassifier   test_score  0.900147
    8      1_original_data_ExtraTreeClassifier   test_score  0.900147
    9          1_original_data_RidgeClassifier   test_score  0.784460
    10      1_original_data_AdaBoostClassifier   test_score  0.577892
    11          1_original_data_LGBMClassifier   test_score  0.828194
    


![png](/images/2020-02-15-Particles/output_26_1.png)


---
# Using the Synthetic Minority Over-sampling Technique

How SMOTe works
Brief description on SMOTe (Synthetic Minority Over-sampling Technique):

it creates synthetic observations of the minority class (bad loans) by:

Finding the k-nearest-neighbors for minority class observations (finding similar observations)
Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.
More explanations can be found here

An other informative article

Oversampling is a well-known way to potentially improve models trained on imbalanced data. But it’s important to remember that oversampling incorrectly can lead to thinking a model will generalize better than it actually does. Random forests are great because the model architecture reduces overfitting (see Brieman 2001 for a proof), but poor sampling practices can still lead to false conclusions about the quality of a model.

When the model is in production, it’s predicting on unseen data. The main point of model validation is to estimate how the model will generalize to new data. If the decision to put a model into production is based on how it performs on a validation set, it’s critical that oversampling is done correctly.



```python
np.unique(y_train, return_counts=True)
```




    (array([  11,   13,  211,  321, 2212], dtype=int64),
     array([  2516,    553, 784766, 128270,  91335], dtype=int64))




```python
def percent_electron_muon():
    """print total nb of particules, percentage of electrons & muons, and the ratio to reach"""
    total = np.unique(y_train, return_counts=True)[1].sum()
    nb_elec = np.unique(y_train, return_counts=True)[1][0]
    nb_muon = np.unique(y_train, return_counts=True)[1][1]
    print(f"total nb of particules : {total}")
    print(f"percentage of electrons : {nb_elec/total*100:.2f}%")
    print(f"percentage of muons : {nb_muon/total*100:.2f}%")

    
percent_electron_muon()
```

    total nb of particules : 1007440
    percentage of electrons : 0.25%
    percentage of muons : 0.05%
    


```python
# dic_types = {11: "electron", 13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}
total = np.unique(y_train, return_counts=True)[1].sum()
nb_elec_goal = int(round(total*1.5/100))
nb_muon_goal = int(round(total*0.8/100))
nb_elec_goal, nb_muon_goal
```




    (15112, 8060)




```python
X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
X_train.shape, y_train.shape
```




    ((1007440, 100), (1007440, 1))



References :
- https://www.kaggle.com/obrunet/credit-card-fraud-detection
- https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets#t72
- https://www.kaggle.com/gargmanish/how-to-handle-imbalance-data-study-in-detail


```python
from imblearn.over_sampling import SMOTE

#The SMOTE is applied on the train set ONLY -When sampling_strategy is a dict, the keys correspond to the targeted classes. 
# The values correspond to the desired number of samples for each targeted class. 
# This is working for both under- and over-sampling algorithms but not for the cleaning algorithms. Use a list instead.
sm = SMOTE(sampling_strategy={11: nb_elec_goal, 13: nb_muon_goal})
X_train, y_train = sm.fit_resample(X_train, y_train)
X_train.shape, y_train.shape
```




    ((1027543, 100), (1027543,))




```python
np.unique(y_train, return_counts=True)
```




    (array([  11,   13,  211,  321, 2212], dtype=int64),
     array([ 15112,   8060, 784766, 128270,  91335], dtype=int64))




```python
train_all_models("2_smote_")
display_scores()
```

                                   model_names     variable     value
    0   1_original_data_RandomForestClassifier  train_score  0.902241
    1   1_original_data_DecisionTreeClassifier  train_score  0.902326
    2      1_original_data_ExtraTreeClassifier  train_score  0.902326
    3          1_original_data_RidgeClassifier  train_score  0.783943
    4       1_original_data_AdaBoostClassifier  train_score  0.577456
    5           1_original_data_LGBMClassifier  train_score  0.828249
    6           2_smote_RandomForestClassifier  train_score  0.900341
    7           2_smote_DecisionTreeClassifier  train_score  0.900457
    8              2_smote_ExtraTreeClassifier  train_score  0.900457
    9                  2_smote_RidgeClassifier  train_score  0.770294
    10              2_smote_AdaBoostClassifier  train_score  0.646488
    11                  2_smote_LGBMClassifier  train_score  0.832237
    12  1_original_data_RandomForestClassifier   test_score  0.900484
    13  1_original_data_DecisionTreeClassifier   test_score  0.900147
    14     1_original_data_ExtraTreeClassifier   test_score  0.900147
    15         1_original_data_RidgeClassifier   test_score  0.784460
    16      1_original_data_AdaBoostClassifier   test_score  0.577892
    17          1_original_data_LGBMClassifier   test_score  0.828194
    18          2_smote_RandomForestClassifier   test_score  0.899575
    19          2_smote_DecisionTreeClassifier   test_score  0.899103
    20             2_smote_ExtraTreeClassifier   test_score  0.899103
    21                 2_smote_RidgeClassifier   test_score  0.784722
    22              2_smote_AdaBoostClassifier   test_score  0.658695
    23                  2_smote_LGBMClassifier   test_score  0.831402
    


![png](/images/2020-02-15-Particles/output_35_1.png)


---

# First submissions


```python
def train_single_model(model):
    """Train & return a single model, print score"""
    model.fit(X_train, y_train)
    print("score train", model.score(X_train, y_train))
    print("score test", model.score(X_test, y_test))
    return model
```


```python
def make_submission(trained_model, csv_name):
    """Load the test pickle file, make prediction with the trained model & create a csv for submission"""
    pkl_file = open('other_data/data_test_file.pkl', 'rb')
    test = pickle.load(pkl_file)
    ss = pd.DataFrame({'image':[t[0] for t in test]})
    test_preds = trained_model.predict_proba([t[1].flatten() for t in test])
    for i in range(len(trained_model.classes_)):
      ss[trained_model.classes_[i]] = test_preds[:,i]
    ss.head()
    ss.to_csv(csv_name + '.csv', index=False)
```


```python
rf = train_single_model(RandomForestClassifier(n_estimators=200, max_depth=5))
make_submission(rf, "submission_rf")
```

submission score = 2.95 // rank : 17th

---



```python
ab_clf = train_single_model(AdaBoostClassifier(n_estimators=200))
```

    score train 0.6194992129382441
    score test 0.6183531043437205
    


```python
# How well does it do?
confusion_matrix(y_test, ab_clf.predict(X_test))
```




    array([[  2160,      0,  15320,      0,   2141],
           [     0,      0,    392,      0,      0],
           [ 30720,     74, 165417,      0,      0],
           [  4295,      0,  27765,      0,      0],
           [  3253,      0,  19506,      0,     61]], dtype=int64)




```python
ad_final = AdaBoostClassifier(n_estimators=200).fit(X, y)
make_submission(ad_final, "submission_ab_final")
```

submission score : 3.87


```python
rf_whole_dataset = train_single_model(RandomForestClassifier(n_estimators=200, max_depth=5)).fit(X, y)
make_submission(rf_whole_dataset, "submission_rf_whole_dataset")
```

    score train 0.7403913453638051
    score test 0.7402435965533523
    

submission score :2.27



```python
def display_confusion_matx(best_model_path):
    if os.path.isfile(best_model_path):
        with open(best_model_path, 'rb') as f:
            best_clf = pickle.load(f)
        plot_confusion_mtx(best_clf)

        
display_confusion_matx('./1_original_data_AdaBoostClassifier')
```


![png](/images/2020-02-15-Particles/output_0_0.png)



```python
display_confusion_matx('1_original_data_LGBMClassifier')
```


![png](/images/2020-02-15-Particles/output_1_0.png)



```python
display_confusion_matx('1_original_data_RidgeClassifier')
```


![png](/images/2020-02-15-Particles/output_2_0.png)


---
# Using the Synthetic Minority Over-sampling Technique



```python
freq_array = np.unique(y, return_counts=True)
class_weight = {k: d for k, d in zip(freq_array[0], freq_array[1])}
class_weight
```




    {11: 3150, 13: 700, 211: 981050, 321: 160300, 2212: 114100}




```python
tot = sum([class_weight[k] for k in class_weight])
tot
```




    1259300




```python
percentages = {k: round(class_weight[k] / tot * 100, 2) for k in class_weight}
percentages
```




    {11: 0.25, 13: 0.06, 211: 77.9, 321: 12.73, 2212: 9.06}




```python
sampling_strategy = {11: int(round(tot / 100)), 13: int(round(tot / 200))}
sampling_strategy                                                    
```




    {11: 12593, 13: 6296}




```python
from imblearn.over_sampling import SMOTE

#The SMOTE is applied on the train set ONLY -When sampling_strategy is a dict, the keys correspond to the targeted classes. 
# The values correspond to the desired number of samples for each targeted class. 
# This is working for both under- and over-sampling algorithms but not for the cleaning algorithms. Use a list instead.
sm = SMOTE(sampling_strategy=sampling_strategy)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
pd.DataFrame(X_train_smote).to_csv('../../Desktop/particle_train_data/X_train_smote.csv', index=False)
pd.DataFrame(y_train_smote).to_csv('../../Desktop/particle_train_data/y_train_smote.csv', index=False)
X_train_smote.shape, y_train_smote.shape
```




    ((1023249, 100), (1023249,))




```python
np.unique(y_train_smote, return_counts=True)
```




    (array([  11,   13,  211,  321, 2212], dtype=int64),
     array([ 12593,   6296, 784840, 128240,  91280], dtype=int64))




```python
train_all_models("2_smote_")
```


```python
display_confusion_matx('./2_smote_AdaBoostClassifier')
```


![png](/images/2020-02-15-Particles/output_11_0.png)



```python
display_confusion_matx('./2_smote_LGBMClassifier')
```


![png](/images/2020-02-15-Particles/output_12_0_3.png)



```python
display_confusion_matx('./2_smote_RidgeClassifier')
```


![png](/images/2020-02-15-Particles/output_13_0_3.png)


## Randomforest with weighted classes


```python
freq_array = np.unique(y, return_counts=True)
class_weight = {k: d for k, d in zip(freq_array[0], freq_array[1])}
class_weight

rf = RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=1, class_weight=class_weight).fit(X_train, y_train)
plot_confusion_mtx(rf)
```


![png](/images/2020-02-15-Particles/output_15_0.png)



```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
```


```python
logreg = LogisticRegression(n_jobs = -1).fit(X_train, y_train)
plot_confusion_mtx(logreg)
```


![png](/images/2020-02-15-Particles/output_17_0.png)



```python
#linsvc = LinearSVC().fit(X_train, y_train)
#plot_confusion_mtx(linsvc)
```


```python
np.unique(y_test, return_counts=True)
```




    (array([  11,   13,  211,  321, 2212], dtype=int64),
     array([   630,    140, 196210,  32060,  22820], dtype=int64))



---

# Using Adasyn


```python
from imblearn.over_sampling import ADASYN

ad = ADASYN(sampling_strategy=sampling_strategy)
X_train_adasyn, y_train_adasyn = sm.fit_resample(X_train, y_train)
pd.DataFrame(X_train_adasyn).to_csv('../../Desktop/particle_train_data/X_train_adasyn.csv', index=False)
pd.DataFrame(y_train_adasyn).to_csv('../../Desktop/particle_train_data/y_train_adasyn.csv', index=False)
X_train_adasyn.shape, y_train_adasyn.shape
```




    ((1023249, 100), (1023249,))




```python
np.unique(y_train_adasyn, return_counts=True)
```




    (array([  11,   13,  211,  321, 2212], dtype=int64),
     array([ 12593,   6296, 784840, 128240,  91280], dtype=int64))




```python
train_all_models("3_adasyn_")
```


```python
display_confusion_matx('./3_adasyn_AdaBoostClassifier')
```


![png](/images/2020-02-15-Particles/output_24_0.png)



```python
display_confusion_matx('./3_adasyn_LGBMClassifier')
```


![png](/images/2020-02-15-Particles/output_25_0.png)



```python
display_confusion_matx('./3_adasyn_RidgeClassifier')
```


![png](/images/2020-02-15-Particles/output_26_0.png)


# Tensorflow

```python
np.unique(y, return_counts=True)
```




    (array([  11,   13,  211,  321, 2212]),
     array([  3150,    700, 981050, 160300, 114100]))




```python
# code to particle name dictionary : 
dic_types = {11: "electron", 13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}
dic_tf = {11: 0, 13: 1, 211: 2, 321: 3, 2212: 4}
y = np.array(pd.DataFrame(y).replace(dic_tf))
np.unique(y, return_counts=True)
```




    (array([0, 1, 2, 3, 4]), array([  3150,    700, 981050, 160300, 114100]))



---
# Data Prep


```python
from sklearn.model_selection import train_test_split

# preprocess dataset, split into training and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((1007440, 100), (251860, 100), (1007440, 1), (251860, 1))




```python
nb_classes = len(np.unique(y))
nb_classes
```




    5




```python
from tensorflow.keras.utils import to_categorical

y_train_cat = to_categorical(y_train, num_classes=nb_classes, dtype='int32')
y_test_cat = to_categorical(y_test, num_classes=nb_classes, dtype='int32')
y_train_cat.shape, y_test_cat.shape
```




    ((1007440, 5), (251860, 5))




```python
y_train_cat
```




    array([[0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           ...,
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)




```python
X.max(), X.min()
```




    (8.0, 0.0)




```python
#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

X_train, X_test = X_train / 8, X_test / 8
```


```python
np.unique(y_train, return_counts=True)
```




    (array([0, 1, 2, 3, 4]), array([  2520,    560, 784840, 128240,  91280]))




```python
np.unique(y_test, return_counts=True)
```




    (array([0, 1, 2, 3, 4]), array([   630,    140, 196210,  32060,  22820]))



# Base line


```python
freq_array = np.unique(y, return_counts=True)
class_weight = {k: d for k, d in zip(freq_array[0], freq_array[1])}
class_weight
```




    {0: 3150, 1: 700, 2: 981050, 3: 160300, 4: 114100}




```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y.flatten())
class_weights
```




    array([7.99555556e+01, 3.59800000e+02, 2.56724938e-01, 1.57117904e+00,
           2.20736196e+00])




```python
initial_bias = np.log([700/981050])
initial_bias
```




    array([-7.24529837])




```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(20, input_dim=input_dim, activation='relu'))#, use_bias = True, bias_initializer='zeros'))
model.add(Dense(10, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='SGD', # nadam
              metrics=['accuracy'])
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_15 (Dense)             (None, 20)                2020      
    _________________________________________________________________
    dense_16 (Dense)             (None, 10)                210       
    _________________________________________________________________
    dense_17 (Dense)             (None, 5)                 55        
    =================================================================
    Total params: 2,285
    Trainable params: 2,285
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#model.get_weights()
```


```python
y_train_cat.shape
```




    (1007440, 5)




```python
model.fit(X_train, y_train,
          class_weight=class_weight,
          epochs=8, 
          batch_size=1000, 
          validation_data=(X_test, y_test))
```

    Train on 1007440 samples, validate on 251860 samples
    Epoch 1/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6918 - acc: 0.7790 - val_loss: 0.6917 - val_acc: 0.7790
    Epoch 2/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6916 - acc: 0.7790 - val_loss: 0.6916 - val_acc: 0.7790
    Epoch 3/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6915 - acc: 0.7790 - val_loss: 0.6915 - val_acc: 0.7790
    Epoch 4/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6914 - acc: 0.7790 - val_loss: 0.6913 - val_acc: 0.7790
    Epoch 5/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6913 - acc: 0.7790 - val_loss: 0.6912 - val_acc: 0.7790
    Epoch 6/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6912 - acc: 0.7790 - val_loss: 0.6911 - val_acc: 0.7790
    Epoch 7/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6911 - acc: 0.7790 - val_loss: 0.6910 - val_acc: 0.7790
    Epoch 8/8
    1007440/1007440 [==============================] - 2s 2us/sample - loss: 0.6909 - acc: 0.7790 - val_loss: 0.6909 - val_acc: 0.7790
    




    <tensorflow.python.keras.callbacks.History at 0x7f648aa8a2e8>




```python
y_pred_train = model.predict(X_train, batch_size=1000)
```


```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train.flatten(), np.argmax(y_pred_train, axis=1))
```




    array([[     0,      0,   2520,      0,      0],
           [     0,      0,    560,      0,      0],
           [     0,      0, 784840,      0,      0],
           [     0,      0, 128240,      0,      0],
           [     0,      0,  91280,      0,      0]])



# Metric


```python
np.unique(y_test, return_counts=True)
```




    (array([0, 1, 2, 3, 4]), array([   630,    140, 196210,  32060,  22820]))




```python
np.unique(X_train, return_counts=True)
```




    (array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ]),
     array([93266949,  4772426,  1724837,   652990,   226204,    69197,
               28314,     2799,      284]))




```python
np.argmax(y_pred_train, axis=1)
```




    array([2, 2, 2, ..., 2, 2, 2])




```python
y_train.flatten()
```




    array([2, 2, 2, ..., 2, 4, 2])




```python
from sklearn.metrics import confusion_matrix, log_loss


import itertools    
class_names = ['11', '13', '211', '321', '2212']


def plot_confusion_mtx(trained_model):
    """Plot the confusion matrix with color and labels"""
    matrix = confusion_matrix(y_test.flatten(),
                              np.argmax(trained_model.predict(X_test, batch_size=50000),
                              axis=1))
    plt.clf()

    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')

    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

    # plot colorbar to the right
    plt.colorbar()
    fmt = 'd'

    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if matrix[i, j] > thresh else "black")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label',size=14)
    plt.xlabel('Predicted label',size=14)
    plt.show()
    
    
def display_confusion_matx(best_model_path):
    if os.path.isfile(best_model_path):
        with open(best_model_path, 'rb') as f:
            best_clf = pickle.load(f)
        plot_confusion_mtx(best_clf)
        
        
#display_confusion_matx('1_original_data_LGBMClassifier')
```


```python
np.unique(y_pred, return_counts=True)
```




    (array([0, 2]), array([3, 3]))




```python
plot_confusion_mtx(model)
```


![png](/images/2020-02-15-Particles/output_29_0.png)

# tensorflow with other param


Using a different model


```python
model = Sequential()
input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

model.add(Dense(10, input_dim=input_dim, activation='relu', name='input'))
model.add(Dense(20, activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(nb_classes, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
```

    WARNING:tensorflow:From C:\Anaconda3\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    


```python
# from sklearn.utils import class_weight
# class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#class_weight = {0 : 1., 1: 20.}
model.fit(X_train, y_train, epochs=3, batch_size=50000, class_weight={0: 3150, 1: 700, 2: 981050, 3: 160300, 4: 114100})
```

    WARNING:tensorflow:From C:\Anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/3
    1007440/1007440 [==============================] - 3s 3us/sample - loss: 1.5577 - acc: 0.4882
    Epoch 2/3
    1007440/1007440 [==============================] - 3s 3us/sample - loss: 1.1783 - acc: 0.7790
    Epoch 3/3
    1007440/1007440 [==============================] - 3s 3us/sample - loss: 0.5487 - acc: 0.7790
    




    <tensorflow.python.keras.callbacks.History at 0x212810bd2b0>




```python
score = model.evaluate(X_test, y_test, batch_size=50000)
score
```

    251860/251860 [==============================] - 0s 1us/sample - loss: 1.0910 - acc: 0.7790
    




    [1.0910033097309182, 0.7790439]




```python
%matplotlib inline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
predictions = model.predict(X_test, batch_size=50000)

LABELS = ['Normal','Fraud'] 

max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.show()
```


![png](/images/2020-02-15-Particles/output_4_0.png)

