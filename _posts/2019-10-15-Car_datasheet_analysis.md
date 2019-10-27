---
title: "Analysis of the Car Datasheet"
date: 2019-10-15
categories:
  - Data Science
tags: [Web scraping, Data Analysis]
header:
  image: "/images/banners/banner_code.png"
excerpt: "Making Scraped Data Usable"
mathjax: "true"
---

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

SCRAPED_CSV = 'scraped_cars.csv'
```

## Load CSV and Review


```python
df_raw = pd.read_csv(SCRAPED_CSV)
df = df_raw.copy() # keep a defensive copy of the original data
df.tail()
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>401</th>
      <td>Ford Mustang Gl</td>
      <td>4</td>
      <td>2790</td>
      <td>1982</td>
      <td>USA</td>
      <td>15.6</td>
      <td>27.0</td>
      <td>86.0</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>402</th>
      <td>Vw Pickup</td>
      <td>4</td>
      <td>2130</td>
      <td>1982</td>
      <td>Europe</td>
      <td>24.6</td>
      <td>44.0</td>
      <td>52.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>403</th>
      <td>Dodge Rampage</td>
      <td>4</td>
      <td>2295</td>
      <td>1982</td>
      <td>USA</td>
      <td>11.6</td>
      <td>32.0</td>
      <td>84.0</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Ford Ranger</td>
      <td>4</td>
      <td>2625</td>
      <td>1982</td>
      <td>USA</td>
      <td>18.6</td>
      <td>28.0</td>
      <td>79.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>405</th>
      <td>Chevy S-10</td>
      <td>4</td>
      <td>2720</td>
      <td>1982</td>
      <td>USA</td>
      <td>19.4</td>
      <td>31.0</td>
      <td>82.0</td>
      <td>119.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (406, 9)




```python
df.sample(5)
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>Pontiac Firebird</td>
      <td>6</td>
      <td>3282</td>
      <td>1971</td>
      <td>USA</td>
      <td>15.0</td>
      <td>19.0</td>
      <td>100.0</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Chevrolet Citation</td>
      <td>6</td>
      <td>2595</td>
      <td>1979</td>
      <td>USA</td>
      <td>11.3</td>
      <td>28.8</td>
      <td>115.0</td>
      <td>173.0</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Ford Pinto</td>
      <td>4</td>
      <td>2639</td>
      <td>1975</td>
      <td>USA</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>83.0</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Buick Lesabre Custom</td>
      <td>8</td>
      <td>4502</td>
      <td>1972</td>
      <td>USA</td>
      <td>13.5</td>
      <td>13.0</td>
      <td>155.0</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Chevrolet Impala</td>
      <td>8</td>
      <td>4997</td>
      <td>1973</td>
      <td>USA</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>150.0</td>
      <td>400.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>406.000000</td>
      <td>406.000000</td>
      <td>406.000000</td>
      <td>406.000000</td>
      <td>398.000000</td>
      <td>400.000000</td>
      <td>406.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.475369</td>
      <td>2979.413793</td>
      <td>1975.921182</td>
      <td>15.519704</td>
      <td>23.514573</td>
      <td>105.082500</td>
      <td>194.779557</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.712160</td>
      <td>847.004328</td>
      <td>3.748737</td>
      <td>2.803359</td>
      <td>7.815984</td>
      <td>38.768779</td>
      <td>104.922458</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>1613.000000</td>
      <td>1970.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>46.000000</td>
      <td>68.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>2226.500000</td>
      <td>1973.000000</td>
      <td>13.700000</td>
      <td>17.500000</td>
      <td>75.750000</td>
      <td>105.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>2822.500000</td>
      <td>1976.000000</td>
      <td>15.500000</td>
      <td>23.000000</td>
      <td>95.000000</td>
      <td>151.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000</td>
      <td>3618.250000</td>
      <td>1979.000000</td>
      <td>17.175000</td>
      <td>29.000000</td>
      <td>130.000000</td>
      <td>302.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>5140.000000</td>
      <td>1982.000000</td>
      <td>24.800000</td>
      <td>46.600000</td>
      <td>230.000000</td>
      <td>455.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Review Strategy


```python
df.territory.value_counts()
```




    USA       254
    Japan      79
    Europe     73
    Name: territory, dtype: int64




```python
df.cylinders.value_counts()
```




    4    207
    8    108
    6     84
    3      4
    5      3
    Name: cylinders, dtype: int64




```python
df.cylinders.value_counts().sort_index()
```




    3      4
    4    207
    5      3
    6     84
    8    108
    Name: cylinders, dtype: int64




```python
ax = sns.countplot(data=df, y='cylinders')
ax.set_title("Counts for Each Cylinder Type");
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_11_0.png)



```python
ax = sns.countplot(data=df, x='year')
ax.set_title('Counts of Cars per Year');
plt.xticks(rotation=45);
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_12_0.png)



```python
df.isnull().sum()
```




    name            0
    cylinders       0
    weight          0
    year            0
    territory       0
    acceleration    0
    mpg             8
    hp              6
    displacement    0
    dtype: int64




```python
df.mpg.isnull()[5:15]
```




    5     False
    6     False
    7     False
    8     False
    9     False
    10     True
    11     True
    12     True
    13     True
    14     True
    Name: mpg, dtype: bool




```python
df[df.mpg.isnull()]
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>Citroen Ds-21 Pallas</td>
      <td>4</td>
      <td>3090</td>
      <td>1970</td>
      <td>Europe</td>
      <td>17.5</td>
      <td>NaN</td>
      <td>115.0</td>
      <td>133.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Chevrolet Chevelle Concours (Sw)</td>
      <td>8</td>
      <td>4142</td>
      <td>1970</td>
      <td>USA</td>
      <td>11.5</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Ford Torino (Sw)</td>
      <td>8</td>
      <td>4034</td>
      <td>1970</td>
      <td>USA</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>153.0</td>
      <td>351.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Plymouth Satellite (Sw)</td>
      <td>8</td>
      <td>4166</td>
      <td>1970</td>
      <td>USA</td>
      <td>10.5</td>
      <td>NaN</td>
      <td>175.0</td>
      <td>383.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Amc Rebel Sst (Sw)</td>
      <td>8</td>
      <td>3850</td>
      <td>1970</td>
      <td>USA</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>175.0</td>
      <td>360.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ford Mustang Boss 302</td>
      <td>8</td>
      <td>3353</td>
      <td>1970</td>
      <td>USA</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>140.0</td>
      <td>302.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Volkswagen Super Beetle 117</td>
      <td>4</td>
      <td>1978</td>
      <td>1971</td>
      <td>Europe</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>367</th>
      <td>Saab 900S</td>
      <td>4</td>
      <td>2800</td>
      <td>1981</td>
      <td>Europe</td>
      <td>15.4</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>121.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.hp.isnull()]
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>Ford Pinto</td>
      <td>4</td>
      <td>2046</td>
      <td>1971</td>
      <td>USA</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Ford Maverick</td>
      <td>6</td>
      <td>2875</td>
      <td>1974</td>
      <td>USA</td>
      <td>17.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>337</th>
      <td>Renault Lecar Deluxe</td>
      <td>4</td>
      <td>1835</td>
      <td>1980</td>
      <td>Europe</td>
      <td>17.3</td>
      <td>40.9</td>
      <td>NaN</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>343</th>
      <td>Ford Mustang Cobra</td>
      <td>4</td>
      <td>2905</td>
      <td>1980</td>
      <td>USA</td>
      <td>14.3</td>
      <td>23.6</td>
      <td>NaN</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>Renault 18I</td>
      <td>4</td>
      <td>2320</td>
      <td>1981</td>
      <td>Europe</td>
      <td>15.8</td>
      <td>34.5</td>
      <td>NaN</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>382</th>
      <td>Amc Concord Dl</td>
      <td>4</td>
      <td>3035</td>
      <td>1982</td>
      <td>USA</td>
      <td>20.5</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>151.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def highlight_max(s):
    '''Highlight the maximum in a Series yellow'''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

df.pivot_table(index=['year'], aggfunc='count').style.apply(highlight_max, axis=1) 
```




<style  type="text/css" >
    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col3 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col7 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col0 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col1 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col2 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col4 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col5 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col6 {
            background-color:  yellow;
        }    #T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col7 {
            background-color:  yellow;
        }</style><table id="T_8a0cd950_e906_11e9_95c8_057a714e3e06" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >acceleration</th>        <th class="col_heading level0 col1" >cylinders</th>        <th class="col_heading level0 col2" >displacement</th>        <th class="col_heading level0 col3" >hp</th>        <th class="col_heading level0 col4" >mpg</th>        <th class="col_heading level0 col5" >name</th>        <th class="col_heading level0 col6" >territory</th>        <th class="col_heading level0 col7" >weight</th>    </tr>    <tr>        <th class="index_name level0" >year</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row0" class="row_heading level0 row0" >1970</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col0" class="data row0 col0" >35</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col1" class="data row0 col1" >35</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col2" class="data row0 col2" >35</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col3" class="data row0 col3" >35</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col4" class="data row0 col4" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col5" class="data row0 col5" >35</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col6" class="data row0 col6" >35</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row0_col7" class="data row0 col7" >35</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row1" class="row_heading level0 row1" >1971</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col0" class="data row1 col0" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col1" class="data row1 col1" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col2" class="data row1 col2" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col3" class="data row1 col3" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col4" class="data row1 col4" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col5" class="data row1 col5" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col6" class="data row1 col6" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row1_col7" class="data row1 col7" >29</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row2" class="row_heading level0 row2" >1972</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col0" class="data row2 col0" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col1" class="data row2 col1" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col2" class="data row2 col2" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col3" class="data row2 col3" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col4" class="data row2 col4" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col5" class="data row2 col5" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col6" class="data row2 col6" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row2_col7" class="data row2 col7" >28</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row3" class="row_heading level0 row3" >1973</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col0" class="data row3 col0" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col1" class="data row3 col1" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col2" class="data row3 col2" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col3" class="data row3 col3" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col4" class="data row3 col4" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col5" class="data row3 col5" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col6" class="data row3 col6" >40</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row3_col7" class="data row3 col7" >40</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row4" class="row_heading level0 row4" >1974</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col0" class="data row4 col0" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col1" class="data row4 col1" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col2" class="data row4 col2" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col3" class="data row4 col3" >26</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col4" class="data row4 col4" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col5" class="data row4 col5" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col6" class="data row4 col6" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row4_col7" class="data row4 col7" >27</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row5" class="row_heading level0 row5" >1975</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col0" class="data row5 col0" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col1" class="data row5 col1" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col2" class="data row5 col2" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col3" class="data row5 col3" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col4" class="data row5 col4" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col5" class="data row5 col5" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col6" class="data row5 col6" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row5_col7" class="data row5 col7" >30</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row6" class="row_heading level0 row6" >1976</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col0" class="data row6 col0" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col1" class="data row6 col1" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col2" class="data row6 col2" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col3" class="data row6 col3" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col4" class="data row6 col4" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col5" class="data row6 col5" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col6" class="data row6 col6" >34</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row6_col7" class="data row6 col7" >34</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row7" class="row_heading level0 row7" >1977</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col0" class="data row7 col0" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col1" class="data row7 col1" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col2" class="data row7 col2" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col3" class="data row7 col3" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col4" class="data row7 col4" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col5" class="data row7 col5" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col6" class="data row7 col6" >28</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row7_col7" class="data row7 col7" >28</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row8" class="row_heading level0 row8" >1978</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col0" class="data row8 col0" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col1" class="data row8 col1" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col2" class="data row8 col2" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col3" class="data row8 col3" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col4" class="data row8 col4" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col5" class="data row8 col5" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col6" class="data row8 col6" >36</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row8_col7" class="data row8 col7" >36</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row9" class="row_heading level0 row9" >1979</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col0" class="data row9 col0" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col1" class="data row9 col1" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col2" class="data row9 col2" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col3" class="data row9 col3" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col4" class="data row9 col4" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col5" class="data row9 col5" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col6" class="data row9 col6" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row9_col7" class="data row9 col7" >29</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row10" class="row_heading level0 row10" >1980</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col0" class="data row10 col0" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col1" class="data row10 col1" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col2" class="data row10 col2" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col3" class="data row10 col3" >27</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col4" class="data row10 col4" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col5" class="data row10 col5" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col6" class="data row10 col6" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row10_col7" class="data row10 col7" >29</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row11" class="row_heading level0 row11" >1981</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col0" class="data row11 col0" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col1" class="data row11 col1" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col2" class="data row11 col2" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col3" class="data row11 col3" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col4" class="data row11 col4" >29</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col5" class="data row11 col5" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col6" class="data row11 col6" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row11_col7" class="data row11 col7" >30</td>
            </tr>
            <tr>
                        <th id="T_8a0cd950_e906_11e9_95c8_057a714e3e06level0_row12" class="row_heading level0 row12" >1982</th>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col0" class="data row12 col0" >31</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col1" class="data row12 col1" >31</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col2" class="data row12 col2" >31</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col3" class="data row12 col3" >30</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col4" class="data row12 col4" >31</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col5" class="data row12 col5" >31</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col6" class="data row12 col6" >31</td>
                        <td id="T_8a0cd950_e906_11e9_95c8_057a714e3e06row12_col7" class="data row12 col7" >31</td>
            </tr>
    </tbody></table>




```python
print(f"Before we drop NaN rows we have {df.shape} rows")
df = df.dropna()
print(f"After we drop NaN rows we have {df.shape} rows")
```

    Before we drop NaN rows we have (406, 9) rows
    After we drop NaN rows we have (392, 9) rows



```python
df.cylinders.value_counts().sort_index()
```




    3      4
    4    199
    5      3
    6     83
    8    103
    Name: cylinders, dtype: int64




```python
df = df.query("cylinders != 3 and cylinders != 5").copy()
df.cylinders.value_counts().sort_index()
```




    4    199
    6     83
    8    103
    Name: cylinders, dtype: int64




```python
df['cylinders_label'] = df.cylinders.apply(lambda v: f"{v} cylinders")
df.cylinders_label.value_counts().sort_index()
```




    4 cylinders    199
    6 cylinders     83
    8 cylinders    103
    Name: cylinders_label, dtype: int64



### What Distribution Does MPG Have?


```python
print(f"Examples of MPG values: {list(df.mpg.sample(10).sort_values())}")
ax = df.mpg.hist()
ax.set_ylabel('Frequency')
ax.set_xlabel('Binned MPG')
ax.set_title('Histogram of Continuous MPG Values');
```

    Examples of MPG values: [14.0, 17.0, 18.0, 20.2, 23.5, 24.0, 25.0, 30.5, 36.0, 37.2]



![png](/images/2019-10-15-Car_datasheet_analysis/output_23_1.png)



```python
ax = df.weight.hist()
ax.set_ylabel('Frequency')
ax.set_xlabel('Binned Weight')
ax.set_title('Histogram of Continuous Weight (lbs) Values');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_24_0.png)


## Reviewing Our Goal


```python
df.pivot_table(index=['year'], aggfunc='mean')
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
      <th>acceleration</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>hp</th>
      <th>mpg</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1970</th>
      <td>12.948276</td>
      <td>6.758621</td>
      <td>281.413793</td>
      <td>147.827586</td>
      <td>17.689655</td>
      <td>3372.793103</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>15.000000</td>
      <td>5.629630</td>
      <td>213.888889</td>
      <td>107.037037</td>
      <td>21.111111</td>
      <td>3030.592593</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>15.185185</td>
      <td>5.925926</td>
      <td>223.870370</td>
      <td>121.037037</td>
      <td>18.703704</td>
      <td>3271.333333</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>14.333333</td>
      <td>6.461538</td>
      <td>261.666667</td>
      <td>131.512821</td>
      <td>17.076923</td>
      <td>3452.230769</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>16.173077</td>
      <td>5.230769</td>
      <td>170.653846</td>
      <td>94.230769</td>
      <td>22.769231</td>
      <td>2878.038462</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>16.050000</td>
      <td>5.600000</td>
      <td>205.533333</td>
      <td>101.066667</td>
      <td>20.266667</td>
      <td>3176.800000</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>15.941176</td>
      <td>5.647059</td>
      <td>197.794118</td>
      <td>101.117647</td>
      <td>21.573529</td>
      <td>3078.735294</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>15.507407</td>
      <td>5.555556</td>
      <td>195.518519</td>
      <td>104.888889</td>
      <td>23.444444</td>
      <td>3007.629630</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>15.802857</td>
      <td>5.371429</td>
      <td>179.142857</td>
      <td>99.600000</td>
      <td>24.168571</td>
      <td>2862.714286</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>15.660714</td>
      <td>5.857143</td>
      <td>207.535714</td>
      <td>102.071429</td>
      <td>25.082143</td>
      <td>3038.392857</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>17.084000</td>
      <td>4.160000</td>
      <td>117.720000</td>
      <td>77.000000</td>
      <td>34.104000</td>
      <td>2422.120000</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>16.325000</td>
      <td>4.642857</td>
      <td>136.571429</td>
      <td>81.035714</td>
      <td>30.185714</td>
      <td>2530.178571</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>16.510000</td>
      <td>4.200000</td>
      <td>128.133333</td>
      <td>81.466667</td>
      <td>32.000000</td>
      <td>2434.166667</td>
    </tr>
  </tbody>
</table>
</div>



## What Correlates with MPG?


```python
df.corr()['mpg'].sort_values()
```




    weight         -0.842681
    displacement   -0.817887
    cylinders      -0.794872
    hp             -0.780259
    acceleration    0.419337
    year            0.579778
    mpg             1.000000
    Name: mpg, dtype: float64



### Exploring MPG vs. Weight


```python
ax = df.plot(kind="scatter", x='weight', y='mpg')
ax.set_title('Scatterplot of MPG vs. Weight (lbs)');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_30_0.png)



```python
ax = df.plot(kind="scatter", x='weight', y='mpg', alpha=0.5)
ax.set_title('Scatterplot of MPG vs. Weight (lbs)');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_31_0.png)



```python
jg = sns.jointplot(data=df, x='weight', y='mpg');
jg.fig.suptitle('Somewhat Non-linear Relationship between HP and Weight');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_32_0.png)



```python
jg=sns.jointplot(data=df, x='weight', y='mpg', kind='hexbin')
jg.fig.suptitle('Hexbin of Counts for MPG vs. Weight');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_33_0.png)



```python
jg = sns.jointplot(data=df, x='weight', y='mpg').plot_joint(sns.kdeplot, zorder=0, n_levels=5)
jg.fig.suptitle('Density Estimate for MPG vs. Weight');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_34_0.png)


### Exploring HP vs. Weight


```python
ax = df.plot(kind="scatter", x='hp', y='weight')
ax.set_title('Reasonably Linear Relationship between HP and Weight');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_36_0.png)



```python
jg = sns.jointplot(data=df, x='hp', y='weight', kind='reg')
jg.fig.suptitle('Reasonably Linear Relationship between HP and Weight');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_37_0.png)



```python
df.query("hp > 200 and weight < 3500") # Fully equipped luxury entrant
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
      <th>cylinders_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>Buick Estate Wagon (Sw)</td>
      <td>8</td>
      <td>3086</td>
      <td>1970</td>
      <td>USA</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>225.0</td>
      <td>455.0</td>
      <td>8 cylinders</td>
    </tr>
  </tbody>
</table>
</div>



## Cylinders and Displacement


```python
sorted_cylinders_label = df.cylinders_label.value_counts().sort_index().index
sorted_cylinders_label
```




    Index(['4 cylinders', '6 cylinders', '8 cylinders'], dtype='object')




```python
ax = df.plot(kind='scatter', x='cylinders', y='displacement')
ax.set_title("Scatterplot of Cylinders vs. Displacement");
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_41_0.png)



```python
ax = sns.stripplot(data=df, x='cylinders_label', y='displacement', order=sorted_cylinders_label)
ax.set_title("Stripplot of Cylinders vs. Displacement with Jitter");
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_42_0.png)



```python
ax = sns.boxplot(data=df, x='cylinders_label', y='weight', order=sorted_cylinders_label)
ax.set_title("Boxplot of Cylinders vs. Weight");
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_43_0.png)



```python
ax = sns.boxplot(data=df, x='cylinders_label', y='displacement', order=sorted_cylinders_label, notch=True) 
ax.set_title("Notched Boxplot of Cylinders vs. Weight");
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_44_0.png)


# Looking at MPG over Time


```python
ax = df.plot(kind="scatter", x='year', y='mpg');
ax.set_title('Scatterplot of MPG by Year');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_46_0.png)



```python
ax = sns.stripplot(data=df, x='year', y='mpg')
ax.set_title('Stripplot of MPG by Year');
plt.xticks(rotation=45);
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_47_0.png)



```python
ax = sns.boxplot(data=df, x='year', y='mpg')
ax.set_title("Boxplot of MPG by Year");
plt.xticks(rotation=45);
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_48_0.png)



```python
df.query('year=="1978" and mpg > 40')
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
      <th>cylinders_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>251</th>
      <td>Volkswagen Rabbit Custom Diesel</td>
      <td>4</td>
      <td>1985</td>
      <td>1978</td>
      <td>Europe</td>
      <td>21.5</td>
      <td>43.1</td>
      <td>48.0</td>
      <td>90.0</td>
      <td>4 cylinders</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.scatterplot(data=df, x='year', y='mpg', size='weight')
ax.set_title('Stripplot of MPG by Year');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_50_0.png)



```python
ax = sns.scatterplot(data=df, x='year', y='mpg', hue='weight')
ax.set_title('Stripplot of MPG by Year');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_51_0.png)



```python
ax = sns.boxplot(data=df, x='year', y='weight')
ax.set_title('Boxplot of Weight by Year');
plt.xticks(rotation=45);
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_52_0.png)


# Text analysis


```python
df.head()
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
      <th>name</th>
      <th>cylinders</th>
      <th>weight</th>
      <th>year</th>
      <th>territory</th>
      <th>acceleration</th>
      <th>mpg</th>
      <th>hp</th>
      <th>displacement</th>
      <th>cylinders_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chevrolet Chevelle Malibu</td>
      <td>8</td>
      <td>3504</td>
      <td>1970</td>
      <td>USA</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>130.0</td>
      <td>307.0</td>
      <td>8 cylinders</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Buick Skylark 320</td>
      <td>8</td>
      <td>3693</td>
      <td>1970</td>
      <td>USA</td>
      <td>11.5</td>
      <td>15.0</td>
      <td>165.0</td>
      <td>350.0</td>
      <td>8 cylinders</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Plymouth Satellite</td>
      <td>8</td>
      <td>3436</td>
      <td>1970</td>
      <td>USA</td>
      <td>11.0</td>
      <td>18.0</td>
      <td>150.0</td>
      <td>318.0</td>
      <td>8 cylinders</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Amc Rebel Sst</td>
      <td>8</td>
      <td>3433</td>
      <td>1970</td>
      <td>USA</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>150.0</td>
      <td>304.0</td>
      <td>8 cylinders</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ford Torino</td>
      <td>8</td>
      <td>3449</td>
      <td>1970</td>
      <td>USA</td>
      <td>10.5</td>
      <td>17.0</td>
      <td>140.0</td>
      <td>302.0</td>
      <td>8 cylinders</td>
    </tr>
  </tbody>
</table>
</div>




```python
ser_car_makes = df.name.str.lower().str.split(n=1, expand=True)[0]
ser_car_makes.value_counts()[:10]
```




    ford          48
    chevrolet     43
    plymouth      31
    dodge         28
    amc           27
    toyota        25
    datsun        23
    buick         17
    pontiac       16
    volkswagen    15
    Name: 0, dtype: int64




```python
ser_car_makes.value_counts()[-10:]
```




    mercedes-benz    2
    cadillac         2
    toyouta          1
    triumph          1
    maxda            1
    chevroelt        1
    capri            1
    vokswagen        1
    nissan           1
    hi               1
    Name: 0, dtype: int64




```python
df['car_makes'] = ser_car_makes
ser_cars_by_territory = df[['territory', 'car_makes']].pivot_table(index=['territory', 'car_makes'], aggfunc='size')
df_cars_by_territory = pd.DataFrame(ser_cars_by_territory) # anonymous Series
df_cars_by_territory.columns = ['size'] # rename anonymous column 0 to a named column
df_cars_by_territory = df_cars_by_territory.sort_values(by=['territory', 'size'], ascending=False)
df_cars_by_territory
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
      <th></th>
      <th>size</th>
    </tr>
    <tr>
      <th>territory</th>
      <th>car_makes</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="15" valign="top">USA</th>
      <th>ford</th>
      <td>48</td>
    </tr>
    <tr>
      <th>chevrolet</th>
      <td>43</td>
    </tr>
    <tr>
      <th>plymouth</th>
      <td>31</td>
    </tr>
    <tr>
      <th>dodge</th>
      <td>28</td>
    </tr>
    <tr>
      <th>amc</th>
      <td>27</td>
    </tr>
    <tr>
      <th>buick</th>
      <td>17</td>
    </tr>
    <tr>
      <th>pontiac</th>
      <td>16</td>
    </tr>
    <tr>
      <th>mercury</th>
      <td>11</td>
    </tr>
    <tr>
      <th>oldsmobile</th>
      <td>10</td>
    </tr>
    <tr>
      <th>chrysler</th>
      <td>6</td>
    </tr>
    <tr>
      <th>chevy</th>
      <td>3</td>
    </tr>
    <tr>
      <th>cadillac</th>
      <td>2</td>
    </tr>
    <tr>
      <th>capri</th>
      <td>1</td>
    </tr>
    <tr>
      <th>chevroelt</th>
      <td>1</td>
    </tr>
    <tr>
      <th>hi</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Japan</th>
      <th>toyota</th>
      <td>25</td>
    </tr>
    <tr>
      <th>datsun</th>
      <td>23</td>
    </tr>
    <tr>
      <th>honda</th>
      <td>13</td>
    </tr>
    <tr>
      <th>mazda</th>
      <td>7</td>
    </tr>
    <tr>
      <th>subaru</th>
      <td>4</td>
    </tr>
    <tr>
      <th>maxda</th>
      <td>1</td>
    </tr>
    <tr>
      <th>nissan</th>
      <td>1</td>
    </tr>
    <tr>
      <th>toyouta</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="13" valign="top">Europe</th>
      <th>volkswagen</th>
      <td>15</td>
    </tr>
    <tr>
      <th>fiat</th>
      <td>8</td>
    </tr>
    <tr>
      <th>peugeot</th>
      <td>8</td>
    </tr>
    <tr>
      <th>volvo</th>
      <td>6</td>
    </tr>
    <tr>
      <th>vw</th>
      <td>6</td>
    </tr>
    <tr>
      <th>audi</th>
      <td>5</td>
    </tr>
    <tr>
      <th>opel</th>
      <td>4</td>
    </tr>
    <tr>
      <th>saab</th>
      <td>4</td>
    </tr>
    <tr>
      <th>renault</th>
      <td>3</td>
    </tr>
    <tr>
      <th>bmw</th>
      <td>2</td>
    </tr>
    <tr>
      <th>mercedes-benz</th>
      <td>2</td>
    </tr>
    <tr>
      <th>triumph</th>
      <td>1</td>
    </tr>
    <tr>
      <th>vokswagen</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
mask = df_cars_by_territory.apply(lambda x: x['size'] > 10, axis=1)
df_cars_by_territory[mask]
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
      <th></th>
      <th>size</th>
    </tr>
    <tr>
      <th>territory</th>
      <th>car_makes</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">USA</th>
      <th>ford</th>
      <td>48</td>
    </tr>
    <tr>
      <th>chevrolet</th>
      <td>43</td>
    </tr>
    <tr>
      <th>plymouth</th>
      <td>31</td>
    </tr>
    <tr>
      <th>dodge</th>
      <td>28</td>
    </tr>
    <tr>
      <th>amc</th>
      <td>27</td>
    </tr>
    <tr>
      <th>buick</th>
      <td>17</td>
    </tr>
    <tr>
      <th>pontiac</th>
      <td>16</td>
    </tr>
    <tr>
      <th>mercury</th>
      <td>11</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Japan</th>
      <th>toyota</th>
      <td>25</td>
    </tr>
    <tr>
      <th>datsun</th>
      <td>23</td>
    </tr>
    <tr>
      <th>honda</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Europe</th>
      <th>volkswagen</th>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



## Report

This report answers the following questions:

* Is there a relationship between Weight and MPG - yes, with more weight we generally see lower MPG
* Is there a relationship between Cylinders and Displacement - yes, with more Cylinders we see a higher Displacement
* Is there a relationship between MPG and Years - yes, generally as we move from 1970 to 1982 we see steady improvements in MPG


```python
print(f"df_raw had shape {df_raw.shape}, after dropping NaN entries we \
reduced from {df_raw.shape[0]} to {df.shape[0]} rows.")

assert ((df.mpg >= 7) & (df.mpg <=47)).all(), "Why do we see a wider range of MPG values now?"
assert ((df.year >= 1970) & (df.year <= 1982)).all(), "Why do we see a wider range of years now?"
```

    df_raw had shape (406, 9), after dropping NaN entries we reduced from 406 to 385 rows.


As weight increases we see a decrease in MPG, this relationship is slightly non-linear with a faster decrease in MPG associated with low-to-mid Weight.


```python
jg = sns.jointplot(data=df, x='weight', y='mpg').plot_joint(sns.kdeplot, zorder=0, n_levels=5)
jg.fig.suptitle('Density Estimate for MPG vs. Weight');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_62_0.png)


Acceleration gets worse as Horsepower decreases.


```python
jg = sns.jointplot(data=df, x='acceleration', y='hp', kind='reg')
jg.fig.suptitle('Regression plot for Acceleration vs. Horsepower');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_64_0.png)


An engine's cubic inches of displacement is strongly related to the count of Cylinders.


```python
ax = sns.boxplot(data=df, x='cylinders_label', y='displacement', order=sorted_cylinders_label, notch=True) 
sns.stripplot(data=df, x='cylinders_label', y='displacement', order=sorted_cylinders_label, alpha=0.5, ax=ax) 
ax.set_title("More Cylinders Generally Have\nHigher Cubic Inches of Displacement");
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_66_0.png)


Heavier vehicles tend to have more Horsepower, the heaviest vehicles have the highest count of Cylinders.


```python
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(data=df, x='weight', y='hp', hue='cylinders_label', 
                     hue_order=sorted_cylinders_label, ax=ax)
ax.set_title('Scatterplot of HP vs. Weight\nColoured by Cylinder Count');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_68_0.png)


Vehicle brands are strongly associated with Territory.


```python
cm = sns.light_palette("green", as_cmap=True)
mask = df_cars_by_territory.apply(lambda x: x['size'] > 10, axis=1)
df_cars_by_territory[mask].style.background_gradient(cmap=cm)
```




<style  type="text/css" >
    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row0_col0 {
            background-color:  #008000;
            color:  #f1f1f1;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row1_col0 {
            background-color:  #1f911f;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row2_col0 {
            background-color:  #69ba69;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row3_col0 {
            background-color:  #7cc57c;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row4_col0 {
            background-color:  #82c882;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row5_col0 {
            background-color:  #c1ebc1;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row6_col0 {
            background-color:  #c7eec7;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row7_col0 {
            background-color:  #e5ffe5;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row8_col0 {
            background-color:  #8fcf8f;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row9_col0 {
            background-color:  #9bd69b;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row10_col0 {
            background-color:  #daf9da;
            color:  #000000;
        }    #T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row11_col0 {
            background-color:  #cdf2cd;
            color:  #000000;
        }</style><table id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06" ><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >size</th>    </tr>    <tr>        <th class="index_name level0" >territory</th>        <th class="index_name level1" >car_makes</th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level0_row0" class="row_heading level0 row0" rowspan=8>USA</th>
                        <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row0" class="row_heading level1 row0" >ford</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row0_col0" class="data row0 col0" >48</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row1" class="row_heading level1 row1" >chevrolet</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row1_col0" class="data row1 col0" >43</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row2" class="row_heading level1 row2" >plymouth</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row2_col0" class="data row2 col0" >31</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row3" class="row_heading level1 row3" >dodge</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row3_col0" class="data row3 col0" >28</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row4" class="row_heading level1 row4" >amc</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row4_col0" class="data row4 col0" >27</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row5" class="row_heading level1 row5" >buick</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row5_col0" class="data row5 col0" >17</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row6" class="row_heading level1 row6" >pontiac</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row6_col0" class="data row6 col0" >16</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row7" class="row_heading level1 row7" >mercury</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row7_col0" class="data row7 col0" >11</td>
            </tr>
            <tr>
                        <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level0_row8" class="row_heading level0 row8" rowspan=3>Japan</th>
                        <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row8" class="row_heading level1 row8" >toyota</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row8_col0" class="data row8 col0" >25</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row9" class="row_heading level1 row9" >datsun</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row9_col0" class="data row9 col0" >23</td>
            </tr>
            <tr>
                                <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row10" class="row_heading level1 row10" >honda</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row10_col0" class="data row10 col0" >13</td>
            </tr>
            <tr>
                        <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level0_row11" class="row_heading level0 row11" >Europe</th>
                        <th id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06level1_row11" class="row_heading level1 row11" >volkswagen</th>
                        <td id="T_c9c4f4ba_e906_11e9_95c8_057a714e3e06row11_col0" class="data row11 col0" >15</td>
            </tr>
    </tbody></table>



Vehicles from the USA consistently have worse MPG than corresponding vehicles from Japan or Europe in this sample. Vehicles in each territory become more efficient over the Years.


```python
fg = sns.lmplot(data=df, x='year', y='mpg', hue='territory')
fg.fig.suptitle('MPG over Year by Territory');
```


![png](/images/2019-10-15-Car_datasheet_analysis/output_72_0.png)