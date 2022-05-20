---
title: "Wind energy generation 2/2 - exploratory data analysis"
date: 2021-04-04
categories:
  - Data Science
tags: [Kaggle Competitions]
header:
  image: "/images/2021-02-25-wind-energy-clustering/pexels-narcisa-aciko-1292464 - cropped.png"
excerpt: "In this 2nd part, let's analyze the data for a country of each cluster (variations over time and so on...)"
mathjax: "true"
---

banner made from an image of [Narcisa Aciko on pexels.com](https://www.pexels.com/fr-fr/photo/photo-de-lot-d-eoliennes-1292464/)


## Introduction

This dataset contains hourly estimates of an area’s energy potential for 1986-2015 as a percentage of a power plant’s maximum output.

In [the previous part](https://obrunet.github.io/data%20science/wind-energy-clustering/), we’ve made clusters of countries with similar profiles of wind generation. In this 2nd part we’re going to analyse and explore datas for one country representative of each cluster. As a reminder, here are what those 6 clusters made of :
Countries grouped by cluster  
cluster nb : 0 EE FI LT LV PL SE  
cluster nb : 1 ES PT  
cluster nb : 2 AT CH CZ HR HU IT SI SK  
cluster nb : 3 BE DE DK FR IE LU NL UK    
cluster nb : 4 CY NO  
cluster nb : 5 BG EL RO  


```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 300

import warnings
warnings.filterwarnings("ignore")

path = "../../../datasets/_classified/kaggle/"


countries_lst = ['FI', 'PT', 'IT', 'FR', 'NO', 'RO']
df_wind_on = pd.read_csv(path + "EMHIRES_WIND_COUNTRY_June2019.csv")
df_wind_on = df_wind_on[countries_lst]
df_wind_on.head(2)
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
      <th>FI</th>
      <th>PT</th>
      <th>IT</th>
      <th>FR</th>
      <th>NO</th>
      <th>RO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0,31303</td>
      <td>0,22683</td>
      <td>0,33069</td>
      <td>0,17573</td>
      <td>0,26292</td>
      <td>0,05124</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0,33866</td>
      <td>0,25821</td>
      <td>0,30066</td>
      <td>0,16771</td>
      <td>0,26376</td>
      <td>0,04665</td>
    </tr>
  </tbody>
</table>
</div>



Dealing with timestamps


```python
def add_time(_df):
    "Returns a DF with two new cols : the time and hour of the day"
    t = pd.date_range(start='1/1/1986', periods=_df.shape[0], freq = 'H')
    t = pd.DataFrame(t)
    _df = pd.concat([_df, t], axis=1)
    _df.rename(columns={ _df.columns[-1]: "time" }, inplace = True)
    _df['hour'] = _df['time'].dt.hour
    _df['month'] = _df['time'].dt.month
    _df['week'] = _df['time'].dt.week
    return _df


for c in df_wind_on.columns:
    df_wind_on[c] = df_wind_on[c].str.replace(',', '.').astype('float64')

df_wind_on = add_time(df_wind_on)
df_wind_on.tail(2)
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
      <th>FI</th>
      <th>PT</th>
      <th>IT</th>
      <th>FR</th>
      <th>NO</th>
      <th>RO</th>
      <th>time</th>
      <th>hour</th>
      <th>month</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>262966</td>
      <td>0.47398</td>
      <td>0.08945</td>
      <td>0.00625</td>
      <td>0.21519</td>
      <td>0.54109</td>
      <td>0.11247</td>
      <td>2015-12-31 22:00:00</td>
      <td>22</td>
      <td>12</td>
      <td>53</td>
    </tr>
    <tr>
      <td>262967</td>
      <td>0.47473</td>
      <td>0.10206</td>
      <td>0.00859</td>
      <td>0.17319</td>
      <td>0.54552</td>
      <td>0.12690</td>
      <td>2015-12-31 23:00:00</td>
      <td>23</td>
      <td>12</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wind_on.dtypes
```




    FI              float64
    PT              float64
    IT              float64
    FR              float64
    NO              float64
    RO              float64
    time     datetime64[ns]
    hour              int64
    month             int64
    week              int64
    dtype: object



# Data Analysis

First, let's take a look at the last 24 hours, the least we can say is that there isn't any similarities or pattern in the graph below :


```python
def plot_hourly(df, title):
    plt.figure(figsize=(14, 9))
    for c in df.columns:
        if c != 'hour':
            sns.lineplot(x="hour", y=c, data=df, label=c)
            #plt.legend(c)
    plt.title(title)
    plt.show()

plot_hourly(df_wind_on[df_wind_on.columns.difference(['time', 'month', 'week'])][-24:], "Efficiency of solar stations per country during the 24 hours")
```


    
![png](/images/2021-04-04-wind-energy-eda/output_7_0.png)



Then let's compare the efficiency of onshore wind stations for each profile during hours of a typical day (based on the data of the last 30 years)


```python
plot_hourly(df_wind_on[df_wind_on.columns.difference(['time', 'month', 'week'])], "Efficiency of onshore wind stations per country during hours of a typical day")
```


    
![png](/images/2021-04-04-wind-energy-eda/output_9_0.png)
    


When we look at the distribution of the station's efficiencies, there isn't many hours in a day the installation produces actually energy


```python
temp_df = df_wind_on[df_wind_on.columns.difference(['time', 'hour', 'month', 'week'])]
plt.figure(figsize=(14, 9))
for col in temp_df.columns:
    sns.distplot(temp_df[temp_df[col] != 0][col], label=col, hist=False)
plt.title("Distribution of the station's efficiency for non null values (ie during the day)")
```




    Text(0.5, 1.0, "Distribution of the station's efficiency for non null values (ie during the day)")




    
![png](/images/2021-04-04-wind-energy-eda/output_11_1.png)
    


Anyway there are many short spikes each year


```python
plt.figure(figsize=(14, 9))
sns.lineplot(x = df_wind_on.time, y = df_wind_on['FR'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd833bdd350>




    
![png](/images/2021-04-04-wind-energy-eda/output_13_1.png)
    


A more interesting insight can be obtained when the efficiency is plotted across the months. It seems that at the opposite of the solar energy generation, the wind produces more in winter


```python
plt.figure(figsize=(12, 6))
for c in countries_lst:
    temp_df = df_wind_on[[c, 'month']]
    sns.lineplot(x=temp_df["month"], y=temp_df[c], label=c)
plt.xlabel("Month of year")
plt.ylabel("Efficiency") 
plt.title("Efficiency across the months per country")
```




    Text(0.5, 1.0, 'Efficiency across the months per country')




    
![png](/images/2021-04-04-wind-energy-eda/output_15_1.png)
    


There are more variations at the week level, but the same conclusion can be made :


```python
plt.figure(figsize=(12, 6))
for c in countries_lst:
    temp_df = df_wind_on[[c, 'week']]
    sns.lineplot(x=temp_df["week"], y=temp_df[c], label=c)
plt.xlabel("Week of year")
plt.ylabel("Efficiency") 
plt.title("Efficiency across the weeks per country")
```




    Text(0.5, 1.0, 'Efficiency across the weeks per country')




    
![png](/images/2021-04-04-wind-energy-eda/output_17_1.png)
    



```python
temp_df = df_wind_on.copy()
temp_df['year'] = temp_df['time'].dt.year
temp_df.head()
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
      <th>FI</th>
      <th>PT</th>
      <th>IT</th>
      <th>FR</th>
      <th>NO</th>
      <th>RO</th>
      <th>time</th>
      <th>hour</th>
      <th>month</th>
      <th>week</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.31303</td>
      <td>0.22683</td>
      <td>0.33069</td>
      <td>0.17573</td>
      <td>0.26292</td>
      <td>0.05124</td>
      <td>1986-01-01 00:00:00</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1986</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.33866</td>
      <td>0.25821</td>
      <td>0.30066</td>
      <td>0.16771</td>
      <td>0.26376</td>
      <td>0.04665</td>
      <td>1986-01-01 01:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1986</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.36834</td>
      <td>0.27921</td>
      <td>0.27052</td>
      <td>0.15877</td>
      <td>0.26695</td>
      <td>0.04543</td>
      <td>1986-01-01 02:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1986</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.39019</td>
      <td>0.33106</td>
      <td>0.24614</td>
      <td>0.14818</td>
      <td>0.27101</td>
      <td>0.04455</td>
      <td>1986-01-01 03:00:00</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1986</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.40209</td>
      <td>0.38668</td>
      <td>0.21655</td>
      <td>0.13631</td>
      <td>0.28097</td>
      <td>0.05438</td>
      <td>1986-01-01 04:00:00</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1986</td>
    </tr>
  </tbody>
</table>
</div>



When we compare the energy production yearly, the average performance is rather poor and can change a lot one year to an other :


```python
plt.figure(figsize=(12, 6))
for c in countries_lst:
    temp_df_ = temp_df[[c, 'year']]
    sns.lineplot(x=temp_df_["year"], y=temp_df_[c], label=c)
plt.xlabel("Year")
plt.ylabel("Efficiency") 
plt.title("Efficiency across the years per country")
```




    Text(0.5, 1.0, 'Efficiency across the years per country')




    
![png](/images/2021-04-04-wind-energy-eda/output_20_1.png)
    



```python
temp_df = temp_df.drop(columns=['time', 'hour', 'month', 'week', 'year'])
temp_df.describe()
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
      <th>FI</th>
      <th>PT</th>
      <th>IT</th>
      <th>FR</th>
      <th>NO</th>
      <th>RO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>262968.000000</td>
      <td>262968.000000</td>
      <td>262968.000000</td>
      <td>262968.000000</td>
      <td>262968.000000</td>
      <td>262968.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.195797</td>
      <td>0.231411</td>
      <td>0.168171</td>
      <td>0.235092</td>
      <td>0.234455</td>
      <td>0.206581</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.158490</td>
      <td>0.200070</td>
      <td>0.174528</td>
      <td>0.197681</td>
      <td>0.138302</td>
      <td>0.199103</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>0.000070</td>
      <td>0.000040</td>
      <td>0.000070</td>
      <td>0.000260</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.062760</td>
      <td>0.080390</td>
      <td>0.036400</td>
      <td>0.082140</td>
      <td>0.119130</td>
      <td>0.058230</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.153390</td>
      <td>0.164075</td>
      <td>0.101570</td>
      <td>0.170620</td>
      <td>0.214960</td>
      <td>0.136680</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.298873</td>
      <td>0.323300</td>
      <td>0.246170</td>
      <td>0.337360</td>
      <td>0.339190</td>
      <td>0.291813</td>
    </tr>
    <tr>
      <td>max</td>
      <td>0.634540</td>
      <td>0.959080</td>
      <td>0.870380</td>
      <td>0.845160</td>
      <td>0.579430</td>
      <td>0.992050</td>
    </tr>
  </tbody>
</table>
</div>



This is confirmed by the bar plot below :


```python
def plot_by_country(_df, title, nb_col):
    _df = _df.describe().iloc[nb_col, :]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=_df.index, y=_df.values)
    plt.title(title)

#plot_by_country("Mean efficiency by country", 1)

plot_by_country(temp_df, "75% efficiency by country", 6)
```


    
![png](/images/2021-04-04-wind-energy-eda/output_23_0.png)
    


An other interesting way is to use violin plots :


```python
# credits : https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn
# This works because pd.melt converts a wide-form dataframe
plt.figure(figsize=(10, 6))
sns.violinplot(x="variable", y="value", data=pd.melt(temp_df))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd840723890>

    
![png](/images/2021-04-04-wind-energy-eda/output_25_1.png)
    

## Correlations

Further more there is no real correlations we could find :


```python
def plot_corr(df_):
    corr = df_.corr()
    corr

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 18))

    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, center=0, square=True, cmap='Spectral', linewidths=.5, cbar_kws={"shrink": .5}) #annot=True

plot_corr(temp_df)
```


    
![png](/images/2021-04-04-wind-energy-eda/output_27_0.png)
    



```python
temp_df.corr()
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
      <th>FI</th>
      <th>PT</th>
      <th>IT</th>
      <th>FR</th>
      <th>NO</th>
      <th>RO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FI</td>
      <td>1.000000</td>
      <td>0.055113</td>
      <td>0.087366</td>
      <td>0.147905</td>
      <td>0.530650</td>
      <td>0.120007</td>
    </tr>
    <tr>
      <td>PT</td>
      <td>0.055113</td>
      <td>1.000000</td>
      <td>0.139590</td>
      <td>0.222474</td>
      <td>0.077719</td>
      <td>0.050059</td>
    </tr>
    <tr>
      <td>IT</td>
      <td>0.087366</td>
      <td>0.139590</td>
      <td>1.000000</td>
      <td>0.209693</td>
      <td>0.134060</td>
      <td>0.286878</td>
    </tr>
    <tr>
      <td>FR</td>
      <td>0.147905</td>
      <td>0.222474</td>
      <td>0.209693</td>
      <td>1.000000</td>
      <td>0.208549</td>
      <td>0.085928</td>
    </tr>
    <tr>
      <td>NO</td>
      <td>0.530650</td>
      <td>0.077719</td>
      <td>0.134060</td>
      <td>0.208549</td>
      <td>1.000000</td>
      <td>0.168180</td>
    </tr>
    <tr>
      <td>RO</td>
      <td>0.120007</td>
      <td>0.050059</td>
      <td>0.286878</td>
      <td>0.085928</td>
      <td>0.168180</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Heatmap month vs hours 

Using the same heatmap used for the solar energy generation but for wind, we can guess that the efficiency differences are significant at the month level not at the hour... 


```python
# credits S Godinho @ https://www.kaggle.com/sgodinho/wind-energy-potential-prediction

df_wind_on['year'] = df_wind_on['time'].dt.year
plt.figure(figsize=(8, 6))
temp_df = df_wind_on[['FR', 'month', 'hour']]
temp_df = temp_df.groupby(['hour', 'month']).mean()
temp_df = temp_df.unstack('month').sort_index(ascending=False)
sns.heatmap(temp_df, vmin = 0.09, vmax = 0.29, cmap = 'plasma')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd83ac16a90>




    
![png](/images/2021-04-04-wind-energy-eda/output_30_1.png)
    


## Conclusion  

In this second part, we’ve explored the data set but we haven't found real pattern in order to assess the impact of meteorological and climate variability on the generation of wind power. Only the variation during the months of the year is significant. Being able to predict the efficiency of wind energy generation is a really complex issue. It could have been interesting to study it at the region / geographic level but i lack the time to achieve it :)
