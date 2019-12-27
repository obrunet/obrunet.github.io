---
title: "Historical consumption regression for electricity supply pricing - part 1/2"
date: 2019-11-15
categories:
  - Data Science
tags: [ENS Data Challenge, Data Analysis]
header:
  image: "/images/2019-11-15-Electricity-supply-pricing/federico-beccari-ahi73ZN5P0Y-unsplash.jpg"
excerpt: "This is the 1st part of a data science challenge organized by the E.N.S, i've made a long time ago. After a detailed introduction, i'll dive in an exploratory data analysis to understand how data is structured, how the weather (and particularly the temperature) could influence the electricity consumption. And the other correlations.
How cyclic time infos are, and how the consumption vary depending on the hour of the day, the day of the week and the week of the year"
mathjax: "true"
---

Photo by Federico Beccari on Unsplash 

This is the 1st part of a data science challenge organized by the E.N.S, i've made a long time ago. After a detailed introduction, i'll dive in an exploratory data analysis to understand how data is structured.

# Introduction

## Context

* Planete OUI offers green electricity supply with prices adapted to the consumption profiles of its clients. 
* The electricity prices are highly variable during time (depending on the market, the global consumption and its client needs).
* The consumption profile of an installation has to be appraised to compute the best estimation of supply tarifs 
    * Most sites have a consumption varying strongly with temperature because of electrical heating systems.
    * Except industrial installations because their consumption might be highly related their uses

The 1st objectives are to __analyze thermosensitivity uses and / or other factors (times series...)__ affecting consumption 

Based on:
* the potential client historical consumption data (i.e its profile)
* electricity prices 
* a given percentile used to cover supply costs for different scenarios
the 2nd objective of Planète OUI is to compute a distribution of supply costs in €/MWh.  

## Planète OUI needs

Extrapolation of one or several years consumption data rebuilt from a single year of measured data supplied by the client (in order to be Tcombined with electricity prices) and to get a larger data set of analysis.

## Data

* The client’s data is often incomplete and spread over a relatively short period.
* 3 datasets:
    * x_train : input data of the training set
    * y_train : output data of the training set
    * x_test : input data of the testing set
* Features:
    * "ID": Data point ID;
    * "timestamp": Complete timestamps with year, month, day and hour, in local time (CET and CEST);
    * "temp1", "temp2", "meannationaltemp": Local and mean national temperatures (°C);
    * "humidity1", "humidity2": Local relative humidities (%);
    * "loc1", "loc2" "locsecondary1", "locsecondary2", "locsecondary3": the coordinates of the studied and secondary sites, in decimal degrees and of the form (latitude, longitude).
    * "consumptionsecondary1", "consumptionsecondary2", "consumptionsecondary3": the consumption data of three secondary sites, whose correlations with studied sites may be of use (kWh). Indeed, the two studied sites and the three secondary sites are used for the same purposes; The output data of the model to be developed takes the following form:
        * "ID": Data point ID;
        * "consumption1", "consumption2": the consumption data of the two studied sites (kWh).

Relative humidities are provided with temperature data because they represent variables of importance for electricity consumption: humidity indeed strongly available for all. It has
influences thermal comfort. To replicate operational conditions, some temperature integrated BCM Energy’s
and humidity data points will be missing. The imputation method must be carefully perimeter in 2017.
considered. The "consumptionsecondaryi" variables are the consumption data of several sites with metering power higher than 250 kVA of the Planète OUIs portfolio.

This correlation of the various sites consumptions shall be studied to precise data
completion or interpolation. Timestamps may be expressed as month or day of year,
day of week and hours, to study the impact of annual, weekly and daily seasonalities.
Particular attention should be paid to national holidays processing.

## Challenge goal

Prediction of the consumption of two given sites during a year, based on measured data of other profiles.

---

# First data insight

## Basic infos


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
```


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 100)
```


```python
from sklearn.preprocessing import MinMaxScaler
import folium
```


```python
input_dir = os.path.join('..', 'input')
output_dir = os.path.join('..', 'output')
print(os.listdir(input_dir))
```

    ['input_training_ssnsrY0.csv', 'Screenshot from 2019-05-28 10-48-59.png', 'input_test_cdKcI0e.csv', 'vacances-scolaires.csv', 'jours_feries_seuls.csv', 'BCM_custom_metric.py', 'output_training_Uf11I9I.csv']



```python
# each id is unique so we can use this column as index
df = pd.read_csv(os.path.join(input_dir, 'input_training_ssnsrY0.csv'), index_col='ID')
df.iloc[23:26]
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
      <th>timestamp</th>
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>loc_1</th>
      <th>loc_2</th>
      <th>loc_secondary_1</th>
      <th>loc_secondary_2</th>
      <th>loc_secondary_3</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>23</th>
      <td>2016-11-01T23:00:00.0</td>
      <td>10.8</td>
      <td>NaN</td>
      <td>11.1</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>143</td>
      <td>67</td>
      <td>162</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2016-11-02T00:00:00.0</td>
      <td>10.8</td>
      <td>NaN</td>
      <td>11.1</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>144</td>
      <td>75</td>
      <td>161</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2016-11-02T01:00:00.0</td>
      <td>10.5</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>81.0</td>
      <td>NaN</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>141</td>
      <td>65</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_out = pd.read_csv(os.path.join('..', 'input', 'output_training_Uf11I9I.csv'), index_col='ID')
df_out.head()
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
      <th>consumption_1</th>
      <th>consumption_2</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101</td>
      <td>94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
      <td>96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape, df_out.shape
```




    ((8760, 14), (8760, 2))




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8760 entries, 0 to 8759
    Data columns (total 14 columns):
    timestamp                  8760 non-null object
    temp_1                     8589 non-null float64
    temp_2                     8429 non-null float64
    mean_national_temp         8760 non-null float64
    humidity_1                 8589 non-null float64
    humidity_2                 8428 non-null float64
    loc_1                      8760 non-null object
    loc_2                      8760 non-null object
    loc_secondary_1            8760 non-null object
    loc_secondary_2            8760 non-null object
    loc_secondary_3            8760 non-null object
    consumption_secondary_1    8760 non-null int64
    consumption_secondary_2    8760 non-null int64
    consumption_secondary_3    8760 non-null int64
    dtypes: float64(5), int64(3), object(6)
    memory usage: 1.0+ MB


## Missing or irrelevant values


```python
df.isnull().sum()
```




    timestamp                    0
    temp_1                     171
    temp_2                     331
    mean_national_temp           0
    humidity_1                 171
    humidity_2                 332
    loc_1                        0
    loc_2                        0
    loc_secondary_1              0
    loc_secondary_2              0
    loc_secondary_3              0
    consumption_secondary_1      0
    consumption_secondary_2      0
    consumption_secondary_3      0
    dtype: int64




```python
df.duplicated().sum()
```




    0




```python
#sns.pairplot(df_viz.select_dtypes(['int64', 'float64']))#.apply(pd.Series.nunique, axis=0)
```

## Transforming time informations 


```python
def transform_datetime_infos(data_frame):
    data_frame['datetime'] = pd.to_datetime(data_frame['timestamp'])
    data_frame['month'] = data_frame['datetime'].dt.month
    data_frame['week of year'] = data_frame['datetime'].dt.weekofyear
    data_frame['day of year'] = data_frame['datetime'].dt.dayofyear
    data_frame['day'] = data_frame['datetime'].dt.weekday_name
    data_frame['hour'] = data_frame['datetime'].dt.hour
    
    # for merging purposes
    data_frame['date'] = data_frame['datetime'].dt.strftime('%Y-%m-%d')
    
    return data_frame
```


```python
df = transform_datetime_infos(df)
df.head(3)
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
      <th>timestamp</th>
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>loc_1</th>
      <th>loc_2</th>
      <th>loc_secondary_1</th>
      <th>loc_secondary_2</th>
      <th>loc_secondary_3</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
      <th>datetime</th>
      <th>month</th>
      <th>week of year</th>
      <th>day of year</th>
      <th>day</th>
      <th>hour</th>
      <th>date</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>0</th>
      <td>2016-11-01T00:00:00.0</td>
      <td>8.3</td>
      <td>NaN</td>
      <td>11.1</td>
      <td>95.0</td>
      <td>NaN</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>143</td>
      <td>74</td>
      <td>168</td>
      <td>2016-11-01 00:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>0</td>
      <td>2016-11-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-11-01T01:00:00.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>11.1</td>
      <td>98.0</td>
      <td>NaN</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>141</td>
      <td>60</td>
      <td>162</td>
      <td>2016-11-01 01:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>1</td>
      <td>2016-11-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-11-01T02:00:00.0</td>
      <td>6.8</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>97.0</td>
      <td>NaN</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>142</td>
      <td>60</td>
      <td>164</td>
      <td>2016-11-01 02:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>2</td>
      <td>2016-11-01</td>
    </tr>
  </tbody>
</table>
</div>



## Considered sites


```python
# Number of unique classes in each object column # Check the nb of sites
df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
```




    timestamp          8759
    loc_1                 1
    loc_2                 1
    loc_secondary_1       1
    loc_secondary_2       1
    loc_secondary_3       1
    day                   7
    date                365
    dtype: int64




```python
for i in ['loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3']:
    print('site : ', i, 'coordinates : ', df[i][0])
```

    site :  loc_1 coordinates :  (50.633, 3.067)
    site :  loc_2 coordinates :  (43.530, 5.447)
    site :  loc_secondary_1 coordinates :  (44.838, -0.579)
    site :  loc_secondary_2 coordinates :  (47.478, -0.563)
    site :  loc_secondary_3 coordinates :  (48.867, 2.333)



```python
# Create a map centered at the given latitude and longitude
france_map = folium.Map(location=[47,1], zoom_start=6)

# Add markers with labels
for i in ['loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3']:
    temp_str = df[i][0].strip('(').strip(')').strip(' ')
    temp_str1, temp_str2 = temp_str.split(', ')
    folium.Marker([float(temp_str1), float(temp_str2)], popup=None, tooltip=i).add_to(france_map) 
    
# Display the map
display(france_map)
```


<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF9lZjUwYzBhNzE1MTU0Nzk4OWU3MGNmYWQzMjUzNGNmOCB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfZWY1MGMwYTcxNTE1NDc5ODllNzBjZmFkMzI1MzRjZjgiID48L2Rpdj4KICAgICAgICAKPC9ib2R5Pgo8c2NyaXB0PiAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFwX2VmNTBjMGE3MTUxNTQ3OTg5ZTcwY2ZhZDMyNTM0Y2Y4ID0gTC5tYXAoCiAgICAgICAgICAgICAgICAibWFwX2VmNTBjMGE3MTUxNTQ3OTg5ZTcwY2ZhZDMyNTM0Y2Y4IiwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBjZW50ZXI6IFs0Ny4wLCAxLjBdLAogICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcsCiAgICAgICAgICAgICAgICAgICAgem9vbTogNiwKICAgICAgICAgICAgICAgICAgICB6b29tQ29udHJvbDogdHJ1ZSwKICAgICAgICAgICAgICAgICAgICBwcmVmZXJDYW52YXM6IGZhbHNlLAogICAgICAgICAgICAgICAgfQogICAgICAgICAgICApOwoKICAgICAgICAgICAgCgogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzgyYjdkZmZkNjM4YTQ1YzNiZjQ2M2RlYTVmNzRlMjdmID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAiaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmciLAogICAgICAgICAgICAgICAgeyJhdHRyaWJ1dGlvbiI6ICJEYXRhIGJ5IFx1MDAyNmNvcHk7IFx1MDAzY2EgaHJlZj1cImh0dHA6Ly9vcGVuc3RyZWV0bWFwLm9yZ1wiXHUwMDNlT3BlblN0cmVldE1hcFx1MDAzYy9hXHUwMDNlLCB1bmRlciBcdTAwM2NhIGhyZWY9XCJodHRwOi8vd3d3Lm9wZW5zdHJlZXRtYXAub3JnL2NvcHlyaWdodFwiXHUwMDNlT0RiTFx1MDAzYy9hXHUwMDNlLiIsICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwgIm1heE5hdGl2ZVpvb20iOiAxOCwgIm1heFpvb20iOiAxOCwgIm1pblpvb20iOiAwLCAibm9XcmFwIjogZmFsc2UsICJvcGFjaXR5IjogMSwgInN1YmRvbWFpbnMiOiAiYWJjIiwgInRtcyI6IGZhbHNlfQogICAgICAgICAgICApLmFkZFRvKG1hcF9lZjUwYzBhNzE1MTU0Nzk4OWU3MGNmYWQzMjUzNGNmOCk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIG1hcmtlcl9jY2NmMmU4NjJkOWE0YjJiODZkZWY3NmRjZTVmNmYxMCA9IEwubWFya2VyKAogICAgICAgICAgICAgICAgWzUwLjYzMywgMy4wNjddLAogICAgICAgICAgICAgICAge30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWY1MGMwYTcxNTE1NDc5ODllNzBjZmFkMzI1MzRjZjgpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIG1hcmtlcl9jY2NmMmU4NjJkOWE0YjJiODZkZWY3NmRjZTVmNmYxMC5iaW5kVG9vbHRpcCgKICAgICAgICAgICAgICAgIGA8ZGl2PgogICAgICAgICAgICAgICAgICAgICBsb2NfMQogICAgICAgICAgICAgICAgIDwvZGl2PmAsCiAgICAgICAgICAgICAgICB7InN0aWNreSI6IHRydWV9CiAgICAgICAgICAgICk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIG1hcmtlcl9hYTlhYzlkODZmMjI0YzBkODNjNDRlNzE2ZmU0MDNiNCA9IEwubWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjUzLCA1LjQ0N10sCiAgICAgICAgICAgICAgICB7fQogICAgICAgICAgICApLmFkZFRvKG1hcF9lZjUwYzBhNzE1MTU0Nzk4OWU3MGNmYWQzMjUzNGNmOCk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgbWFya2VyX2FhOWFjOWQ4NmYyMjRjMGQ4M2M0NGU3MTZmZTQwM2I0LmJpbmRUb29sdGlwKAogICAgICAgICAgICAgICAgYDxkaXY+CiAgICAgICAgICAgICAgICAgICAgIGxvY18yCiAgICAgICAgICAgICAgICAgPC9kaXY+YCwKICAgICAgICAgICAgICAgIHsic3RpY2t5IjogdHJ1ZX0KICAgICAgICAgICAgKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFya2VyXzQ2ZmJkMzQ5MmZiNzRkNzk5MTdjMmVkNmYyZjM5YWEwID0gTC5tYXJrZXIoCiAgICAgICAgICAgICAgICBbNDQuODM4LCAtMC41NzldLAogICAgICAgICAgICAgICAge30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWY1MGMwYTcxNTE1NDc5ODllNzBjZmFkMzI1MzRjZjgpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIG1hcmtlcl80NmZiZDM0OTJmYjc0ZDc5OTE3YzJlZDZmMmYzOWFhMC5iaW5kVG9vbHRpcCgKICAgICAgICAgICAgICAgIGA8ZGl2PgogICAgICAgICAgICAgICAgICAgICBsb2Nfc2Vjb25kYXJ5XzEKICAgICAgICAgICAgICAgICA8L2Rpdj5gLAogICAgICAgICAgICAgICAgeyJzdGlja3kiOiB0cnVlfQogICAgICAgICAgICApOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBtYXJrZXJfMDkwMjIwOWMzZjM3NGFjMWI3OWMyNDhhNjEyMDNlOGUgPSBMLm1hcmtlcigKICAgICAgICAgICAgICAgIFs0Ny40NzgsIC0wLjU2M10sCiAgICAgICAgICAgICAgICB7fQogICAgICAgICAgICApLmFkZFRvKG1hcF9lZjUwYzBhNzE1MTU0Nzk4OWU3MGNmYWQzMjUzNGNmOCk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgbWFya2VyXzA5MDIyMDljM2YzNzRhYzFiNzljMjQ4YTYxMjAzZThlLmJpbmRUb29sdGlwKAogICAgICAgICAgICAgICAgYDxkaXY+CiAgICAgICAgICAgICAgICAgICAgIGxvY19zZWNvbmRhcnlfMgogICAgICAgICAgICAgICAgIDwvZGl2PmAsCiAgICAgICAgICAgICAgICB7InN0aWNreSI6IHRydWV9CiAgICAgICAgICAgICk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIG1hcmtlcl84ZDQwZGYwMGUyZjk0ZjkxOWVhMjYyYjY3MzkxYmEzZiA9IEwubWFya2VyKAogICAgICAgICAgICAgICAgWzQ4Ljg2NywgMi4zMzNdLAogICAgICAgICAgICAgICAge30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWY1MGMwYTcxNTE1NDc5ODllNzBjZmFkMzI1MzRjZjgpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIG1hcmtlcl84ZDQwZGYwMGUyZjk0ZjkxOWVhMjYyYjY3MzkxYmEzZi5iaW5kVG9vbHRpcCgKICAgICAgICAgICAgICAgIGA8ZGl2PgogICAgICAgICAgICAgICAgICAgICBsb2Nfc2Vjb25kYXJ5XzMKICAgICAgICAgICAgICAgICA8L2Rpdj5gLAogICAgICAgICAgICAgICAgeyJzdGlja3kiOiB0cnVlfQogICAgICAgICAgICApOwogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>

![png](/images/2019-11-15-Electricity-supply-pricing/Screenshot_folium.png)

* loc_1 is in the north near Lille.
* loc_1 is in the south east near Marseille.
* loc_secondary_1 is in the south west near Bordeaux.
* loc_secondary_2 is in the west near Le Mans.
* loc_secondary_3 is in the north near Paris.

---

# Exploratory Data Analysis

## Distribution of consumption


```python
plt.figure(figsize=(16, 8))
sns.kdeplot(df_out['consumption_1'], shade=True) #, label = 'consumption_1')
sns.kdeplot(df_out['consumption_2'], shade=True) #, label = 'consumption_2')
sns.kdeplot(df['consumption_secondary_1'], shade=True) #, label = 'consumption_1')
sns.kdeplot(df['consumption_secondary_2'], shade=True) #, label = 'consumption_1')
sns.kdeplot(df['consumption_secondary_3'], shade=True) #, label = 'consumption_1')
plt.show()
```

![png](/images/2019-11-15-Electricity-supply-pricing/output_27_0.png)


## Consumption variation during time


```python
df.groupby(['hour'])['consumption_secondary_1'].mean().values
```




    array([160.77534247, 160.8739726 , 160.5260274 , 160.83835616,
           161.46027397, 168.88767123, 191.30136986, 210.9369863 ,
           228.49863014, 248.32328767, 256.64657534, 256.21369863,
           240.23287671, 238.76438356, 244.09041096, 242.99178082,
           237.61643836, 227.66027397, 208.08767123, 195.21917808,
           182.35068493, 166.50958904, 161.92054795, 161.87671233])




```python
df_viz = pd.concat((df, df_out), axis=1)
plt.figure(figsize=(16, 8))
plt.title("Consumption Evolution for each site over hours of the day")
for c in ['consumption_1', 'consumption_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']:
    sns.lineplot(x="hour", y=c, data=df_viz, label=c)
plt.show()
```


![png](/images/2019-11-15-Electricity-supply-pricing/output_30_0.png)


The graph above clearly shows that the electricity consumption is higher during working hours of the day.


```python
plt.figure(figsize=(16, 8))
plt.title("Consumption Evolution for each site over weekday")
for c in ['consumption_1', 'consumption_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']:
    sns.lineplot(x="day", y=c, data=df_viz, label=c)
plt.show()
```


![png](/images/2019-11-15-Electricity-supply-pricing/output_32_0.png)


The consumption is lower during the week-end. So we can deduce that those sites are not housings.


```python
plt.figure(figsize=(16, 8))
plt.title("Consumption Evolution for each site over a year")
for c in ['consumption_1', 'consumption_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']:
    sns.lineplot(x="day of year", y=c, data=df_viz, label=c)
plt.show()
```


![png](/images/2019-11-15-Electricity-supply-pricing/output_34_0.png)


We can see that globally the mean consumption decreases until the summer, then increases until the end of the year. 


```python
plt.figure(figsize=(16, 8))
plt.title("Consumption Evolution for each site over a year during WORKING DAYS")
for c in ['consumption_1', 'consumption_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']:
    sns.lineplot(x="day of year", y=c, data=df_viz[~df_viz['day'].isin(['Saturday', 'Sunday'])], label=c)
plt.show()
```


![png](/images/2019-11-15-Electricity-supply-pricing/output_36_0.png)



```python
plt.figure(figsize=(16, 8))
plt.title("Mean Consumption Evolution for each site over months")
for c in ['consumption_1', 'consumption_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']:
    sns.lineplot(data=df_viz.groupby(['month'])[c].mean(), label=c)
plt.show()
```


![png](/images/2019-11-15-Electricity-supply-pricing/output_37_0.png)



```python
plt.figure(figsize=(16, 8))
plt.title("Mean Consumption Evolution for each site over weeks")
for c in ['consumption_1', 'consumption_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']:
    sns.lineplot(data=df_viz.groupby(['week of year'])[c].mean(), label=c)
plt.show()
```


![png](/images/2019-11-15-Electricity-supply-pricing/output_38_0.png)



```python
temp_consumption_cols = ['consumption_1', 'consumption_2', 'temp_1', 'temp_2']
df_viz[temp_consumption_cols] = MinMaxScaler().fit_transform(df_viz[temp_consumption_cols])

plt.figure(figsize=(16, 8))
plt.title("Correlation between Mean Consumption & Mean Temperatures / for each site over months")
for c in temp_consumption_cols:
    sns.lineplot(data=df_viz.groupby(['month'])[c].mean(), label=c)
plt.show()
```

    /home/sunflowa/Anaconda/lib/python3.7/site-packages/sklearn/preprocessing/data.py:334: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)



![png](/images/2019-11-15-Electricity-supply-pricing/output_39_1.png)


Electricity consumption and temperatures seem to be negatively correlated.

---

# Features Engineering

## Special dates - non working days "Jours fériés" 
(credits: Antoine Augusti https://github.com/AntoineAugusti/jours-feries-france)


```python
jf = pd.read_csv(os.path.join(input_dir, 'jours_feries_seuls.csv')).drop(columns=['nom_jour_ferie'])
# the date column is kept as a string for merging purposes
jf['est_jour_ferie'] = jf['est_jour_ferie'].astype('int')
jf.tail()
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
      <th>date</th>
      <th>est_jour_ferie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1102</th>
      <td>2050-07-14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>2050-08-15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>2050-11-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>2050-11-11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>2050-12-25</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
jf.est_jour_ferie.unique()
```




    array([1])



## Holidays 
(credits: Antoine Augusti https://www.data.gouv.fr/fr/datasets/vacances-scolaires-par-zones/)


```python
holidays = pd.read_csv(os.path.join(input_dir, 'vacances-scolaires.csv')).drop(columns=['nom_vacances'])
# the date column is kept as a string for merging purposes
for col in ['vacances_zone_a', 'vacances_zone_b', 'vacances_zone_c']:
    holidays[col] = holidays[col].astype('int')
holidays.tail()
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
      <th>date</th>
      <th>vacances_zone_a</th>
      <th>vacances_zone_b</th>
      <th>vacances_zone_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11318</th>
      <td>2020-12-27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11319</th>
      <td>2020-12-28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11320</th>
      <td>2020-12-29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11321</th>
      <td>2020-12-30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11322</th>
      <td>2020-12-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Sunlight hours
work in progress, more to come in the next days...stay tuned :)

## Merging all infos


```python
def merge_infos(data_frame):
    data_frame = pd.merge(data_frame, holidays, on='date', how='left')
    data_frame = pd.merge(data_frame, jf, on='date', how='left')
    return data_frame
```


```python
df = merge_infos(df)
df.vacances_zone_a.value_counts()/24
```




    0    243.958333
    1    121.041667
    Name: vacances_zone_a, dtype: float64




```python
df.est_jour_ferie.value_counts()/24
```




    1.0    11.0
    Name: est_jour_ferie, dtype: float64



# Cleaning Data


```python
def cleaning_data(data_frame):

    # The Nan values of the column "est_jour_ferie" correspond to working days 
    # because in the dataset merged with, there is only non working days 
    data_frame['est_jour_ferie'] = data_frame['est_jour_ferie'].fillna(0)
    
    # At first, missing values in the temperatures and humidity columns are replaced by the median ones
    # later another approach with open data of means of the corresponding months will be used
    for c in ['temp_1', 'temp_2', 'humidity_1', 'humidity_2']:
        data_frame[c] = data_frame[c].fillna(data_frame[c].median())
        
    return data_frame
```


```python
df = cleaning_data(df)
df.isnull().sum()
```




    timestamp                  0
    temp_1                     0
    temp_2                     0
    mean_national_temp         0
    humidity_1                 0
    humidity_2                 0
    loc_1                      0
    loc_2                      0
    loc_secondary_1            0
    loc_secondary_2            0
    loc_secondary_3            0
    consumption_secondary_1    0
    consumption_secondary_2    0
    consumption_secondary_3    0
    datetime                   0
    month                      0
    week of year               0
    day of year                0
    day                        0
    hour                       0
    date                       0
    vacances_zone_a            0
    vacances_zone_b            0
    vacances_zone_c            0
    est_jour_ferie             0
    dtype: int64




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
      <th>timestamp</th>
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>loc_1</th>
      <th>loc_2</th>
      <th>loc_secondary_1</th>
      <th>loc_secondary_2</th>
      <th>loc_secondary_3</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
      <th>datetime</th>
      <th>month</th>
      <th>week of year</th>
      <th>day of year</th>
      <th>day</th>
      <th>hour</th>
      <th>date</th>
      <th>vacances_zone_a</th>
      <th>vacances_zone_b</th>
      <th>vacances_zone_c</th>
      <th>est_jour_ferie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-11-01T00:00:00.0</td>
      <td>8.3</td>
      <td>14.5</td>
      <td>11.1</td>
      <td>95.0</td>
      <td>65.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>143</td>
      <td>74</td>
      <td>168</td>
      <td>2016-11-01 00:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>0</td>
      <td>2016-11-01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-11-01T01:00:00.0</td>
      <td>8.0</td>
      <td>14.5</td>
      <td>11.1</td>
      <td>98.0</td>
      <td>65.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>141</td>
      <td>60</td>
      <td>162</td>
      <td>2016-11-01 01:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>1</td>
      <td>2016-11-01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-11-01T02:00:00.0</td>
      <td>6.8</td>
      <td>14.5</td>
      <td>11.0</td>
      <td>97.0</td>
      <td>65.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>142</td>
      <td>60</td>
      <td>164</td>
      <td>2016-11-01 02:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>2</td>
      <td>2016-11-01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-11-01T03:00:00.0</td>
      <td>7.5</td>
      <td>14.5</td>
      <td>10.9</td>
      <td>99.0</td>
      <td>65.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>139</td>
      <td>60</td>
      <td>162</td>
      <td>2016-11-01 03:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>3</td>
      <td>2016-11-01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-11-01T04:00:00.0</td>
      <td>6.1</td>
      <td>14.5</td>
      <td>10.8</td>
      <td>98.0</td>
      <td>65.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>154</td>
      <td>60</td>
      <td>164</td>
      <td>2016-11-01 04:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>4</td>
      <td>2016-11-01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



---

# Correlations


```python
# after dummies
# sns.pairplot(df_viz.select_dtypes(['int64', 'float64']))#.apply(pd.Series.nunique, axis=0)
```


```python
df_viz = pd.concat((df, df_out), axis=1)
corr = df_viz.corr()

# makes all correlations positive for the heatmap
corr = np.sqrt(corr * corr)
```


```python
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
plt.title('Correlations - beware all are made positive, in reality some are negative ones')
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fce98470f28>




![png](/images/2019-11-15-Electricity-supply-pricing/output_61_1.png)


There medium correlations between weather informations. The given sites have also positively correlated consumption, this makes sense because all sites are housings. The targets seem to be weakly correlated with time infos...

---

## Comparison between X_train & X_test


```python
X_test = pd.read_csv(os.path.join(input_dir, 'input_test_cdKcI0e.csv'), index_col='ID')
X_test.iloc[23:26]
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
      <th>timestamp</th>
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>loc_1</th>
      <th>loc_2</th>
      <th>loc_secondary_1</th>
      <th>loc_secondary_2</th>
      <th>loc_secondary_3</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>8783</th>
      <td>2017-11-01T23:00:00.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.4</td>
      <td>78.0</td>
      <td>95.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>172</td>
      <td>131</td>
      <td>170</td>
    </tr>
    <tr>
      <th>8784</th>
      <td>2017-11-02T00:00:00.0</td>
      <td>8.8</td>
      <td>8.6</td>
      <td>9.4</td>
      <td>79.0</td>
      <td>95.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>172</td>
      <td>136</td>
      <td>172</td>
    </tr>
    <tr>
      <th>8785</th>
      <td>2017-11-02T01:00:00.0</td>
      <td>7.6</td>
      <td>8.8</td>
      <td>9.4</td>
      <td>82.0</td>
      <td>96.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>174</td>
      <td>126</td>
      <td>171</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape, X_test.shape
```




    ((8760, 25), (8736, 14))



There are 24 lines (hours) more wich correspond to one day, because 2016 was a leap year


```python
X_test.iloc[[0, -1]]
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
      <th>timestamp</th>
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>loc_1</th>
      <th>loc_2</th>
      <th>loc_secondary_1</th>
      <th>loc_secondary_2</th>
      <th>loc_secondary_3</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>8760</th>
      <td>2017-11-01T00:00:00.0</td>
      <td>6.5</td>
      <td>7.1</td>
      <td>8.8</td>
      <td>91.0</td>
      <td>82.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>190</td>
      <td>126</td>
      <td>177</td>
    </tr>
    <tr>
      <th>17495</th>
      <td>2018-10-30T23:00:00.0</td>
      <td>7.5</td>
      <td>11.2</td>
      <td>6.7</td>
      <td>85.0</td>
      <td>77.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>228</td>
      <td>114</td>
      <td>178</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[[0, -1]]
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
      <th>timestamp</th>
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>loc_1</th>
      <th>loc_2</th>
      <th>loc_secondary_1</th>
      <th>loc_secondary_2</th>
      <th>loc_secondary_3</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
      <th>datetime</th>
      <th>month</th>
      <th>week of year</th>
      <th>day of year</th>
      <th>day</th>
      <th>hour</th>
      <th>date</th>
      <th>vacances_zone_a</th>
      <th>vacances_zone_b</th>
      <th>vacances_zone_c</th>
      <th>est_jour_ferie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-11-01T00:00:00.0</td>
      <td>8.3</td>
      <td>14.5</td>
      <td>11.1</td>
      <td>95.0</td>
      <td>65.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>143</td>
      <td>74</td>
      <td>168</td>
      <td>2016-11-01 00:00:00</td>
      <td>11</td>
      <td>44</td>
      <td>306</td>
      <td>Tuesday</td>
      <td>0</td>
      <td>2016-11-01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8759</th>
      <td>2017-10-31T23:00:00.0</td>
      <td>7.0</td>
      <td>7.9</td>
      <td>8.8</td>
      <td>90.0</td>
      <td>78.0</td>
      <td>(50.633, 3.067)</td>
      <td>(43.530, 5.447)</td>
      <td>(44.838, -0.579)</td>
      <td>(47.478, -0.563)</td>
      <td>(48.867, 2.333)</td>
      <td>198</td>
      <td>128</td>
      <td>189</td>
      <td>2017-10-31 23:00:00</td>
      <td>10</td>
      <td>44</td>
      <td>304</td>
      <td>Tuesday</td>
      <td>23</td>
      <td>2017-10-31</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



There is a difference in the length of the two dataframes X_train & X_test

---

# Data preparation


```python
feat_to_drop = [
'timestamp',
'loc_1',
'loc_2',
'loc_secondary_1',
'loc_secondary_2',
'loc_secondary_3',
'datetime',
'date']
```


```python
feat_to_scale = [
'temp_1',
'temp_2',
'mean_national_temp',
'humidity_1',
'humidity_2',
'consumption_secondary_1',
'consumption_secondary_2',
'consumption_secondary_3']
```


```python
feat_to_dummies = [
'day',
'month',
'week of year',
#'day of year',
'hour']
```

Side note : let's try to run models without dummification of day of year


```python
def prepare_feat_df(data_frame):
    data_frame = data_frame.drop(columns=feat_to_drop)
    data_frame[feat_to_scale] = MinMaxScaler().fit_transform(data_frame[feat_to_scale])
    data_frame = pd.get_dummies(data=data_frame, columns=feat_to_dummies, drop_first=True)
    return data_frame
```


```python
df = prepare_feat_df(df)
df.head(2)
```

    /home/sunflowa/Anaconda/lib/python3.7/site-packages/sklearn/preprocessing/data.py:334: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)





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
      <th>temp_1</th>
      <th>temp_2</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>humidity_2</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
      <th>day of year</th>
      <th>vacances_zone_a</th>
      <th>vacances_zone_b</th>
      <th>vacances_zone_c</th>
      <th>est_jour_ferie</th>
      <th>day_Monday</th>
      <th>day_Saturday</th>
      <th>day_Sunday</th>
      <th>day_Thursday</th>
      <th>day_Tuesday</th>
      <th>day_Wednesday</th>
      <th>month_2</th>
      <th>month_3</th>
      <th>month_4</th>
      <th>month_5</th>
      <th>month_6</th>
      <th>month_7</th>
      <th>month_8</th>
      <th>month_9</th>
      <th>month_10</th>
      <th>month_11</th>
      <th>month_12</th>
      <th>week of year_2</th>
      <th>week of year_3</th>
      <th>week of year_4</th>
      <th>week of year_5</th>
      <th>week of year_6</th>
      <th>week of year_7</th>
      <th>week of year_8</th>
      <th>week of year_9</th>
      <th>week of year_10</th>
      <th>week of year_11</th>
      <th>week of year_12</th>
      <th>week of year_13</th>
      <th>week of year_14</th>
      <th>week of year_15</th>
      <th>week of year_16</th>
      <th>week of year_17</th>
      <th>week of year_18</th>
      <th>week of year_19</th>
      <th>week of year_20</th>
      <th>week of year_21</th>
      <th>...</th>
      <th>week of year_26</th>
      <th>week of year_27</th>
      <th>week of year_28</th>
      <th>week of year_29</th>
      <th>week of year_30</th>
      <th>week of year_31</th>
      <th>week of year_32</th>
      <th>week of year_33</th>
      <th>week of year_34</th>
      <th>week of year_35</th>
      <th>week of year_36</th>
      <th>week of year_37</th>
      <th>week of year_38</th>
      <th>week of year_39</th>
      <th>week of year_40</th>
      <th>week of year_41</th>
      <th>week of year_42</th>
      <th>week of year_43</th>
      <th>week of year_44</th>
      <th>week of year_45</th>
      <th>week of year_46</th>
      <th>week of year_47</th>
      <th>week of year_48</th>
      <th>week of year_49</th>
      <th>week of year_50</th>
      <th>week of year_51</th>
      <th>week of year_52</th>
      <th>hour_1</th>
      <th>hour_2</th>
      <th>hour_3</th>
      <th>hour_4</th>
      <th>hour_5</th>
      <th>hour_6</th>
      <th>hour_7</th>
      <th>hour_8</th>
      <th>hour_9</th>
      <th>hour_10</th>
      <th>hour_11</th>
      <th>hour_12</th>
      <th>hour_13</th>
      <th>hour_14</th>
      <th>hour_15</th>
      <th>hour_16</th>
      <th>hour_17</th>
      <th>hour_18</th>
      <th>hour_19</th>
      <th>hour_20</th>
      <th>hour_21</th>
      <th>hour_22</th>
      <th>hour_23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.356234</td>
      <td>0.466667</td>
      <td>0.428571</td>
      <td>0.936709</td>
      <td>0.609195</td>
      <td>0.155263</td>
      <td>0.208451</td>
      <td>0.155462</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>0.348601</td>
      <td>0.466667</td>
      <td>0.428571</td>
      <td>0.974684</td>
      <td>0.609195</td>
      <td>0.150000</td>
      <td>0.169014</td>
      <td>0.142857</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
  </tbody>
</table>
<p>2 rows × 104 columns</p>
</div>



## Data Preparation


```python
def prepare_data(input_file_name, output_file_name1, output_file_name2):
    data_frame = pd.read_csv(os.path.join(input_dir, input_file_name), index_col='ID')
    
    data_frame = transform_datetime_infos(data_frame)
    data_frame = merge_infos(data_frame)
    data_frame = cleaning_data(data_frame)
    data_frame = prepare_feat_df(data_frame)
    
    df_1 = data_frame.drop(columns=['temp_2', 'humidity_2'])
    df_2 = data_frame.drop(columns=['temp_1', 'humidity_1'])
    
    df_1.to_csv(os.path.join(output_dir, output_file_name1), index=False)
    df_2.to_csv(os.path.join(output_dir, output_file_name2), index=False)
    
    #return df_1, df_2
```


```python
prepare_data('input_training_ssnsrY0.csv', 'X_train_1.csv', 'X_train_2.csv')
prepare_data('input_test_cdKcI0e.csv', 'X_test_1.csv', 'X_test_2.csv')
```

    /home/sunflowa/Anaconda/lib/python3.7/site-packages/sklearn/preprocessing/data.py:334: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)
    /home/sunflowa/Anaconda/lib/python3.7/site-packages/sklearn/preprocessing/data.py:334: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)



```python
pd.read_csv(os.path.join(output_dir, 'X_train_1.csv')).head()
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
      <th>temp_1</th>
      <th>mean_national_temp</th>
      <th>humidity_1</th>
      <th>consumption_secondary_1</th>
      <th>consumption_secondary_2</th>
      <th>consumption_secondary_3</th>
      <th>day of year</th>
      <th>vacances_zone_a</th>
      <th>vacances_zone_b</th>
      <th>vacances_zone_c</th>
      <th>est_jour_ferie</th>
      <th>day_Monday</th>
      <th>day_Saturday</th>
      <th>day_Sunday</th>
      <th>day_Thursday</th>
      <th>day_Tuesday</th>
      <th>day_Wednesday</th>
      <th>month_2</th>
      <th>month_3</th>
      <th>month_4</th>
      <th>month_5</th>
      <th>month_6</th>
      <th>month_7</th>
      <th>month_8</th>
      <th>month_9</th>
      <th>month_10</th>
      <th>month_11</th>
      <th>month_12</th>
      <th>week of year_2</th>
      <th>week of year_3</th>
      <th>week of year_4</th>
      <th>week of year_5</th>
      <th>week of year_6</th>
      <th>week of year_7</th>
      <th>week of year_8</th>
      <th>week of year_9</th>
      <th>week of year_10</th>
      <th>week of year_11</th>
      <th>week of year_12</th>
      <th>week of year_13</th>
      <th>week of year_14</th>
      <th>week of year_15</th>
      <th>week of year_16</th>
      <th>week of year_17</th>
      <th>week of year_18</th>
      <th>week of year_19</th>
      <th>week of year_20</th>
      <th>week of year_21</th>
      <th>week of year_22</th>
      <th>week of year_23</th>
      <th>...</th>
      <th>week of year_26</th>
      <th>week of year_27</th>
      <th>week of year_28</th>
      <th>week of year_29</th>
      <th>week of year_30</th>
      <th>week of year_31</th>
      <th>week of year_32</th>
      <th>week of year_33</th>
      <th>week of year_34</th>
      <th>week of year_35</th>
      <th>week of year_36</th>
      <th>week of year_37</th>
      <th>week of year_38</th>
      <th>week of year_39</th>
      <th>week of year_40</th>
      <th>week of year_41</th>
      <th>week of year_42</th>
      <th>week of year_43</th>
      <th>week of year_44</th>
      <th>week of year_45</th>
      <th>week of year_46</th>
      <th>week of year_47</th>
      <th>week of year_48</th>
      <th>week of year_49</th>
      <th>week of year_50</th>
      <th>week of year_51</th>
      <th>week of year_52</th>
      <th>hour_1</th>
      <th>hour_2</th>
      <th>hour_3</th>
      <th>hour_4</th>
      <th>hour_5</th>
      <th>hour_6</th>
      <th>hour_7</th>
      <th>hour_8</th>
      <th>hour_9</th>
      <th>hour_10</th>
      <th>hour_11</th>
      <th>hour_12</th>
      <th>hour_13</th>
      <th>hour_14</th>
      <th>hour_15</th>
      <th>hour_16</th>
      <th>hour_17</th>
      <th>hour_18</th>
      <th>hour_19</th>
      <th>hour_20</th>
      <th>hour_21</th>
      <th>hour_22</th>
      <th>hour_23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.356234</td>
      <td>0.428571</td>
      <td>0.936709</td>
      <td>0.155263</td>
      <td>0.208451</td>
      <td>0.155462</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>0.348601</td>
      <td>0.428571</td>
      <td>0.974684</td>
      <td>0.150000</td>
      <td>0.169014</td>
      <td>0.142857</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>0.318066</td>
      <td>0.425249</td>
      <td>0.962025</td>
      <td>0.152632</td>
      <td>0.169014</td>
      <td>0.147059</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>0.335878</td>
      <td>0.421927</td>
      <td>0.987342</td>
      <td>0.144737</td>
      <td>0.169014</td>
      <td>0.142857</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    <tr>
      <th>4</th>
      <td>0.300254</td>
      <td>0.418605</td>
      <td>0.974684</td>
      <td>0.184211</td>
      <td>0.169014</td>
      <td>0.147059</td>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 102 columns</p>
</div>




```python
df_out['consumption_1'].to_csv(os.path.join('..', 'output', 'y_train_1.csv'), index=False)
df_out['consumption_2'].to_csv(os.path.join('..', 'output', 'y_train_2.csv'), index=False)
```

---

# Conclusion

In this first part, we've made an exploration of the data, thus you can see :
* how the weather (and particularly the temperature) could influence the electricity consumption. And the other correlations.
* how cyclic time infos are, and how the consumption vary depending on the hour of the day, the day of the week and the week of the year
Then we've have cleaned and prepared the datasets that will be used in the following part by the machine learning models.
