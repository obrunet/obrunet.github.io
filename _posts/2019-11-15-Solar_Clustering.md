---
title: "Solar energy generation 1/3 - clustering countries & regions"
date: 2019-11-15
categories:
  - Data Science
tags: [Kaggle Competitions]
header:
  image: "/images/2019-11-15-Solar-generation/banner.jpg"
excerpt: "In this 1st part, we're going to make clusters of european countries with similar solar generation capacities."
mathjax: "true"
---

__Description of the data set__

This dataset contains hourly estimates of an area's energy potential for 1986-2015 as a percentage of a power plant's maximum output.

The overall scope of EMHIRES is to allow users to assess the impact of meteorological and climate variability on the generation of solar power in Europe and not to mime the actual evolution of solar power production in the latest decades. For this reason, the hourly solar power generation time series are released for meteorological conditions of the years 1986-2015 (30 years) without considering any changes in the solar installed capacity. Thus, the installed capacity considered is fixed as the one installed at the end of 2015. For this reason, data from EMHIRES should not be compared with actual power generation data other than referring to the reference year 2015.

__Content__
- The data is available at both the national level and the [NUTS 2 level](https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics). The NUTS 2 system divides the EU into 276 statistical units.
- Please see the manual for the technical details of how these estimates were generated.
- This product is intended for policy analysis over a wide area and is not the best for estimating the output from a single system. Please don't use it commercially.

__Acknowledgements__

This dataset was kindly made available by [the European Commission's STETIS program](https://setis.ec.europa.eu/about-setis). You can find the original dataset here.

__Goal of this 1st step__

This is the first part of three. Here we're going to study solar generation on a country level in order to make cluster of country which present the same profile so that each group can be investigate in more details later.


```python
# import of needed libraries

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 300

import warnings
warnings.filterwarnings("ignore")
```

First let's start with the data set for each country :


```python
# let's see what our data set look like

df_solar_co = pd.read_csv("solar_generation_by_country.csv")
df_solar_co.head(2)
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
      <th>AT</th>
      <th>BE</th>
      <th>BG</th>
      <th>CH</th>
      <th>CY</th>
      <th>CZ</th>
      <th>DE</th>
      <th>DK</th>
      <th>EE</th>
      <th>ES</th>
      <th>FI</th>
      <th>FR</th>
      <th>EL</th>
      <th>HR</th>
      <th>HU</th>
      <th>IE</th>
      <th>IT</th>
      <th>LT</th>
      <th>LU</th>
      <th>LV</th>
      <th>NL</th>
      <th>NO</th>
      <th>PL</th>
      <th>PT</th>
      <th>RO</th>
      <th>SI</th>
      <th>SK</th>
      <th>SE</th>
      <th>UK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Each column represent a country, we can list them easily :


```python
df_solar_co.columns
```




    Index(['AT', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR',
           'EL', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT',
           'RO', 'SI', 'SK', 'SE', 'UK'],
          dtype='object')



If needed, here is a dictionnary in python that can help us to make the conversion between the 2 letters and the real name of each country : 


```python
country_dict = {
'AT': 'Austria',
'BE': 'Belgium',
'BG': 'Bulgaria',
'CH': 'Switzerland',
'CY': 'Cyprus',
'CZ': 'Czech Republic',
'DE': 'Germany',
'DK': 'Denmark',
'EE': 'Estonia',
'ES': 'Spain',
'FI': 'Finland',
'FR': 'France',
'EL': 'Greece',
'UK': 'United Kingdom',
'HU': 'Hungary',
'HR': 'Croatia',
'IE': 'Ireland',
'IT': 'Italy',
'LT': 'Lithuania',
'LU': 'Luxembourg',
'LV': 'Latvia',
'NO': 'Norway',
'NL': 'Netherlands',
'PL': 'Poland',
'PT': 'Portugal',
'RO': 'Romania',
'SE': 'Sweden',
'SI': 'Slovenia',
'SK': 'Slovakia'
    }
```

How many columns and lines of records do we have :


```python
df_solar_co.shape
```




    (262968, 29)



Then, let's take a look at the data set at the NUTS 2 level system :


```python
df_solar_nu = pd.read_csv("solar_generation_by_station.csv")
df_solar_nu = df_solar_nu.drop(columns=['time_step'])
df_solar_nu.tail(2)
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
      <th>AT11</th>
      <th>AT21</th>
      <th>AT12</th>
      <th>AT31</th>
      <th>AT32</th>
      <th>AT22</th>
      <th>AT33</th>
      <th>AT34</th>
      <th>AT13</th>
      <th>BE21</th>
      <th>BE31</th>
      <th>BE32</th>
      <th>BE33</th>
      <th>BE22</th>
      <th>BE34</th>
      <th>BE35</th>
      <th>BE23</th>
      <th>BE10</th>
      <th>BE24</th>
      <th>BE25</th>
      <th>BG32</th>
      <th>BG33</th>
      <th>BG31</th>
      <th>BG34</th>
      <th>BG41</th>
      <th>BG42</th>
      <th>CZ06</th>
      <th>CZ03</th>
      <th>CZ08</th>
      <th>CZ01</th>
      <th>CZ05</th>
      <th>CZ04</th>
      <th>CZ02</th>
      <th>CZ07</th>
      <th>DEA5</th>
      <th>DE30</th>
      <th>DE40</th>
      <th>DE91</th>
      <th>DE50</th>
      <th>DED1</th>
      <th>DE71</th>
      <th>DEE1</th>
      <th>DEA4</th>
      <th>DED2</th>
      <th>DEA1</th>
      <th>DE13</th>
      <th>DE72</th>
      <th>DEE2</th>
      <th>DE60</th>
      <th>DE92</th>
      <th>DE12</th>
      <th>DE73</th>
      <th>DEB1</th>
      <th>DEA2</th>
      <th>DED3</th>
      <th>DE93</th>
      <th>DEE3</th>
      <th>DE80</th>
      <th>DE25</th>
      <th>DEA3</th>
      <th>DE22</th>
      <th>DE21</th>
      <th>DE24</th>
      <th>DE23</th>
      <th>DEB3</th>
      <th>DEF0</th>
      <th>DE27</th>
      <th>DE11</th>
      <th>DEG0</th>
      <th>DEB2</th>
      <th>DE14</th>
      <th>DE26</th>
      <th>DE94</th>
      <th>ES61</th>
      <th>ES24</th>
      <th>ES12</th>
      <th>ES13</th>
      <th>ES41</th>
      <th>ES42</th>
      <th>ES51</th>
      <th>ES30</th>
      <th>ES52</th>
      <th>ES43</th>
      <th>ES11</th>
      <th>ES53</th>
      <th>ES23</th>
      <th>ES22</th>
      <th>ES21</th>
      <th>ES62</th>
      <th>FI20</th>
      <th>FI1C</th>
      <th>FI1D</th>
      <th>FI1B</th>
      <th>FI19</th>
      <th>FR42</th>
      <th>FR61</th>
      <th>FR72</th>
      <th>FR25</th>
      <th>FR26</th>
      <th>FR52</th>
      <th>FR24</th>
      <th>FR21</th>
      <th>FR83</th>
      <th>FR43</th>
      <th>FR23</th>
      <th>FR10</th>
      <th>FR81</th>
      <th>FR63</th>
      <th>FR41</th>
      <th>FR62</th>
      <th>FR30</th>
      <th>FR51</th>
      <th>FR22</th>
      <th>FR53</th>
      <th>FR82</th>
      <th>FR71</th>
      <th>EL51</th>
      <th>EL30</th>
      <th>EL63</th>
      <th>EL53</th>
      <th>EL62</th>
      <th>EL54</th>
      <th>EL52</th>
      <th>EL43</th>
      <th>EL42</th>
      <th>EL65</th>
      <th>EL64</th>
      <th>EL61</th>
      <th>EL41</th>
      <th>HU33</th>
      <th>HU23</th>
      <th>HU32</th>
      <th>HU31</th>
      <th>HU21</th>
      <th>HU10</th>
      <th>HU22</th>
      <th>CH02</th>
      <th>CH03</th>
      <th>CH05</th>
      <th>CH01</th>
      <th>CH07</th>
      <th>CH06</th>
      <th>CH04</th>
      <th>IE01</th>
      <th>IE02</th>
      <th>ITF1</th>
      <th>ITF5</th>
      <th>ITF6</th>
      <th>ITF3</th>
      <th>ITH5</th>
      <th>ITH4</th>
      <th>ITI4</th>
      <th>ITC3</th>
      <th>ITC4</th>
      <th>ITI3</th>
      <th>ITF2</th>
      <th>ITC1</th>
      <th>ITF4</th>
      <th>ITG2</th>
      <th>ITG1</th>
      <th>ITI1</th>
      <th>ITH2</th>
      <th>ITI2</th>
      <th>ITC2</th>
      <th>ITH3</th>
      <th>NL13</th>
      <th>NL23</th>
      <th>NL12</th>
      <th>NL22</th>
      <th>NL11</th>
      <th>NL42</th>
      <th>NL41</th>
      <th>NL32</th>
      <th>NL21</th>
      <th>NL31</th>
      <th>NL34</th>
      <th>NL33</th>
      <th>NO04</th>
      <th>NO02</th>
      <th>NO01</th>
      <th>NO03</th>
      <th>NO05</th>
      <th>PL51</th>
      <th>PL61</th>
      <th>PL31</th>
      <th>PL43</th>
      <th>PL11</th>
      <th>PL21</th>
      <th>PL12</th>
      <th>PL52</th>
      <th>PL32</th>
      <th>PL34</th>
      <th>PL63</th>
      <th>PL22</th>
      <th>PL33</th>
      <th>PL62</th>
      <th>PL41</th>
      <th>PL42</th>
      <th>PT18</th>
      <th>PT15</th>
      <th>PT16</th>
      <th>PT17</th>
      <th>PT11</th>
      <th>RO32</th>
      <th>RO12</th>
      <th>RO21</th>
      <th>RO11</th>
      <th>RO31</th>
      <th>RO22</th>
      <th>RO41</th>
      <th>RO42</th>
      <th>SE32</th>
      <th>SE31</th>
      <th>SE12</th>
      <th>SE33</th>
      <th>SE21</th>
      <th>SE11</th>
      <th>SE22</th>
      <th>SE23</th>
      <th>SK01</th>
      <th>SK03</th>
      <th>SK04</th>
      <th>SK02</th>
      <th>UKH2</th>
      <th>UKJ1</th>
      <th>UKD6</th>
      <th>UKK3</th>
      <th>UKD1</th>
      <th>UKF1</th>
      <th>UKK4</th>
      <th>UKK2</th>
      <th>UKH1</th>
      <th>UKE1</th>
      <th>UKL2</th>
      <th>UKM2</th>
      <th>UKH3</th>
      <th>UKK1</th>
      <th>UKD3</th>
      <th>UKJ3</th>
      <th>UKG1</th>
      <th>UKM6</th>
      <th>UKI3UKI4</th>
      <th>UKJ4</th>
      <th>UKD4</th>
      <th>UKF2</th>
      <th>UKF3</th>
      <th>UKD7</th>
      <th>UKM5</th>
      <th>UKE2</th>
      <th>UKN0</th>
      <th>UKC2</th>
      <th>UKI5UKI6</th>
      <th>UKG2</th>
      <th>UKM3</th>
      <th>UKE3</th>
      <th>UKJ2</th>
      <th>UKC1</th>
      <th>UKG3</th>
      <th>UKL1</th>
      <th>UKE4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>262966</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>262967</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_solar_nu.shape
```




    (262968, 260)



---

# Groups of countries or regions with similar profiles

## Clustering with the KMean model 

The objective of clustering is to identify distinct groups in a dataset such that the observations within a group are similar to each other but different from observations in other groups. In k-means clustering, we specify the number of desired clusters k, and the algorithm will assign each observation to exactly one of these k clusters. The algorithm optimizes the groups by minimizing the within-cluster variation (also known as inertia) such that the sum of the within-cluster variations across all k clusters is as small as possible. 

Different runs of k-means will result in slightly different cluster assignments because k-means randomly assigns each observation to one of the k clusters to kick off the clustering process. k-means does this random initialization to speed up the clustering process. After this random initialization, k-means reassigns the observations to different clusters as it attempts to minimize the Euclidean distance between each observation and its cluster’s center point, or centroid. This random initialization is a source of randomness, resulting in slightly different clustering assignments, from one k-means run to another. 

Typically, the k-means algorithm does several runs and chooses the run that has the best separation, defined as the lowest total sum of within-cluster variations across all k clusters. 

Reference : [Hands-On Unsupervised Learning Using Python](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/)

## Evaluating the cluster quality

The goal here isn’t just to make clusters, but to make good, meaningful clusters. Quality clustering is when the datapoints within a cluster are close together, and afar from other clusters.
The two methods to measure the cluster quality are described below:
- Inertia: Intuitively, inertia tells how far away the points within a cluster are. Therefore, a small of inertia is aimed for. The range of inertia’s value starts from zero and goes up.
- Silhouette score: Silhouette score tells how far away the datapoints in one cluster are, from the datapoints in another cluster. The range of silhouette score is from -1 to 1. Score should be closer to 1 than -1.

Reference : [Towards Data Science](https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a)

__Optimal K: the elbow method__

How many clusters would you choose ?

A common, empirical method, is the elbow method. You plot the mean distance of every point toward its cluster center, as a function of the number of clusters. Sometimes the plot has an arm shape, and the elbow would be the optimal K.


```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

## On the NUTS 2 level

Let's keep the records of one year and tranpose the dataset, because we need to have one line per region.


```python
df_solar_transposed = df_solar_nu[-24*365:].T
df_solar_transposed.tail(2)
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
      <th>254208</th>
      <th>254209</th>
      <th>254210</th>
      <th>254211</th>
      <th>254212</th>
      <th>254213</th>
      <th>254214</th>
      <th>254215</th>
      <th>254216</th>
      <th>254217</th>
      <th>254218</th>
      <th>254219</th>
      <th>254220</th>
      <th>254221</th>
      <th>254222</th>
      <th>254223</th>
      <th>254224</th>
      <th>254225</th>
      <th>254226</th>
      <th>254227</th>
      <th>254228</th>
      <th>254229</th>
      <th>254230</th>
      <th>254231</th>
      <th>254232</th>
      <th>254233</th>
      <th>254234</th>
      <th>254235</th>
      <th>254236</th>
      <th>254237</th>
      <th>254238</th>
      <th>254239</th>
      <th>254240</th>
      <th>254241</th>
      <th>254242</th>
      <th>254243</th>
      <th>254244</th>
      <th>254245</th>
      <th>254246</th>
      <th>254247</th>
      <th>254248</th>
      <th>254249</th>
      <th>254250</th>
      <th>254251</th>
      <th>254252</th>
      <th>254253</th>
      <th>254254</th>
      <th>254255</th>
      <th>254256</th>
      <th>254257</th>
      <th>254258</th>
      <th>254259</th>
      <th>254260</th>
      <th>254261</th>
      <th>254262</th>
      <th>254263</th>
      <th>254264</th>
      <th>254265</th>
      <th>254266</th>
      <th>254267</th>
      <th>254268</th>
      <th>254269</th>
      <th>254270</th>
      <th>254271</th>
      <th>254272</th>
      <th>254273</th>
      <th>254274</th>
      <th>254275</th>
      <th>254276</th>
      <th>254277</th>
      <th>254278</th>
      <th>254279</th>
      <th>254280</th>
      <th>254281</th>
      <th>254282</th>
      <th>254283</th>
      <th>254284</th>
      <th>254285</th>
      <th>254286</th>
      <th>254287</th>
      <th>254288</th>
      <th>254289</th>
      <th>254290</th>
      <th>254291</th>
      <th>254292</th>
      <th>254293</th>
      <th>254294</th>
      <th>254295</th>
      <th>254296</th>
      <th>254297</th>
      <th>254298</th>
      <th>254299</th>
      <th>254300</th>
      <th>254301</th>
      <th>254302</th>
      <th>254303</th>
      <th>254304</th>
      <th>254305</th>
      <th>254306</th>
      <th>254307</th>
      <th>254308</th>
      <th>254309</th>
      <th>254310</th>
      <th>254311</th>
      <th>254312</th>
      <th>254313</th>
      <th>254314</th>
      <th>254315</th>
      <th>254316</th>
      <th>254317</th>
      <th>254318</th>
      <th>254319</th>
      <th>254320</th>
      <th>254321</th>
      <th>254322</th>
      <th>254323</th>
      <th>254324</th>
      <th>254325</th>
      <th>254326</th>
      <th>254327</th>
      <th>254328</th>
      <th>254329</th>
      <th>254330</th>
      <th>254331</th>
      <th>254332</th>
      <th>254333</th>
      <th>254334</th>
      <th>254335</th>
      <th>254336</th>
      <th>254337</th>
      <th>254338</th>
      <th>254339</th>
      <th>254340</th>
      <th>254341</th>
      <th>254342</th>
      <th>254343</th>
      <th>254344</th>
      <th>254345</th>
      <th>254346</th>
      <th>254347</th>
      <th>254348</th>
      <th>254349</th>
      <th>254350</th>
      <th>254351</th>
      <th>254352</th>
      <th>254353</th>
      <th>254354</th>
      <th>254355</th>
      <th>254356</th>
      <th>254357</th>
      <th>...</th>
      <th>262818</th>
      <th>262819</th>
      <th>262820</th>
      <th>262821</th>
      <th>262822</th>
      <th>262823</th>
      <th>262824</th>
      <th>262825</th>
      <th>262826</th>
      <th>262827</th>
      <th>262828</th>
      <th>262829</th>
      <th>262830</th>
      <th>262831</th>
      <th>262832</th>
      <th>262833</th>
      <th>262834</th>
      <th>262835</th>
      <th>262836</th>
      <th>262837</th>
      <th>262838</th>
      <th>262839</th>
      <th>262840</th>
      <th>262841</th>
      <th>262842</th>
      <th>262843</th>
      <th>262844</th>
      <th>262845</th>
      <th>262846</th>
      <th>262847</th>
      <th>262848</th>
      <th>262849</th>
      <th>262850</th>
      <th>262851</th>
      <th>262852</th>
      <th>262853</th>
      <th>262854</th>
      <th>262855</th>
      <th>262856</th>
      <th>262857</th>
      <th>262858</th>
      <th>262859</th>
      <th>262860</th>
      <th>262861</th>
      <th>262862</th>
      <th>262863</th>
      <th>262864</th>
      <th>262865</th>
      <th>262866</th>
      <th>262867</th>
      <th>262868</th>
      <th>262869</th>
      <th>262870</th>
      <th>262871</th>
      <th>262872</th>
      <th>262873</th>
      <th>262874</th>
      <th>262875</th>
      <th>262876</th>
      <th>262877</th>
      <th>262878</th>
      <th>262879</th>
      <th>262880</th>
      <th>262881</th>
      <th>262882</th>
      <th>262883</th>
      <th>262884</th>
      <th>262885</th>
      <th>262886</th>
      <th>262887</th>
      <th>262888</th>
      <th>262889</th>
      <th>262890</th>
      <th>262891</th>
      <th>262892</th>
      <th>262893</th>
      <th>262894</th>
      <th>262895</th>
      <th>262896</th>
      <th>262897</th>
      <th>262898</th>
      <th>262899</th>
      <th>262900</th>
      <th>262901</th>
      <th>262902</th>
      <th>262903</th>
      <th>262904</th>
      <th>262905</th>
      <th>262906</th>
      <th>262907</th>
      <th>262908</th>
      <th>262909</th>
      <th>262910</th>
      <th>262911</th>
      <th>262912</th>
      <th>262913</th>
      <th>262914</th>
      <th>262915</th>
      <th>262916</th>
      <th>262917</th>
      <th>262918</th>
      <th>262919</th>
      <th>262920</th>
      <th>262921</th>
      <th>262922</th>
      <th>262923</th>
      <th>262924</th>
      <th>262925</th>
      <th>262926</th>
      <th>262927</th>
      <th>262928</th>
      <th>262929</th>
      <th>262930</th>
      <th>262931</th>
      <th>262932</th>
      <th>262933</th>
      <th>262934</th>
      <th>262935</th>
      <th>262936</th>
      <th>262937</th>
      <th>262938</th>
      <th>262939</th>
      <th>262940</th>
      <th>262941</th>
      <th>262942</th>
      <th>262943</th>
      <th>262944</th>
      <th>262945</th>
      <th>262946</th>
      <th>262947</th>
      <th>262948</th>
      <th>262949</th>
      <th>262950</th>
      <th>262951</th>
      <th>262952</th>
      <th>262953</th>
      <th>262954</th>
      <th>262955</th>
      <th>262956</th>
      <th>262957</th>
      <th>262958</th>
      <th>262959</th>
      <th>262960</th>
      <th>262961</th>
      <th>262962</th>
      <th>262963</th>
      <th>262964</th>
      <th>262965</th>
      <th>262966</th>
      <th>262967</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>UKL1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00454</td>
      <td>0.03268</td>
      <td>0.03582</td>
      <td>0.03150</td>
      <td>0.03397</td>
      <td>0.01626</td>
      <td>0.00882</td>
      <td>0.00228</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.06154</td>
      <td>0.21109</td>
      <td>0.31614</td>
      <td>0.38716</td>
      <td>0.43612</td>
      <td>0.28436</td>
      <td>0.12670</td>
      <td>0.00333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02394</td>
      <td>0.03295</td>
      <td>0.03424</td>
      <td>0.09291</td>
      <td>0.14847</td>
      <td>0.14145</td>
      <td>0.08795</td>
      <td>0.0051</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.03058</td>
      <td>0.14815</td>
      <td>0.23625</td>
      <td>0.25697</td>
      <td>0.19358</td>
      <td>0.09569</td>
      <td>0.03562</td>
      <td>0.00691</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01195</td>
      <td>0.03161</td>
      <td>0.04202</td>
      <td>0.05229</td>
      <td>0.03885</td>
      <td>0.02274</td>
      <td>0.02475</td>
      <td>0.00777</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04288</td>
      <td>0.11208</td>
      <td>0.23709</td>
      <td>0.30366</td>
      <td>0.33393</td>
      <td>0.25748</td>
      <td>0.11807</td>
      <td>0.00818</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01058</td>
      <td>0.01703</td>
      <td>0.05131</td>
      <td>0.03971</td>
      <td>0.05035</td>
      <td>0.03112</td>
      <td>0.00590</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04350</td>
      <td>0.13731</td>
      <td>0.12993</td>
      <td>0.08754</td>
      <td>0.10142</td>
      <td>0.03618</td>
      <td>0.01122</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00540</td>
      <td>0.01848</td>
      <td>0.02360</td>
      <td>0.02316</td>
      <td>0.05459</td>
      <td>0.09607</td>
      <td>0.02429</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.06691</td>
      <td>0.18368</td>
      <td>0.25561</td>
      <td>0.27386</td>
      <td>0.15795</td>
      <td>0.13123</td>
      <td>0.05063</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00780</td>
      <td>0.01089</td>
      <td>0.02214</td>
      <td>0.03015</td>
      <td>0.03136</td>
      <td>0.03555</td>
      <td>0.02274</td>
      <td>0.00038</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02876</td>
      <td>0.06367</td>
      <td>0.05851</td>
      <td>0.05175</td>
      <td>0.02705</td>
      <td>0.05831</td>
      <td>0.04033</td>
      <td>0.00142</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>UKE4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00422</td>
      <td>0.05240</td>
      <td>0.05711</td>
      <td>0.06402</td>
      <td>0.07822</td>
      <td>0.07162</td>
      <td>0.00393</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.06132</td>
      <td>0.22173</td>
      <td>0.31866</td>
      <td>0.34098</td>
      <td>0.26557</td>
      <td>0.23731</td>
      <td>0.04649</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00084</td>
      <td>0.01680</td>
      <td>0.00704</td>
      <td>0.03624</td>
      <td>0.09647</td>
      <td>0.16272</td>
      <td>0.02156</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.09211</td>
      <td>0.30877</td>
      <td>0.45304</td>
      <td>0.50767</td>
      <td>0.42091</td>
      <td>0.31502</td>
      <td>0.12314</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.09572</td>
      <td>0.24905</td>
      <td>0.29321</td>
      <td>0.33915</td>
      <td>0.20611</td>
      <td>0.12507</td>
      <td>0.00189</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04483</td>
      <td>0.03067</td>
      <td>0.07332</td>
      <td>0.09302</td>
      <td>0.23959</td>
      <td>0.22082</td>
      <td>0.08960</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00581</td>
      <td>0.00787</td>
      <td>0.01153</td>
      <td>0.08664</td>
      <td>0.06156</td>
      <td>0.03130</td>
      <td>0.02076</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04454</td>
      <td>0.31123</td>
      <td>0.41020</td>
      <td>0.45262</td>
      <td>0.41357</td>
      <td>0.30795</td>
      <td>0.01848</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00355</td>
      <td>0.02302</td>
      <td>0.04822</td>
      <td>0.03563</td>
      <td>0.01173</td>
      <td>0.00904</td>
      <td>0.00448</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.08471</td>
      <td>0.29928</td>
      <td>0.41253</td>
      <td>0.44490</td>
      <td>0.35211</td>
      <td>0.12400</td>
      <td>0.02514</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02898</td>
      <td>0.01921</td>
      <td>0.00756</td>
      <td>0.01593</td>
      <td>0.03128</td>
      <td>0.02615</td>
      <td>0.00790</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.03240</td>
      <td>0.29711</td>
      <td>0.39308</td>
      <td>0.19770</td>
      <td>0.19014</td>
      <td>0.10068</td>
      <td>0.03251</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 8760 columns</p>
</div>




```python
def plot_elbow_scores(df_, cluster_nb):
    km_inertias, km_scores = [], []

    for k in range(2, cluster_nb):
        km = KMeans(n_clusters=k).fit(df_)
        km_inertias.append(km.inertia_)
        km_scores.append(silhouette_score(df_, km.labels_))

    sns.lineplot(range(2, cluster_nb), km_inertias)
    plt.title('elbow graph / inertia depending on k')
    plt.show()

    sns.lineplot(range(2, cluster_nb), km_scores)
    plt.title('scores depending on k')
    plt.show()
    
plot_elbow_scores(df_solar_transposed, 20)
```


![png](/images/2019-11-15-Solar-generation/output_22_0.png)



![png](/images/2019-11-15-Solar-generation/output_22_1.png)


The best nb k of clusters seems to be 7 even if there isn't any real elbow on the 1st plot.

## On the country level

Let's do exactly the same thing but this same at the country level :


```python
df_solar_transposed = df_solar_co[-24*365*10:].T
plot_elbow_scores(df_solar_transposed, 20)
```


![png](/images/2019-11-15-Solar-generation/output_26_0.png)



![png](/images/2019-11-15-Solar-generation/output_26_1.png)


The best nb k of clusters seems to be 6 even if there isn't any real elbow on the 1st plot.

Finally, we can keep the optimal number k of clusters, and retrieve infos on each group such as number of countries, and names of those countries :


```python
X = df_solar_transposed

km = KMeans(n_clusters=6).fit(X)
X['label'] = km.labels_
print("Cluster nb / Nb of countries in the cluster", X.label.value_counts())

print("\nCountries grouped by cluster")
for k in range(6):
    print(f'\ncluster nb {k} : ', " ".join([country_dict[c] + f' ({c}),' for c in list(X[X.label == k].index)]))
```

    Cluster nb / Nb of countries in the cluster 3    8
    1    8
    5    4
    0    4
    2    3
    4    2
    Name: label, dtype: int64
    
    Countries grouped by cluster
    
    cluster nb 0 :  Estonia (EE), Lithuania (LT), Latvia (LV), Poland (PL),
    
    cluster nb 1 :  Austria (AT), Switzerland (CH), Czech Republic (CZ), Croatia (HR), Hungary (HU), Italy (IT), Slovenia (SI), Slovakia (SK),
    
    cluster nb 2 :  Bulgaria (BG), Greece (EL), Romania (RO),
    
    cluster nb 3 :  Belgium (BE), Germany (DE), Denmark (DK), France (FR), Ireland (IE), Luxembourg (LU), Netherlands (NL), United Kingdom (UK),
    
    cluster nb 4 :  Spain (ES), Portugal (PT),
    
    cluster nb 5 :  Cyprus (CY), Finland (FI), Norway (NO), Sweden (SE),


---
# Conclusions

In this first part, we've managed to make cluster of countries / regions with similar profiles when it comes to solar generation. This can be convenient when, in the second part, we'll analyze in depth the data for one country representative of each cluster instead of 30.

References :
- [the European Commission's STETIS program](https://setis.ec.europa.eu/about-setis)
- [Hands-On Unsupervised Learning Using Python](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/)
- [Towards Data Science](https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a)
