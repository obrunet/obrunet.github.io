---
title: "Evolution of the carbon dioxide emissions over years"
date: 2019-06-12
categories:
  - Projects
tags: [Projects]
header:
  image: "/images/2019-06-12-carbon_dioxide/banner.jpg"
excerpt: "Data analysis for each country per capita"
mathjax: "true"
---

Header photo by [Carlos "Grury" Santos](https://unsplash.com/@grury)

---


```python
# needed libs 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import requests
```


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 100)
```

# Introduction

Carbon dioxide (chemical formula CO 2) [...] is the most significant long-lived greenhouse gas in Earth's atmosphere. Since the Industrial Revolution anthropogenic emissions – primarily from use of fossil fuels and deforestation – `have rapidly increased its concentration in the atmosphere`, leading to global warming [...].

[Source - Wikipedia](https://en.wikipedia.org/wiki/Carbon_dioxide)

__Goal:__ This analysis will show the evolution of CO 2 emissions over the last decades. It's organized in two steps : first per capita, then for each entire country. So what are the countries with the highest emissions ? 

---

# Different datasets aggregation

## Informations per capita

The dataset `CO2_per_capita.csv` comes from the github repo of [Cabonmap](https://github.com/kiln/carbonmap.org/tree/master/data/Shading/With%20alpha-2) for more infos on where the data come from, please visite their [website](http://www.carbonmap.org/) and graphics which are very instructives. An other dataset can be found [here](https://github.com/open-numbers/ddf--gapminder--co2_emission)


```python
# Load the CSV file  / ParserError: Error tokenizing data. C error: Expected 1 fields in line 1094, saw 2 // with no delimiter
df = pd.read_csv('input/CO2_per_capita.csv', delimiter=';')
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Year</th>
      <th>CO2 Per Capita (metric tons)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1961</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1962</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1963</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1964</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Columns names are self explanatory.


```python
df.Year.unique()
```




    array([1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970,
           1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981,
           1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
           1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
           2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011])



## Country codes and continents

This dataset consists of list of countries by continent. Continent codes and country codes are also included.
Credits : [JohnSnowLabs via Datahub.io](https://datahub.io/JohnSnowLabs/country-and-continent-codes-list)


```python
df_continent = pd.read_csv('input/country-and-continent-codes-list-csv_csv.csv')
df_continent.head()
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
      <th>Continent_Name</th>
      <th>Continent_Code</th>
      <th>Country_Name</th>
      <th>Two_Letter_Country_Code</th>
      <th>Three_Letter_Country_Code</th>
      <th>Country_Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Asia</td>
      <td>AS</td>
      <td>Afghanistan, Islamic Republic of</td>
      <td>AF</td>
      <td>AFG</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Europe</td>
      <td>EU</td>
      <td>Albania, Republic of</td>
      <td>AL</td>
      <td>ALB</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Antarctica</td>
      <td>AN</td>
      <td>Antarctica (the territory South of 60 deg S)</td>
      <td>AQ</td>
      <td>ATA</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Africa</td>
      <td>AF</td>
      <td>Algeria, People's Democratic Republic of</td>
      <td>DZ</td>
      <td>DZA</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Oceania</td>
      <td>OC</td>
      <td>American Samoa</td>
      <td>AS</td>
      <td>ASM</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# select only interesting cols
df_continent = df_continent[['Continent_Name', 'Three_Letter_Country_Code']]
# rename them
df_continent.columns = ['Continent', 'Country Code']
# merge two df
df = pd.merge(df, df_continent, on='Country Code')
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Year</th>
      <th>CO2 Per Capita (metric tons)</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1960</td>
      <td>NaN</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1961</td>
      <td>NaN</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1962</td>
      <td>NaN</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1963</td>
      <td>NaN</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1964</td>
      <td>NaN</td>
      <td>North America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_continent.shape
```




    (262, 2)




```python
df_continent.isnull().sum()
```




    Continent       0
    Country Code    4
    dtype: int64



## Countries population over years

This database presents population and other demographic estimates and projections from 1960 to 2050. They are disaggregated by age-group and sex and covers more than 200 economies.
Here i'll keep only relevant infos for our analysis. The db come from [worldbank.org](https://datacatalog.worldbank.org/dataset/population-estimates-and-projections)


```python
df_population = pd.read_csv('input/Population-EstimatesData.csv')

# keep only total population
df_population = df_population[df_population['Indicator Name'] == 'Population, total']

# keep only corresponding years and remove unecessary cols
df_population = df_population.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code', '2012', '2013',
       '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022',
       '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031',
       '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040',
       '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049',
       '2050', 'Unnamed: 95'])
df_population.head()
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
      <th>Country Code</th>
      <th>1960</th>
      <th>1961</th>
      <th>1962</th>
      <th>1963</th>
      <th>1964</th>
      <th>1965</th>
      <th>1966</th>
      <th>1967</th>
      <th>1968</th>
      <th>1969</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>ARB</td>
      <td>9.249093e+07</td>
      <td>9.504450e+07</td>
      <td>9.768229e+07</td>
      <td>1.004111e+08</td>
      <td>1.032399e+08</td>
      <td>1.061750e+08</td>
      <td>1.092306e+08</td>
      <td>1.124069e+08</td>
      <td>1.156802e+08</td>
      <td>1.190165e+08</td>
      <td>1.223984e+08</td>
      <td>1.258074e+08</td>
      <td>1.292694e+08</td>
      <td>1.328634e+08</td>
      <td>1.366968e+08</td>
      <td>1.408433e+08</td>
      <td>1.453324e+08</td>
      <td>1.501331e+08</td>
      <td>1.551837e+08</td>
      <td>1.603925e+08</td>
      <td>1.656895e+08</td>
      <td>1.710520e+08</td>
      <td>1.764901e+08</td>
      <td>1.820058e+08</td>
      <td>1.876108e+08</td>
      <td>1.933103e+08</td>
      <td>1.990938e+08</td>
      <td>2.049425e+08</td>
      <td>2.108448e+08</td>
      <td>2.167874e+08</td>
      <td>2.247354e+08</td>
      <td>2.308299e+08</td>
      <td>2.350372e+08</td>
      <td>2.412861e+08</td>
      <td>2.474359e+08</td>
      <td>2.550297e+08</td>
      <td>2.608435e+08</td>
      <td>2.665751e+08</td>
      <td>2.722351e+08</td>
      <td>2.779629e+08</td>
      <td>2.838320e+08</td>
      <td>2.898504e+08</td>
      <td>2.960266e+08</td>
      <td>3.024345e+08</td>
      <td>3.091620e+08</td>
      <td>3.162647e+08</td>
      <td>3.237733e+08</td>
      <td>3.316538e+08</td>
      <td>3.398255e+08</td>
      <td>3.481451e+08</td>
      <td>3.565089e+08</td>
      <td>3.648959e+08</td>
    </tr>
    <tr>
      <th>341</th>
      <td>CSS</td>
      <td>4.198307e+06</td>
      <td>4.277802e+06</td>
      <td>4.357746e+06</td>
      <td>4.436804e+06</td>
      <td>4.513246e+06</td>
      <td>4.585777e+06</td>
      <td>4.653919e+06</td>
      <td>4.718167e+06</td>
      <td>4.779624e+06</td>
      <td>4.839881e+06</td>
      <td>4.900059e+06</td>
      <td>4.960647e+06</td>
      <td>5.021359e+06</td>
      <td>5.082049e+06</td>
      <td>5.142246e+06</td>
      <td>5.201705e+06</td>
      <td>5.260062e+06</td>
      <td>5.317542e+06</td>
      <td>5.375393e+06</td>
      <td>5.435143e+06</td>
      <td>5.497756e+06</td>
      <td>5.564200e+06</td>
      <td>5.633661e+06</td>
      <td>5.702754e+06</td>
      <td>5.766957e+06</td>
      <td>5.823242e+06</td>
      <td>5.870023e+06</td>
      <td>5.908886e+06</td>
      <td>5.943661e+06</td>
      <td>5.979907e+06</td>
      <td>6.021614e+06</td>
      <td>6.070204e+06</td>
      <td>6.124265e+06</td>
      <td>6.181538e+06</td>
      <td>6.238576e+06</td>
      <td>6.292827e+06</td>
      <td>6.343683e+06</td>
      <td>6.392040e+06</td>
      <td>6.438587e+06</td>
      <td>6.484510e+06</td>
      <td>6.530691e+06</td>
      <td>6.577216e+06</td>
      <td>6.623792e+06</td>
      <td>6.670276e+06</td>
      <td>6.716373e+06</td>
      <td>6.761932e+06</td>
      <td>6.806838e+06</td>
      <td>6.851221e+06</td>
      <td>6.895315e+06</td>
      <td>6.939534e+06</td>
      <td>6.984096e+06</td>
      <td>7.029022e+06</td>
    </tr>
    <tr>
      <th>516</th>
      <td>CEB</td>
      <td>9.140176e+07</td>
      <td>9.223274e+07</td>
      <td>9.300950e+07</td>
      <td>9.384002e+07</td>
      <td>9.471580e+07</td>
      <td>9.544099e+07</td>
      <td>9.614634e+07</td>
      <td>9.704327e+07</td>
      <td>9.788402e+07</td>
      <td>9.860663e+07</td>
      <td>9.913455e+07</td>
      <td>9.963526e+07</td>
      <td>1.003572e+08</td>
      <td>1.011127e+08</td>
      <td>1.019399e+08</td>
      <td>1.028606e+08</td>
      <td>1.037761e+08</td>
      <td>1.046169e+08</td>
      <td>1.053294e+08</td>
      <td>1.059486e+08</td>
      <td>1.065767e+08</td>
      <td>1.071915e+08</td>
      <td>1.077700e+08</td>
      <td>1.083261e+08</td>
      <td>1.088535e+08</td>
      <td>1.093607e+08</td>
      <td>1.098466e+08</td>
      <td>1.102964e+08</td>
      <td>1.106867e+08</td>
      <td>1.108016e+08</td>
      <td>1.107431e+08</td>
      <td>1.104695e+08</td>
      <td>1.101115e+08</td>
      <td>1.100419e+08</td>
      <td>1.100216e+08</td>
      <td>1.098642e+08</td>
      <td>1.096262e+08</td>
      <td>1.094220e+08</td>
      <td>1.092383e+08</td>
      <td>1.090610e+08</td>
      <td>1.084478e+08</td>
      <td>1.076600e+08</td>
      <td>1.069598e+08</td>
      <td>1.066242e+08</td>
      <td>1.063317e+08</td>
      <td>1.060419e+08</td>
      <td>1.057725e+08</td>
      <td>1.053787e+08</td>
      <td>1.050019e+08</td>
      <td>1.048005e+08</td>
      <td>1.044214e+08</td>
      <td>1.041740e+08</td>
    </tr>
    <tr>
      <th>691</th>
      <td>EAR</td>
      <td>9.792874e+08</td>
      <td>1.002524e+09</td>
      <td>1.026587e+09</td>
      <td>1.051415e+09</td>
      <td>1.077037e+09</td>
      <td>1.103433e+09</td>
      <td>1.130587e+09</td>
      <td>1.158571e+09</td>
      <td>1.187274e+09</td>
      <td>1.216766e+09</td>
      <td>1.247053e+09</td>
      <td>1.278138e+09</td>
      <td>1.310016e+09</td>
      <td>1.342709e+09</td>
      <td>1.376073e+09</td>
      <td>1.410094e+09</td>
      <td>1.444720e+09</td>
      <td>1.480010e+09</td>
      <td>1.516216e+09</td>
      <td>1.553704e+09</td>
      <td>1.592674e+09</td>
      <td>1.633180e+09</td>
      <td>1.675079e+09</td>
      <td>1.718098e+09</td>
      <td>1.761829e+09</td>
      <td>1.805996e+09</td>
      <td>1.850487e+09</td>
      <td>1.895290e+09</td>
      <td>1.940220e+09</td>
      <td>1.985084e+09</td>
      <td>2.031828e+09</td>
      <td>2.076398e+09</td>
      <td>2.120567e+09</td>
      <td>2.164508e+09</td>
      <td>2.208444e+09</td>
      <td>2.252579e+09</td>
      <td>2.297015e+09</td>
      <td>2.341634e+09</td>
      <td>2.386185e+09</td>
      <td>2.430487e+09</td>
      <td>2.474601e+09</td>
      <td>2.518353e+09</td>
      <td>2.561813e+09</td>
      <td>2.605067e+09</td>
      <td>2.648272e+09</td>
      <td>2.691528e+09</td>
      <td>2.734860e+09</td>
      <td>2.778276e+09</td>
      <td>2.821797e+09</td>
      <td>2.865440e+09</td>
      <td>2.909411e+09</td>
      <td>2.953406e+09</td>
    </tr>
    <tr>
      <th>866</th>
      <td>EAS</td>
      <td>1.040034e+09</td>
      <td>1.043597e+09</td>
      <td>1.058046e+09</td>
      <td>1.083797e+09</td>
      <td>1.109192e+09</td>
      <td>1.135651e+09</td>
      <td>1.165546e+09</td>
      <td>1.194209e+09</td>
      <td>1.223467e+09</td>
      <td>1.256390e+09</td>
      <td>1.289320e+09</td>
      <td>1.323021e+09</td>
      <td>1.354873e+09</td>
      <td>1.385130e+09</td>
      <td>1.415205e+09</td>
      <td>1.442315e+09</td>
      <td>1.466537e+09</td>
      <td>1.489432e+09</td>
      <td>1.512228e+09</td>
      <td>1.535457e+09</td>
      <td>1.558242e+09</td>
      <td>1.581867e+09</td>
      <td>1.607789e+09</td>
      <td>1.633686e+09</td>
      <td>1.658311e+09</td>
      <td>1.683505e+09</td>
      <td>1.710226e+09</td>
      <td>1.738329e+09</td>
      <td>1.766707e+09</td>
      <td>1.794458e+09</td>
      <td>1.821518e+09</td>
      <td>1.847580e+09</td>
      <td>1.871877e+09</td>
      <td>1.895331e+09</td>
      <td>1.918823e+09</td>
      <td>1.941909e+09</td>
      <td>1.964618e+09</td>
      <td>1.986766e+09</td>
      <td>2.008138e+09</td>
      <td>2.028093e+09</td>
      <td>2.047139e+09</td>
      <td>2.065520e+09</td>
      <td>2.082948e+09</td>
      <td>2.099537e+09</td>
      <td>2.115551e+09</td>
      <td>2.131356e+09</td>
      <td>2.147021e+09</td>
      <td>2.162088e+09</td>
      <td>2.177418e+09</td>
      <td>2.192343e+09</td>
      <td>2.207155e+09</td>
      <td>2.221935e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_population.shape
```




    (259, 53)



There are many missing value here, so a little cleaning is needed first


```python
#df_population.isnull().sum()
#df_population[df_population['1960'].isnull()]
```


```python
df_population = df_population.drop(index=5066)

cols_with_nan = ['1960', '1961', '1962', '1963', '1964', 
    '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', 
    '1978', '1979', '1980', '1981', '1982', '1983', '1984','1985', '1986', '1987', '1988', '1989']
idx = [36916, 44791]
    
df_population.loc[idx, cols_with_nan] = df_population.loc[idx, '1990']
```


```python
df_population.loc[37616] = df_population.loc[37616].fillna(df_population.loc[37616, '1998'])
```


```python
df_population = df_population.melt(id_vars=["Country Code"], 
            #value_vars :  Column(s) to unpivot. If not specified, uses all columns that are not set as `id_vars`. 
            value_name="Population")

# Create a unique key for future join
#df_population['key'] = df_population['Country Code'] + str(df_population['variable'])
#df_population.head()
```


```python
df_population = df_population.rename(index=str, columns={"variable": "Year"})
df_population.Year = df_population.Year.astype('int')
df_population.head()
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
      <th>Country Code</th>
      <th>Year</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARB</td>
      <td>1960</td>
      <td>9.249093e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CSS</td>
      <td>1960</td>
      <td>4.198307e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CEB</td>
      <td>1960</td>
      <td>9.140176e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EAR</td>
      <td>1960</td>
      <td>9.792874e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EAS</td>
      <td>1960</td>
      <td>1.040034e+09</td>
    </tr>
  </tbody>
</table>
</div>



Aggregation of all datasets


```python
df = pd.merge(df, df_population, on=['Country Code', 'Year'])
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Year</th>
      <th>CO2 Per Capita (metric tons)</th>
      <th>Continent</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1960</td>
      <td>NaN</td>
      <td>North America</td>
      <td>54211.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1961</td>
      <td>NaN</td>
      <td>North America</td>
      <td>55438.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1962</td>
      <td>NaN</td>
      <td>North America</td>
      <td>56225.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1963</td>
      <td>NaN</td>
      <td>North America</td>
      <td>56695.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>1964</td>
      <td>NaN</td>
      <td>North America</td>
      <td>57032.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let's check values
#temp[temp['Country Name'] == 'France']
```

# First insights / data cleaning

Number of lines, types of values, irrelevant or weird values...


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11388 entries, 0 to 11387
    Data columns (total 6 columns):
    Country Name                    11388 non-null object
    Country Code                    11388 non-null object
    Year                            11388 non-null int64
    CO2 Per Capita (metric tons)    9233 non-null float64
    Continent                       11388 non-null object
    Population                      11385 non-null float64
    dtypes: float64(2), int64(1), object(3)
    memory usage: 622.8+ KB



```python
df.shape
```




    (11388, 6)




```python
df.duplicated().sum()
```




    0




```python
df.loc[[2650, 2651, 10502, 10503]]
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Year</th>
      <th>CO2 Per Capita (metric tons)</th>
      <th>Continent</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2650</th>
      <td>Cyprus</td>
      <td>CYP</td>
      <td>2011</td>
      <td>6.735376</td>
      <td>Europe</td>
      <td>1124835.0</td>
    </tr>
    <tr>
      <th>2651</th>
      <td>Cyprus</td>
      <td>CYP</td>
      <td>2011</td>
      <td>6.735376</td>
      <td>Asia</td>
      <td>1124835.0</td>
    </tr>
    <tr>
      <th>10502</th>
      <td>Turkey</td>
      <td>TUR</td>
      <td>2011</td>
      <td>4.383105</td>
      <td>Europe</td>
      <td>73409455.0</td>
    </tr>
    <tr>
      <th>10503</th>
      <td>Turkey</td>
      <td>TUR</td>
      <td>2011</td>
      <td>4.383105</td>
      <td>Asia</td>
      <td>73409455.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(index=[2651, 10503])
df.shape
```




    (11386, 6)




```python
df.isnull().sum()
```




    Country Name                       0
    Country Code                       0
    Year                               0
    CO2 Per Capita (metric tons)    2155
    Continent                          0
    Population                         3
    dtype: int64




```python
# Nb of different countries
df['Country Name'].nunique()
```




    212




```python
# Nb of years
df['Year'].nunique()
```




    52




```python
plt.figure(figsize=(16, 6))
sns.distplot(df['CO2 Per Capita (metric tons)'].dropna())
# same thing but longer
#sns.distplot(df[df['CO2 Per Capita (metric tons)'].notnull()]['CO2 Per Capita (metric tons)'])
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_45_0.png)


* At first glance, there are many years/countries with little emissions while very few countries seem to produce a lot of CO2... Let's check this later with other plots.
* There is not any abnormal negative values. Now, where are the missing values i.e in which countries are there only missing values ? What is the proportion of Nan per country...


```python
df[df['CO2 Per Capita (metric tons)'].isna()]['Year'].value_counts().plot(kind='bar', figsize=(16, 6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa36b8a89b0>




![png](/images/2019-06-12-carbon_dioxide/output_47_1.png)


It seems that emissions were not fully recorded before the 90's... Let's dig a little deeper.


```python
# Countries by number of missing values - there are 52 years in the record
df[df['CO2 Per Capita (metric tons)'].isna()]['Country Name'].value_counts().plot(kind='bar', figsize=(16, 6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa36b774898>




![png](/images/2019-06-12-carbon_dioxide/output_49_1.png)


On the bar plot above, one can see that 
* exept Ukraine, Russia, Croatia, Germany
* countries with at least 20 missing values for 52 years of record are not big countries.

Therefore, they can be omitted in our analysis.


```python
# retrieve countries with a least 20 years of missing values
temp_df = df[df['CO2 Per Capita (metric tons)'].isna()]['Country Name'].value_counts() > 20
countries_with_na = pd.DataFrame(temp_df).index
countries_with_na
```




    Index(['Armenia', 'Kazakhstan', 'Azerbaijan', 'Russian Federation', 'Georgia',
           'San Marino', 'South Sudan', 'Sint Maarten (Dutch part)',
           'Virgin Islands (U.S.)', 'Isle of Man', 'Guam', 'Curacao', 'Monaco',
           'St. Martin (French part)', 'American Samoa', 'Puerto Rico',
           'Northern Mariana Islands', 'Tuvalu', 'Liechtenstein', 'Serbia',
           'Montenegro', 'Lesotho', 'Timor-Leste', 'Korea, Dem. People_s Rep.',
           'West Bank and Gaza', 'Micronesia, Fed. Sts.', 'Andorra',
           'Turks and Caicos Islands', 'Eritrea', 'Moldova', 'Macedonia, FYR',
           'Czech Republic', 'Croatia', 'Slovenia', 'Latvia', 'Uzbekistan',
           'Kyrgyz Republic', 'Estonia', 'Belarus', 'Tajikistan',
           'Slovak Republic', 'Turkmenistan', 'Lithuania', 'Ukraine',
           'Bosnia and Herzegovina', 'Germany', 'Marshall Islands', 'Namibia',
           'Aruba', 'Bangladesh', 'Botswana', 'Maldives', 'Malaysia', 'Bhutan',
           'Oman', 'Zambia', 'Somalia', 'Zimbabwe', 'Malawi', 'Seychelles',
           'Kuwait', 'Vanuatu', 'Swaziland', 'Burundi', 'Kiribati', 'Senegal'],
          dtype='object')




```python
# removing countries with more than 20 missing values
df = df[~df['Country Name'].isin(countries_with_na)]
df.shape
```




    (7694, 6)




```python
# filling remaining missing values with an interpolation 
df = df.interpolate()
```


```python
# check if there isn't any Nan anymore
df.isnull().sum()
```




    Country Name                    0
    Country Code                    0
    Year                            0
    CO2 Per Capita (metric tons)    0
    Continent                       0
    Population                      0
    dtype: int64



---

# Analysis per capita

## Which countries have the highest emissions historically ?


```python
df_hist = pd.DataFrame(df.groupby(by='Country Name', as_index=False)['CO2 Per Capita (metric tons)'].mean())
df_hist = df_hist.sort_values(by=['CO2 Per Capita (metric tons)'], ascending=False)
df_hist.head()
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
      <th>Country Name</th>
      <th>CO2 Per Capita (metric tons)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>111</th>
      <td>Qatar</td>
      <td>54.423341</td>
    </tr>
    <tr>
      <th>139</th>
      <td>United Arab Emirates</td>
      <td>31.844877</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Luxembourg</td>
      <td>28.196509</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Brunei Darussalam</td>
      <td>21.497854</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bahrain</td>
      <td>19.867874</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style="whitegrid")
plt.figure(figsize=(16, 40))
sns.barplot(x="CO2 Per Capita (metric tons)", 
            y="Country Name", 
            data=df_hist)
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_59_0.png)


## Which countries have the highest emissions lately ?


```python
# for instance in year 2011
df_lately = df[df.Year == 2011]
df_lately = df_lately.sort_values(by=['CO2 Per Capita (metric tons)'], ascending=False)
plt.figure(figsize=(16, 40))
ax = sns.barplot(x="CO2 Per Capita (metric tons)", y="Country Name", data=df_lately)
```


![png](/images/2019-06-12-carbon_dioxide/output_61_0.png)


## Are the annual emissions decreasing or increasing ?
Let's select few countries to show the evolution 


```python
selected_countries = ['France', 'Israel', 'Switzerland', 'Chile', 'China', 
                      'Colombia', 'United Kingdom', 'United States', 'Brazil', 'Australia']
plt.figure(figsize=(12, 8))
sns.lineplot(x="Year", 
             y="CO2 Per Capita (metric tons)", 
             hue="Country Name", 
             data=df[df["Country Name"].isin(selected_countries)])
plt.title('Evolution of the CO2 emissions per capita for few countries')
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_63_0.png)



```python
plt.figure(figsize=(12, 8))
plt.title('Evolution of the emissions per capita for each continent')

sns.lineplot(x="Year", 
             y="CO2 Per Capita (metric tons)", 
             hue="Continent", 
             data=df)

plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_64_0.png)



```python
df_mean = pd.DataFrame(df.groupby(by=['Continent', 'Year'], as_index=False)['CO2 Per Capita (metric tons)'].mean())

plt.figure(figsize=(12, 8))
plt.title('Evolution of the MEAN emissions per capita for each continent')

sns.lineplot(x="Year", 
             y="CO2 Per Capita (metric tons)", 
             hue="Continent", 
             data=df_mean)

plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_65_0.png)



```python
df_mean.head()
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
      <th>Continent</th>
      <th>Year</th>
      <th>CO2 Per Capita (metric tons)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Africa</td>
      <td>1960</td>
      <td>0.298993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Africa</td>
      <td>1961</td>
      <td>0.310182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Africa</td>
      <td>1962</td>
      <td>0.308247</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Africa</td>
      <td>1963</td>
      <td>0.323735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Africa</td>
      <td>1964</td>
      <td>0.348756</td>
    </tr>
  </tbody>
</table>
</div>



## Evolution of emission share


```python
df_mean_pivot = pd.pivot_table(df_mean, index='Year', values='CO2 Per Capita (metric tons)', columns='Continent')
df_mean_pivot.head()
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
      <th>Continent</th>
      <th>Africa</th>
      <th>Asia</th>
      <th>Europe</th>
      <th>North America</th>
      <th>Oceania</th>
      <th>South America</th>
    </tr>
    <tr>
      <th>Year</th>
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
      <th>1960</th>
      <td>0.298993</td>
      <td>1.015958</td>
      <td>5.310022</td>
      <td>2.134758</td>
      <td>2.734202</td>
      <td>1.563707</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>0.310182</td>
      <td>1.230886</td>
      <td>5.445797</td>
      <td>2.348035</td>
      <td>3.131762</td>
      <td>1.508886</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>0.308247</td>
      <td>1.290272</td>
      <td>5.703817</td>
      <td>2.543537</td>
      <td>2.683807</td>
      <td>1.549366</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>0.323735</td>
      <td>4.151895</td>
      <td>5.997686</td>
      <td>2.315408</td>
      <td>2.848258</td>
      <td>1.549151</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>0.348756</td>
      <td>4.054674</td>
      <td>6.281629</td>
      <td>2.664622</td>
      <td>3.515654</td>
      <td>1.595202</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_mean_perc = df_mean_pivot.divide(df_mean_pivot.sum(axis=1), axis=0)

plt.figure(figsize=(12, 8))

# Make the plot
plt.stackplot(range(1,53),
              df_mean_perc['Africa'], 
              df_mean_perc["Asia"], 
              df_mean_perc["Europe"],
              df_mean_perc["North America"],
              df_mean_perc["Oceania"],
              df_mean_perc["South America"],
              labels=['Africa','Asia','Europe','North America','Oceania','South America'])

# Formatting the plot
plt.legend(loc='upper left')
plt.margins(0,0)
plt.title('Evolution of emissions per capita share over time')
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_69_0.png)


## World map


```python
# create a map
m = folium.Map()
```


```python
countries_list = list(df["Country Name"].unique())

# removing names that are not recognized by the API
rem = ['Congo, Dem. Rep.', 'Congo, Rep.', 'Egypt, Arab Rep.', 'French Polynesia', 'Hong Kong SAR, China',
      'Iran, Islamic Rep.', 'Korea, Rep.', 'Lao PDR', 'Macao SAR, China', 'New Caledonia', 'Philippines', 
       'Venezuela, RB', 'Yemen, Rep.']

for c in rem:
    countries_list.remove(c)
    
countries_list.sort()
countries_list
```




    ['Afghanistan',
     'Albania',
     'Algeria',
     'Angola',
     'Antigua and Barbuda',
     'Argentina',
     'Australia',
     'Austria',
     'Bahamas, The',
     'Bahrain',
     'Barbados',
     'Belgium',
     'Belize',
     'Benin',
     'Bermuda',
     'Bolivia',
     'Brazil',
     'Brunei Darussalam',
     'Bulgaria',
     'Burkina Faso',
     'Cabo Verde',
     'Cambodia',
     'Cameroon',
     'Canada',
     'Cayman Islands',
     'Central African Republic',
     'Chad',
     'Chile',
     'China',
     'Colombia',
     'Comoros',
     'Costa Rica',
     "Cote d'Ivoire",
     'Cuba',
     'Cyprus',
     'Denmark',
     'Djibouti',
     'Dominica',
     'Dominican Republic',
     'Ecuador',
     'El Salvador',
     'Equatorial Guinea',
     'Ethiopia',
     'Faroe Islands',
     'Fiji',
     'Finland',
     'France',
     'Gabon',
     'Gambia, The',
     'Ghana',
     'Greece',
     'Greenland',
     'Grenada',
     'Guatemala',
     'Guinea',
     'Guinea-Bissau',
     'Guyana',
     'Haiti',
     'Honduras',
     'Hungary',
     'Iceland',
     'India',
     'Indonesia',
     'Iraq',
     'Ireland',
     'Israel',
     'Italy',
     'Jamaica',
     'Japan',
     'Jordan',
     'Kenya',
     'Lebanon',
     'Liberia',
     'Libya',
     'Luxembourg',
     'Madagascar',
     'Mali',
     'Malta',
     'Mauritania',
     'Mauritius',
     'Mexico',
     'Mongolia',
     'Morocco',
     'Mozambique',
     'Myanmar',
     'Nepal',
     'Netherlands',
     'New Zealand',
     'Nicaragua',
     'Niger',
     'Nigeria',
     'Norway',
     'Pakistan',
     'Palau',
     'Panama',
     'Papua New Guinea',
     'Paraguay',
     'Peru',
     'Poland',
     'Portugal',
     'Qatar',
     'Romania',
     'Rwanda',
     'Samoa',
     'Sao Tome and Principe',
     'Saudi Arabia',
     'Sierra Leone',
     'Singapore',
     'Solomon Islands',
     'South Africa',
     'Spain',
     'Sri Lanka',
     'St. Kitts and Nevis',
     'St. Lucia',
     'St. Vincent and the Grenadines',
     'Sudan',
     'Suriname',
     'Sweden',
     'Switzerland',
     'Syrian Arab Republic',
     'Tanzania',
     'Thailand',
     'Togo',
     'Tonga',
     'Trinidad and Tobago',
     'Tunisia',
     'Turkey',
     'Uganda',
     'United Arab Emirates',
     'United Kingdom',
     'United States',
     'Uruguay',
     'Vietnam']




```python
def get_boundingbox_country(country, output_as='boundingbox'):
    """
    get the bounding box of a country in EPSG4326 given a country name

    Parameters
    ----------
    country : str
        name of the country in english and lowercase
    output_as : 'str
        chose from 'boundingbox' or 'center'. 
         - 'boundingbox' for [latmin, latmax, lonmin, lonmax]
         - 'center' for [latcenter, loncenter]

    Returns
    -------
    output : list
        list with coordinates as str
    """
    # create url
    url = '{0}{1}{2}'.format('http://nominatim.openstreetmap.org/search?country=',
                             country,
                             '&format=json&polygon=0')
    response = requests.get(url).json()[0]

    # parse response to list
    if output_as == 'boundingbox':
        lst = response[output_as]
        output = [float(i) for i in lst]
    if output_as == 'center':
        lst = [response.get(key) for key in ['lat','lon']]
        output = [float(i) for i in lst]
    return output

# Example
print("Coordinates of France are long={} and lat={}".format(
            get_boundingbox_country("El Salvador", output_as="center")[0],
            get_boundingbox_country("El Salvador", output_as="center")[1]))
```

    Coordinates of France are long=13.8000382 and lat=-88.9140683



```python
df_lately[df_lately['Country Name'] == 'Turkey']['CO2 Per Capita (metric tons)']
```




    10502    4.383105
    Name: CO2 Per Capita (metric tons), dtype: float64




```python
for country in countries_list:
    resp = get_boundingbox_country(country)
    long, lat = resp[0], resp[1]
    emission = float(df_lately[df_lately['Country Name'] == country]['CO2 Per Capita (metric tons)'].values)
    folium.Circle(
                location=[long, lat],
                popup=country,
                radius=100 * emission
                ).add_to(m)
```


```python
m
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTciID48L2Rpdj4KICAgICAgICAKPC9ib2R5Pgo8c2NyaXB0PiAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3ID0gTC5tYXAoCiAgICAgICAgICAgICAgICAibWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3IiwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBjZW50ZXI6IFswLCAwXSwKICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3LAogICAgICAgICAgICAgICAgICAgIHpvb206IDEsCiAgICAgICAgICAgICAgICAgICAgem9vbUNvbnRyb2w6IHRydWUsCiAgICAgICAgICAgICAgICAgICAgcHJlZmVyQ2FudmFzOiBmYWxzZSwKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgKTsKCiAgICAgICAgICAgIAoKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl81MTVkOWRlMjJjMmI0NmNlYmY5MGYzNjBlYjQzMGEwNiA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgImh0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nIiwKICAgICAgICAgICAgICAgIHsiYXR0cmlidXRpb24iOiAiRGF0YSBieSBcdTAwMjZjb3B5OyBcdTAwM2NhIGhyZWY9XCJodHRwOi8vb3BlbnN0cmVldG1hcC5vcmdcIlx1MDAzZU9wZW5TdHJlZXRNYXBcdTAwM2MvYVx1MDAzZSwgdW5kZXIgXHUwMDNjYSBocmVmPVwiaHR0cDovL3d3dy5vcGVuc3RyZWV0bWFwLm9yZy9jb3B5cmlnaHRcIlx1MDAzZU9EYkxcdTAwM2MvYVx1MDAzZS4iLCAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsICJtYXhOYXRpdmVab29tIjogMTgsICJtYXhab29tIjogMTgsICJtaW5ab29tIjogMCwgIm5vV3JhcCI6IGZhbHNlLCAib3BhY2l0eSI6IDEsICJzdWJkb21haW5zIjogImFiYyIsICJ0bXMiOiBmYWxzZX0KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMWMyYTEzNzJiZDhhNGRhNmI3OTc3MjEzMTM5Y2Q5MDggPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsyOS4zNzcyLCAzOC40OTEwNjgyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA0Mi41MjYyMTA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zNDcxZWUyNjJkMDU0YzY0OWZjNDY5ZmZmZmZjMjI5ZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOGIxZTc2NjMwYmUwNGZiNGI0ZDZkNmFmZjk5NDIxYmEgPSAkKGA8ZGl2IGlkPSJodG1sXzhiMWU3NjYzMGJlMDRmYjRiNGQ2ZDZhZmY5OTQyMWJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZmdoYW5pc3RhbjwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8zNDcxZWUyNjJkMDU0YzY0OWZjNDY5ZmZmZmZjMjI5ZC5zZXRDb250ZW50KGh0bWxfOGIxZTc2NjMwYmUwNGZiNGI0ZDZkNmFmZjk5NDIxYmEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMWMyYTEzNzJiZDhhNGRhNmI3OTc3MjEzMTM5Y2Q5MDguYmluZFBvcHVwKHBvcHVwXzM0NzFlZTI2MmQwNTRjNjQ5ZmM0NjlmZmZmZmMyMjlkKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzBkZTk4ZjcxOWY5YjQyNTA5ZWYyMzUwMzFhNDU3MGZlID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMzkuNjQ0ODYyNSwgNDIuNjYxMDg0OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTYwLjcwMzc3MSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZWQ2ZDZjM2UxMWIwNGY2OWEzNDE2NGRiMDIwYzQ2ZDggPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2M4NGY4MDgzN2VjMjQyMzZiODk1NGViMTFlZDY0MmVmID0gJChgPGRpdiBpZD0iaHRtbF9jODRmODA4MzdlYzI0MjM2Yjg5NTRlYjExZWQ2NDJlZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWxiYW5pYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lZDZkNmMzZTExYjA0ZjY5YTM0MTY0ZGIwMjBjNDZkOC5zZXRDb250ZW50KGh0bWxfYzg0ZjgwODM3ZWMyNDIzNmI4OTU0ZWIxMWVkNjQyZWYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMGRlOThmNzE5ZjliNDI1MDllZjIzNTAzMWE0NTcwZmUuYmluZFBvcHVwKHBvcHVwX2VkNmQ2YzNlMTFiMDRmNjlhMzQxNjRkYjAyMGM0NmQ4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2ZlZGRkYzMzOGUwZDQ1ZTVhYTgwYWJlOWM5M2UyNzJhID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTguOTY4MTQ3LCAzNy4yOTYyMDU1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzMzEuNjAzNzg5MiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfY2RmNDY0NWVmNDMwNDAzMGJhNTE3YTk2OGRkZjI3M2MgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzkzYTJhMDU3NTA5NjRmZjA5OTM4NTNkMmNlNjIyODRmID0gJChgPGRpdiBpZD0iaHRtbF85M2EyYTA1NzUwOTY0ZmYwOTkzODUzZDJjZTYyMjg0ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWxnZXJpYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jZGY0NjQ1ZWY0MzA0MDMwYmE1MTdhOTY4ZGRmMjczYy5zZXRDb250ZW50KGh0bWxfOTNhMmEwNTc1MDk2NGZmMDk5Mzg1M2QyY2U2MjI4NGYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZmVkZGRjMzM4ZTBkNDVlNWFhODBhYmU5YzkzZTI3MmEuYmluZFBvcHVwKHBvcHVwX2NkZjQ2NDVlZjQzMDQwMzBiYTUxN2E5NjhkZGYyNzNjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzRjYzRlOTI4NjE0NjRlNDlhOGUzMTVhOTQ2NWJlNDFmID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTE4LjAzODk0NSwgLTQuMzg4MDYzNF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTM1LjQwMDc1MywgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZmU2MmNkM2NhM2FiNGM0OWE4ZGY3NmYxMWYwODNlNjQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2RjOThmN2YzZTQwZDRkYjhhYTM2MTQ5ZDFjYTYzNDliID0gJChgPGRpdiBpZD0iaHRtbF9kYzk4ZjdmM2U0MGQ0ZGI4YWEzNjE0OWQxY2E2MzQ5YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QW5nb2xhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2ZlNjJjZDNjYTNhYjRjNDlhOGRmNzZmMTFmMDgzZTY0LnNldENvbnRlbnQoaHRtbF9kYzk4ZjdmM2U0MGQ0ZGI4YWEzNjE0OWQxY2E2MzQ5Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV80Y2M0ZTkyODYxNDY0ZTQ5YThlMzE1YTk0NjViZTQxZi5iaW5kUG9wdXAocG9wdXBfZmU2MmNkM2NhM2FiNGM0OWE4ZGY3NmYxMWYwODNlNjQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZTY3NWU5N2RkYTAwNDA0NWEzNjRiYTM5ODExMzY3OGQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxNi43NTczOTAxLCAxNy45MjldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDU4Mi4zODA0MzM4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF83NDJiMWVhY2QxYmQ0MDg4ODQ2MjUxYzI3MzkzMzNmOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNGZjNjBlYTBmZmY2NDkyZjgzM2E3OTljYTMyMGIyMzkgPSAkKGA8ZGl2IGlkPSJodG1sXzRmYzYwZWEwZmZmNjQ5MmY4MzNhNzk5Y2EzMjBiMjM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BbnRpZ3VhIGFuZCBCYXJidWRhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzc0MmIxZWFjZDFiZDQwODg4NDYyNTFjMjczOTMzM2Y5LnNldENvbnRlbnQoaHRtbF80ZmM2MGVhMGZmZjY0OTJmODMzYTc5OWNhMzIwYjIzOSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9lNjc1ZTk3ZGRhMDA0MDQ1YTM2NGJhMzk4MTEzNjc4ZC5iaW5kUG9wdXAocG9wdXBfNzQyYjFlYWNkMWJkNDA4ODg0NjI1MWMyNzM5MzMzZjkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZGUyNjQ0NWFkNjJlNDAyNjg3MmExNDQ1NTQ3YjZhYzQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstNTUuMTg1MDc2MSwgLTIxLjc4MTE2OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDU2LjIwNDg1MTE5OTk5OTk1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mOWU0ZmViOGI5NTM0ZTZhYjYyNTdmNTQ4NWQyYTNiNyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMjUwNmY3YjU2NzdjNDk5ZWJjM2M4Yjg2YTRlOTI3ZmYgPSAkKGA8ZGl2IGlkPSJodG1sXzI1MDZmN2I1Njc3YzQ5OWViYzNjOGI4NmE0ZTkyN2ZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BcmdlbnRpbmE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjllNGZlYjhiOTUzNGU2YWI2MjU3ZjU0ODVkMmEzYjcuc2V0Q29udGVudChodG1sXzI1MDZmN2I1Njc3YzQ5OWViYzNjOGI4NmE0ZTkyN2ZmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2RlMjY0NDVhZDYyZTQwMjY4NzJhMTQ0NTU0N2I2YWM0LmJpbmRQb3B1cChwb3B1cF9mOWU0ZmViOGI5NTM0ZTZhYjYyNTdmNTQ4NWQyYTNiNykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV82Y2QxOGRiMWZiNzM0MzdjOTlkZDc2N2U2NzBlY2JmNiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy01NS4zMjI4MTc1LCAtOS4wODgwMTI1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNjUxLjkyMDk5MiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMmYxZjRhNDAzMjBjNDRlZmI3MTIwZjkzYWJiYjU5NzcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzcwOGE0Mzk1NTVlNTQwNWE4OTllZGI1OGUyYTUzNTFkID0gJChgPGRpdiBpZD0iaHRtbF83MDhhNDM5NTU1ZTU0MDVhODk5ZWRiNThlMmE1MzUxZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QXVzdHJhbGlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzJmMWY0YTQwMzIwYzQ0ZWZiNzEyMGY5M2FiYmI1OTc3LnNldENvbnRlbnQoaHRtbF83MDhhNDM5NTU1ZTU0MDVhODk5ZWRiNThlMmE1MzUxZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV82Y2QxOGRiMWZiNzM0MzdjOTlkZDc2N2U2NzBlY2JmNi5iaW5kUG9wdXAocG9wdXBfMmYxZjRhNDAzMjBjNDRlZmI3MTIwZjkzYWJiYjU5NzcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfM2YwN2QxZTRkMTZkNDY1Yzg3MWEwNDJkODMyZGRjYzUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni4zNzIyNzYxLCA0OS4wMjA1MzA1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA3NzYuOTk4MzQyMjk5OTk5OSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfODZmMzY5NmVjMTQyNDBiZGI2Mjk5Y2FlMDdjYWQ0ZDkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2VjMDNkZDkwNmI3NjQ5NGE5MTU2NzZlZmM0ZGRlODBlID0gJChgPGRpdiBpZD0iaHRtbF9lYzAzZGQ5MDZiNzY0OTRhOTE1Njc2ZWZjNGRkZTgwZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QXVzdHJpYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84NmYzNjk2ZWMxNDI0MGJkYjYyOTljYWUwN2NhZDRkOS5zZXRDb250ZW50KGh0bWxfZWMwM2RkOTA2Yjc2NDk0YTkxNTY3NmVmYzRkZGU4MGUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfM2YwN2QxZTRkMTZkNDY1Yzg3MWEwNDJkODMyZGRjYzUuYmluZFBvcHVwKHBvcHVwXzg2ZjM2OTZlYzE0MjQwYmRiNjI5OWNhZTA3Y2FkNGQ5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzU1ZmI0NmU2ZTliZjRjNmI4OTliZWI5NGM4N2U5ZDZlID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMjAuNzA1OTg0NiwgMjcuNDczNDU1MV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNTE5Ljk4NDQwMTksICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2U5YjA3MjQyMzk2YjQxZWJhMDJkNDI2ZDMzMDNlNDc5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF84MmM3ZTg3ZTQzYmQ0NmE0OGZjMTQ4Mzk1OWI1NzBjMCA9ICQoYDxkaXYgaWQ9Imh0bWxfODJjN2U4N2U0M2JkNDZhNDhmYzE0ODM5NTliNTcwYzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJhaGFtYXMsIFRoZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lOWIwNzI0MjM5NmI0MWViYTAyZDQyNmQzMzAzZTQ3OS5zZXRDb250ZW50KGh0bWxfODJjN2U4N2U0M2JkNDZhNDhmYzE0ODM5NTliNTcwYzApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNTVmYjQ2ZTZlOWJmNGM2Yjg5OWJlYjk0Yzg3ZTlkNmUuYmluZFBvcHVwKHBvcHVwX2U5YjA3MjQyMzk2YjQxZWJhMDJkNDI2ZDMzMDNlNDc5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2UwZGRiYzViOGU5YzQ5MGFiMzY4YzVjYmQ3ZTVlYjU4ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMjUuNTM1LCAyNi42ODcyNDQ0XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNzk0LjczMjk4MDk5OTk5OTksICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzVjNzdhZTEzN2U3MjQwNWJiMmEwOTRlN2Q2YzM4YjQ5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mNDFiOGQ1ZGEwNDI0ZTBiOWI1YjcwODQ0MWJjOTBmOCA9ICQoYDxkaXYgaWQ9Imh0bWxfZjQxYjhkNWRhMDQyNGUwYjliNWI3MDg0NDFiYzkwZjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJhaHJhaW48L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWM3N2FlMTM3ZTcyNDA1YmIyYTA5NGU3ZDZjMzhiNDkuc2V0Q29udGVudChodG1sX2Y0MWI4ZDVkYTA0MjRlMGI5YjViNzA4NDQxYmM5MGY4KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2UwZGRiYzViOGU5YzQ5MGFiMzY4YzVjYmQ3ZTVlYjU4LmJpbmRQb3B1cChwb3B1cF81Yzc3YWUxMzdlNzI0MDViYjJhMDk0ZTdkNmMzOGI0OSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8yZGQ0NjZjM2RmYzU0ZDAwOWQ0MDE5NzU1NTdmZDk1ZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzEyLjg0NSwgMTMuNTM1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1NTguMDE3NzYxNzk5OTk5OSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNmJhZjIxOWU3NWRmNGUzNDk0NDVhOTg0YTExNDI0MzcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzcwNDU3YTZmNGZhZjRlMjY5MWI4YTA5Y2JlMWRkZDRjID0gJChgPGRpdiBpZD0iaHRtbF83MDQ1N2E2ZjRmYWY0ZTI2OTFiOGEwOWNiZTFkZGQ0YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmFyYmFkb3M8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNmJhZjIxOWU3NWRmNGUzNDk0NDVhOTg0YTExNDI0Mzcuc2V0Q29udGVudChodG1sXzcwNDU3YTZmNGZhZjRlMjY5MWI4YTA5Y2JlMWRkZDRjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzJkZDQ2NmMzZGZjNTRkMDA5ZDQwMTk3NTU1N2ZkOTVlLmJpbmRQb3B1cChwb3B1cF82YmFmMjE5ZTc1ZGY0ZTM0OTQ0NWE5ODRhMTE0MjQzNykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9lYzEzZmQwNjhiNWQ0OGJmOWRmMDM3ZDFlNmE4Yzk1OSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQ5LjQ5Njk4MjEsIDUxLjU1Mzg4NzZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDg4NC45Mzk4MzAyMDAwMDAxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wODE1MTc0ZWFmMWE0NGNhYTZkNjFhNmM3YmY0MjQwOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNzA5NDExNDFlNGYzNGZiZThhZmJhMjJmOGU0NzNhNjkgPSAkKGA8ZGl2IGlkPSJodG1sXzcwOTQxMTQxZTRmMzRmYmU4YWZiYTIyZjhlNDczYTY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZWxnaXVtPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzA4MTUxNzRlYWYxYTQ0Y2FhNmQ2MWE2YzdiZjQyNDA5LnNldENvbnRlbnQoaHRtbF83MDk0MTE0MWU0ZjM0ZmJlOGFmYmEyMmY4ZTQ3M2E2OSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9lYzEzZmQwNjhiNWQ0OGJmOWRmMDM3ZDFlNmE4Yzk1OS5iaW5kUG9wdXAocG9wdXBfMDgxNTE3NGVhZjFhNDRjYWE2ZDYxYTZjN2JmNDI0MDkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNzlmZjdlNjYxYjJjNDA5ZjkyNzNiZTAzYjc1ZjgyMTUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxNS44ODU3Mjg2LCAxOC40OTU5MTQzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNjcuMDkwNDMwMiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfM2Y5NDU5YjVmOTJmNGIwYTg5Y2ZkN2Y2YjRiZWY4NzQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzYwMGY1YTM1NzY3MjQ4ZWU5YjJjY2U1ZjYzMDcwZWQ1ID0gJChgPGRpdiBpZD0iaHRtbF82MDBmNWEzNTc2NzI0OGVlOWIyY2NlNWY2MzA3MGVkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVsaXplPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzNmOTQ1OWI1ZjkyZjRiMGE4OWNmZDdmNmI0YmVmODc0LnNldENvbnRlbnQoaHRtbF82MDBmNWEzNTc2NzI0OGVlOWIyY2NlNWY2MzA3MGVkNSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV83OWZmN2U2NjFiMmM0MDlmOTI3M2JlMDNiNzVmODIxNS5iaW5kUG9wdXAocG9wdXBfM2Y5NDU5YjVmOTJmNGIwYTg5Y2ZkN2Y2YjRiZWY4NzQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMmM3NTc5MDM5YTBkNDk2N2E0OGUyMjc3ODQ5Yjk4ZmIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs2LjAzOTg2OTYsIDEyLjQwOTI0NDddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUwLjk5NjIyMjU5OTk5OTk5NiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYjVhZjNiZDA3Y2IxNDM3Yzk2NTY1ZmM3OWQwNmNiYjUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzViMzQ3OGYzMDU0ODQ2MGY5ZjYxZWNiYjUzNDE3MzBjID0gJChgPGRpdiBpZD0iaHRtbF81YjM0NzhmMzA1NDg0NjBmOWY2MWVjYmI1MzQxNzMwYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVuaW48L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjVhZjNiZDA3Y2IxNDM3Yzk2NTY1ZmM3OWQwNmNiYjUuc2V0Q29udGVudChodG1sXzViMzQ3OGYzMDU0ODQ2MGY5ZjYxZWNiYjUzNDE3MzBjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzJjNzU3OTAzOWEwZDQ5NjdhNDhlMjI3Nzg0OWI5OGZiLmJpbmRQb3B1cChwb3B1cF9iNWFmM2JkMDdjYjE0MzdjOTY1NjVmYzc5ZDA2Y2JiNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV82NTUwNmY5NzBlZmM0MzFiYTExN2I1ZmU0NWFjODJiNCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzMyLjA0Njk2NTEsIDMyLjU5MTM2OTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDYwNy43MjEwMjEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzY2ZDVjMzQ4NjU5ODRjMmM4YTlmYTAyNWI5YTIwNGM2ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zMmJiMWE0ZmFiZDI0ZTI5OWU4OTA1OWI4MzM2NDQ4ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMzJiYjFhNGZhYmQyNGUyOTllODkwNTliODMzNjQ0OGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcm11ZGE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjZkNWMzNDg2NTk4NGMyYzhhOWZhMDI1YjlhMjA0YzYuc2V0Q29udGVudChodG1sXzMyYmIxYTRmYWJkMjRlMjk5ZTg5MDU5YjgzMzY0NDhkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzY1NTA2Zjk3MGVmYzQzMWJhMTE3YjVmZTQ1YWM4MmI0LmJpbmRQb3B1cChwb3B1cF82NmQ1YzM0ODY1OTg0YzJjOGE5ZmEwMjViOWEyMDRjNikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9jNzBmMDQ3YTYwNTE0MjgyOGYwOTM1NjNhNWYzZmM0MSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0yMi44OTgyNzQyLCAtOS42Njg5NDM4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNTkuOTQ5OTAzOSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDhlYTc4ZWU0ZDJkNDFhMTg3ZTY3ZGUxMTlmNmY4ZWQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzUzNzU3NzMyZWYyYjQ5YWE4ZDE5N2E2MWEwZDJhNDMyID0gJChgPGRpdiBpZD0iaHRtbF81Mzc1NzczMmVmMmI0OWFhOGQxOTdhNjFhMGQyYTQzMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Qm9saXZpYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9kOGVhNzhlZTRkMmQ0MWExODdlNjdkZTExOWY2ZjhlZC5zZXRDb250ZW50KGh0bWxfNTM3NTc3MzJlZjJiNDlhYThkMTk3YTYxYTBkMmE0MzIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfYzcwZjA0N2E2MDUxNDI4MjhmMDkzNTYzYTVmM2ZjNDEuYmluZFBvcHVwKHBvcHVwX2Q4ZWE3OGVlNGQyZDQxYTE4N2U2N2RlMTE5ZjZmOGVkKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2NkMzc2NzhjYzFhYTQyYWRhOTdlZmM3ZmIyODY2Nzg2ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTMzLjg2ODkwNTYsIDUuMjg0Mjg3M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjE5LjEzOTM1NjQwMDAwMDAzLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hNzRjMWU5OGEyMTQ0NmM5YjJkZWM3OWYyNzUxMDA2NSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYjE1MmE4YmVkNTcyNDM0NDg3YTc5ZTczNWYyMWMzMTcgPSAkKGA8ZGl2IGlkPSJodG1sX2IxNTJhOGJlZDU3MjQzNDQ4N2E3OWU3MzVmMjFjMzE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CcmF6aWw8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYTc0YzFlOThhMjE0NDZjOWIyZGVjNzlmMjc1MTAwNjUuc2V0Q29udGVudChodG1sX2IxNTJhOGJlZDU3MjQzNDQ4N2E3OWU3MzVmMjFjMzE3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2NkMzc2NzhjYzFhYTQyYWRhOTdlZmM3ZmIyODY2Nzg2LmJpbmRQb3B1cChwb3B1cF9hNzRjMWU5OGEyMTQ0NmM5YjJkZWM3OWYyNzUxMDA2NSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV84YTFjMGYxNGZmYmY0ZmFkOGUyMDIwODIxZGU4NWNmZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQuMDAyNTA4LCA1LjEwMTE4NTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDI0MzkuMjAxMzM4MDAwMDAwMywgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzBhM2ZlOTk5MjViNDZmZjk1NWVhMmQyMjI4ODk2NTMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzRiYzE2ZTUwYTVjZDRjZmRhYzY5NmE0NjZmZjUwYmVkID0gJChgPGRpdiBpZD0iaHRtbF80YmMxNmU1MGE1Y2Q0Y2ZkYWM2OTZhNDY2ZmY1MGJlZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnJ1bmVpIERhcnVzc2FsYW08L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzBhM2ZlOTk5MjViNDZmZjk1NWVhMmQyMjI4ODk2NTMuc2V0Q29udGVudChodG1sXzRiYzE2ZTUwYTVjZDRjZmRhYzY5NmE0NjZmZjUwYmVkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzhhMWMwZjE0ZmZiZjRmYWQ4ZTIwMjA4MjFkZTg1Y2ZlLmJpbmRQb3B1cChwb3B1cF83MGEzZmU5OTkyNWI0NmZmOTU1ZWEyZDIyMjg4OTY1MykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9mYzcyNzRkMmNkNDE0OGY0YjFiODhiZWY4NDBmODM3NiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQxLjIzNTM5MjksIDQ0LjIxNjcwNjRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDY3MS40MzgyNTEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzdmZmZiYjQ2ZGRkODRmY2E5NDRkNzEwYTk2MzM4OTM2ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wN2FkMTI4MDc2ODI0Y2RkYjk1ZDQyMWI0NGExOTE5MyA9ICQoYDxkaXYgaWQ9Imh0bWxfMDdhZDEyODA3NjgyNGNkZGI5NWQ0MjFiNDRhMTkxOTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJ1bGdhcmlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzdmZmZiYjQ2ZGRkODRmY2E5NDRkNzEwYTk2MzM4OTM2LnNldENvbnRlbnQoaHRtbF8wN2FkMTI4MDc2ODI0Y2RkYjk1ZDQyMWI0NGExOTE5Myk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9mYzcyNzRkMmNkNDE0OGY0YjFiODhiZWY4NDBmODM3Ni5iaW5kUG9wdXAocG9wdXBfN2ZmZmJiNDZkZGQ4NGZjYTk0NGQ3MTBhOTYzMzg5MzYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNzk5MmI4MjYyODI2NDJlNTk0OTRkNTIyMDZkZWYwYzUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs5LjQxMDQ3MTgsIDE1LjA4NF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTEuOTk4MDU2LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mNGNiNDdiZWYxZTI0ZTViODc4YjIyN2I1YzYxZjk3MyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTI4ZDIwMzUzNWM0NDNhZjhiNTdlYTY1ZWEzNjZhODUgPSAkKGA8ZGl2IGlkPSJodG1sXzUyOGQyMDM1MzVjNDQzYWY4YjU3ZWE2NWVhMzY2YTg1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXJraW5hIEZhc288L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjRjYjQ3YmVmMWUyNGU1Yjg3OGIyMjdiNWM2MWY5NzMuc2V0Q29udGVudChodG1sXzUyOGQyMDM1MzVjNDQzYWY4YjU3ZWE2NWVhMzY2YTg1KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzc5OTJiODI2MjgyNjQyZTU5NDk0ZDUyMjA2ZGVmMGM1LmJpbmRQb3B1cChwb3B1cF9mNGNiNDdiZWYxZTI0ZTViODc4YjIyN2I1YzYxZjk3MykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8yYjQ1ODYzOTRmNWM0NTYzYWNkOWJiNzdkM2ZjNjZmNSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzE0LjYwNjYyMjksIDE3LjQwNzExN10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogODUuOTA2MTQzMywgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNDAyYjY0NmNjN2VlNGEyOThmODRhYjM3OGZjYjUzMGIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzg2NTA4MTMzYTRjMjQwM2M4YjFkYTk0NTM4MzQ3YzVmID0gJChgPGRpdiBpZD0iaHRtbF84NjUwODEzM2E0YzI0MDNjOGIxZGE5NDUzODM0N2M1ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FibyBWZXJkZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80MDJiNjQ2Y2M3ZWU0YTI5OGY4NGFiMzc4ZmNiNTMwYi5zZXRDb250ZW50KGh0bWxfODY1MDgxMzNhNGMyNDAzYzhiMWRhOTQ1MzgzNDdjNWYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMmI0NTg2Mzk0ZjVjNDU2M2FjZDliYjc3ZDNmYzY2ZjUuYmluZFBvcHVwKHBvcHVwXzQwMmI2NDZjYzdlZTRhMjk4Zjg0YWIzNzhmY2I1MzBiKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzgyZGQ3M2IzNGRiODQwZmNhYzgwZGE4YmE3M2Q1Yjk0ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbOS40NzUyNjM5LCAxNC42OTA0MjI0XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzMC44MDczMTUyLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mYjlkNTJlZWYwZmE0M2FiOWRlZTJjNmEzOWRkMzU1ZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNmEwM2FmMjM2ZjJkNDNlMDkyMjYwNWUwYjQ3YTk0YTMgPSAkKGA8ZGl2IGlkPSJodG1sXzZhMDNhZjIzNmYyZDQzZTA5MjI2MDVlMGI0N2E5NGEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYW1ib2RpYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mYjlkNTJlZWYwZmE0M2FiOWRlZTJjNmEzOWRkMzU1Zi5zZXRDb250ZW50KGh0bWxfNmEwM2FmMjM2ZjJkNDNlMDkyMjYwNWUwYjQ3YTk0YTMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfODJkZDczYjM0ZGI4NDBmY2FjODBkYThiYTczZDViOTQuYmluZFBvcHVwKHBvcHVwX2ZiOWQ1MmVlZjBmYTQzYWI5ZGVlMmM2YTM5ZGQzNTVmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2E1M2NiZTU2OTc5ZTQ3M2RiMzA1NTcyYzg5OWJkZTZhID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMS42NTQ2NjU5LCAxMy4wODMzMzNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDI2LjgwOTE3OTMwMDAwMDAwNCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTkwMjI1YWE4YTVjNDA2MjhlYjM1ODQ1YjkxNDBhZmQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FkOWYxMmIwMDI3NTRkYzc5YjBkZmJkNDQxODMzOGU3ID0gJChgPGRpdiBpZD0iaHRtbF9hZDlmMTJiMDAyNzU0ZGM3OWIwZGZiZDQ0MTgzMzhlNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FtZXJvb248L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTkwMjI1YWE4YTVjNDA2MjhlYjM1ODQ1YjkxNDBhZmQuc2V0Q29udGVudChodG1sX2FkOWYxMmIwMDI3NTRkYzc5YjBkZmJkNDQxODMzOGU3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2E1M2NiZTU2OTc5ZTQ3M2RiMzA1NTcyYzg5OWJkZTZhLmJpbmRQb3B1cChwb3B1cF9lOTAyMjVhYThhNWM0MDYyOGViMzU4NDViOTE0MGFmZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9lOTdlODIxMjIyYzg0NWNlODRjNGI2M2QzZDA0MDhkZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQxLjY3NjU1NTYsIDgzLjMzNjIxMjhdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDE0MTMuNTgxMzM4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iYWQ0MzU5NzEwOWY0ZjZhODRiNDRjN2RkMWNiYzRiZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTQ4NmJjNjZlMDcyNDRhNWFjZGZmZDM4MmZiMTc3MjggPSAkKGA8ZGl2IGlkPSJodG1sXzU0ODZiYzY2ZTA3MjQ0YTVhY2RmZmQzODJmYjE3NzI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYW5hZGE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYmFkNDM1OTcxMDlmNGY2YTg0YjQ0YzdkZDFjYmM0YmQuc2V0Q29udGVudChodG1sXzU0ODZiYzY2ZTA3MjQ0YTVhY2RmZmQzODJmYjE3NzI4KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2U5N2U4MjEyMjJjODQ1Y2U4NGM0YjYzZDNkMDQwOGRlLmJpbmRQb3B1cChwb3B1cF9iYWQ0MzU5NzEwOWY0ZjZhODRiNDRjN2RkMWNiYzRiZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8wMTFhYjdkNzgyNjc0ZWNmODIzZDY2ODY4OWQwNzBjYSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzE5LjA2MjA2MTksIDE5Ljk1NzM3NTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDEwMzAuNDkzMTA3LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8xZmUzOTU4Y2RlNzE0NjI0YmI2ZTNkY2YzZjM0ZmU5OCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOTRjMWQ3NDBhZGFjNGU1N2FkN2U2OTQ4ODQ5ZGFkYTYgPSAkKGA8ZGl2IGlkPSJodG1sXzk0YzFkNzQwYWRhYzRlNTdhZDdlNjk0ODg0OWRhZGE2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYXltYW4gSXNsYW5kczwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8xZmUzOTU4Y2RlNzE0NjI0YmI2ZTNkY2YzZjM0ZmU5OC5zZXRDb250ZW50KGh0bWxfOTRjMWQ3NDBhZGFjNGU1N2FkN2U2OTQ4ODQ5ZGFkYTYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMDExYWI3ZDc4MjY3NGVjZjgyM2Q2Njg2ODlkMDcwY2EuYmluZFBvcHVwKHBvcHVwXzFmZTM5NThjZGU3MTQ2MjRiYjZlM2RjZjNmMzRmZTk4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzU1NTM3NWQ2OTZkODQyNmViYTRhMjQ1NjE2Zjk0M2RiID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMi4yMDUzODk4LCAxMS4wMDEzODldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDYuMzEyNzgxNCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNjE0MWIzNzRlNGJhNDYwYjg3NjVlOGVjMTQ0ZTZmZjUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzViZmM4ZjUwZTUwNzRhNDNiNDNhNjI5ZGQ5Mzg3N2I1ID0gJChgPGRpdiBpZD0iaHRtbF81YmZjOGY1MGU1MDc0YTQzYjQzYTYyOWRkOTM4NzdiNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBBZnJpY2FuIFJlcHVibGljPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzYxNDFiMzc0ZTRiYTQ2MGI4NzY1ZThlYzE0NGU2ZmY1LnNldENvbnRlbnQoaHRtbF81YmZjOGY1MGU1MDc0YTQzYjQzYTYyOWRkOTM4NzdiNSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV81NTUzNzVkNjk2ZDg0MjZlYmE0YTI0NTYxNmY5NDNkYi5iaW5kUG9wdXAocG9wdXBfNjE0MWIzNzRlNGJhNDYwYjg3NjVlOGVjMTQ0ZTZmZjUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZmMzMjdkZWU0NmZkNDgzMWE3ZmQzYmI0NTc1OWI3MTUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs3LjQ0MTA3LCAyMy40OTc1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA0LjM4MzA0MjQwMDAwMDAwMSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTFkZjM4N2ViZjk3NGE4MmIxOTdmZTUxZWQ1OWYxNWYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FhN2YzMjhjZmNkYTQ0MTlhNDA2MjlkMzRkODQ4ZjIzID0gJChgPGRpdiBpZD0iaHRtbF9hYTdmMzI4Y2ZjZGE0NDE5YTQwNjI5ZDM0ZDg0OGYyMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hhZDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lMWRmMzg3ZWJmOTc0YTgyYjE5N2ZlNTFlZDU5ZjE1Zi5zZXRDb250ZW50KGh0bWxfYWE3ZjMyOGNmY2RhNDQxOWE0MDYyOWQzNGQ4NDhmMjMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZmMzMjdkZWU0NmZkNDgzMWE3ZmQzYmI0NTc1OWI3MTUuYmluZFBvcHVwKHBvcHVwX2UxZGYzODdlYmY5NzRhODJiMTk3ZmU1MWVkNTlmMTVmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2E0YmYwMGM4MGVlNzQxYTliMDEyOTJlNWVhZDJiZGE4ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTU2LjcyNSwgLTE3LjQ5ODM5OThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQ2MS42NDQ1MzgsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzI3OTVlZDI2MjM2ZDQ3MTc4ODAyNWM5ZDYxZjMyMDViID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wYjc4YTViZjUzMWQ0ODg0OWMxNjM0MjIxY2QwZDYwMyA9ICQoYDxkaXYgaWQ9Imh0bWxfMGI3OGE1YmY1MzFkNDg4NDljMTYzNDIyMWNkMGQ2MDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaWxlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzI3OTVlZDI2MjM2ZDQ3MTc4ODAyNWM5ZDYxZjMyMDViLnNldENvbnRlbnQoaHRtbF8wYjc4YTViZjUzMWQ0ODg0OWMxNjM0MjIxY2QwZDYwMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9hNGJmMDBjODBlZTc0MWE5YjAxMjkyZTVlYWQyYmRhOC5iaW5kUG9wdXAocG9wdXBfMjc5NWVkMjYyMzZkNDcxNzg4MDI1YzlkNjFmMzIwNWIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfOTM0ZGM5NzY4YTYwNDVkMWE2OTY5ODU2YjI4NDEwNTQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs4LjgzODM0MzYsIDUzLjU2MDgxNTRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDY3MS4wMzAxOTkxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hZjUxMmRlMGQ0MTI0ZWY5YjI3YWM2NGIzMzFhZjUyNiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfN2M0N2JlYTk1M2E3NGYwY2IwZTBiY2IwZWI0YjQxZjIgPSAkKGA8ZGl2IGlkPSJodG1sXzdjNDdiZWE5NTNhNzRmMGNiMGUwYmNiMGViNGI0MWYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaGluYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hZjUxMmRlMGQ0MTI0ZWY5YjI3YWM2NGIzMzFhZjUyNi5zZXRDb250ZW50KGh0bWxfN2M0N2JlYTk1M2E3NGYwY2IwZTBiY2IwZWI0YjQxZjIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfOTM0ZGM5NzY4YTYwNDVkMWE2OTY5ODU2YjI4NDEwNTQuYmluZFBvcHVwKHBvcHVwX2FmNTEyZGUwZDQxMjRlZjliMjdhYzY0YjMzMWFmNTI2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2M4OTI1MTRlN2JmMDQ4MDQ5MTVjZTNiZTcwYjk1ODNiID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTQuMjMxNjg3MiwgMTYuMDU3MTI2OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTU2LjA2MjkwOTksICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzM2M2YwZWNkNjJkMzQwMzJiNjQ2MmFmNjk2ODU0Y2E1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kZDc2MmI0NTk5YjE0NmJiYTgzYjcwY2U0NzM0MmU4NSA9ICQoYDxkaXYgaWQ9Imh0bWxfZGQ3NjJiNDU5OWIxNDZiYmE4M2I3MGNlNDczNDJlODUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbG9tYmlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzM2M2YwZWNkNjJkMzQwMzJiNjQ2MmFmNjk2ODU0Y2E1LnNldENvbnRlbnQoaHRtbF9kZDc2MmI0NTk5YjE0NmJiYTgzYjcwY2U0NzM0MmU4NSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9jODkyNTE0ZTdiZjA0ODA0OTE1Y2UzYmU3MGI5NTgzYi5iaW5kUG9wdXAocG9wdXBfMzYzZjBlY2Q2MmQzNDAzMmI2NDYyYWY2OTY4NTRjYTUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMzgwMGVkNDA2Y2RjNGFmY2E1NmUxY2I0ZGE5NTY4YjIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMTIuNjIxLCAtMTEuMTY1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAyMi4wMjMzNDczLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kMzVjZWY1MGMzODE0ZWU4ODEyZjVkNzJmNTI4NzY3MSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMWQxMDJkNDQ5YWM3NDA2Nzk1YjIwM2ZmN2JlMzk2N2YgPSAkKGA8ZGl2IGlkPSJodG1sXzFkMTAyZDQ0OWFjNzQwNjc5NWIyMDNmZjdiZTM5NjdmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21vcm9zPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2QzNWNlZjUwYzM4MTRlZTg4MTJmNWQ3MmY1Mjg3NjcxLnNldENvbnRlbnQoaHRtbF8xZDEwMmQ0NDlhYzc0MDY3OTViMjAzZmY3YmUzOTY3Zik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8zODAwZWQ0MDZjZGM0YWZjYTU2ZTFjYjRkYTk1NjhiMi5iaW5kUG9wdXAocG9wdXBfZDM1Y2VmNTBjMzgxNGVlODgxMmY1ZDcyZjUyODc2NzEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMWIwYzFlMTViNjBlNGYyZGEzZWJjNjlkNGMxMDRmYzYgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1LjMzMjk2OTgsIDExLjIxOTU2ODRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDE3MC40OTc0NDk1MDAwMDAwMiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYWY0M2RmODc0YzlkNGM0OGFkZTIzMTY1ZDQ3MjJhZWUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzlkMDMzN2Y5NWM2NTQ0ZDY5Mzk2ZjZhMDM2NzBhZTIwID0gJChgPGRpdiBpZD0iaHRtbF85ZDAzMzdmOTVjNjU0NGQ2OTM5NmY2YTAzNjcwYWUyMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q29zdGEgUmljYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hZjQzZGY4NzRjOWQ0YzQ4YWRlMjMxNjVkNDcyMmFlZS5zZXRDb250ZW50KGh0bWxfOWQwMzM3Zjk1YzY1NDRkNjkzOTZmNmEwMzY3MGFlMjApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMWIwYzFlMTViNjBlNGYyZGEzZWJjNjlkNGMxMDRmYzYuYmluZFBvcHVwKHBvcHVwX2FmNDNkZjg3NGM5ZDRjNDhhZGUyMzE2NWQ0NzIyYWVlKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2NiNGJkNDAwYTBiYTRiMDA5NTQxYTE4NmNkOGQ4NDI0ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNC4xNjIxMjA1LCAxMC43NDAxOTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDMxLjI4Nzc3MDMwMDAwMDAwMiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNGZjZGM1ZjA3ZWQzNGJkYmFkNzdkZTc5ZjdiYWI0NDYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzI2NjIyNjU5MzYwMDQxZmY4M2VlNDc2ZWRkYWU5ZTUyID0gJChgPGRpdiBpZD0iaHRtbF8yNjYyMjY1OTM2MDA0MWZmODNlZTQ3NmVkZGFlOWU1MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q290ZSBkJ0l2b2lyZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80ZmNkYzVmMDdlZDM0YmRiYWQ3N2RlNzlmN2JhYjQ0Ni5zZXRDb250ZW50KGh0bWxfMjY2MjI2NTkzNjAwNDFmZjgzZWU0NzZlZGRhZTllNTIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfY2I0YmQ0MDBhMGJhNGIwMDk1NDFhMTg2Y2Q4ZDg0MjQuYmluZFBvcHVwKHBvcHVwXzRmY2RjNWYwN2VkMzRiZGJhZDc3ZGU3OWY3YmFiNDQ2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2RlMTNiYzY1MGNkYjRmZTg4MTZkNmQyMGM5NjMxOWZlID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTkuNjI3NTI5NCwgMjMuNDgxNjk3Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMzE3LjIzMTUwOTEwMDAwMDA0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lNTRlYTVhNzdkZTU0MTFhODA0ZTQ1ZDJiZDc3M2QyZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjc3NzkyY2Q3ZjZjNDE4OTk1MzMzNjk2OGE3YzU0YjggPSAkKGA8ZGl2IGlkPSJodG1sXzY3Nzc5MmNkN2Y2YzQxODk5NTMzMzY5NjhhN2M1NGI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DdWJhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U1NGVhNWE3N2RlNTQxMWE4MDRlNDVkMmJkNzczZDJlLnNldENvbnRlbnQoaHRtbF82Nzc3OTJjZDdmNmM0MTg5OTUzMzM2OTY4YTdjNTRiOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kZTEzYmM2NTBjZGI0ZmU4ODE2ZDZkMjBjOTYzMTlmZS5iaW5kUG9wdXAocG9wdXBfZTU0ZWE1YTc3ZGU1NDExYTgwNGU0NWQyYmQ3NzNkMmUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNDM4M2FhODAyMmUxNDQ2Mjg3MmM5OTQ1M2RjY2U4NzkgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszNC40MzgzNzA2LCAzNS45MTMyNTJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDY3My41Mzc1ODIyOTk5OTk5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hYTEzMTE0MWY2YjA0Mzc3YWExZTVmMGRjYzA5ZTQ5MSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMjk3NTFlNTBhNWFhNGE3OTg3ODgwMDAyZWVhMjM0ZmIgPSAkKGA8ZGl2IGlkPSJodG1sXzI5NzUxZTUwYTVhYTRhNzk4Nzg4MDAwMmVlYTIzNGZiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DeXBydXM8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYWExMzExNDFmNmIwNDM3N2FhMWU1ZjBkY2MwOWU0OTEuc2V0Q29udGVudChodG1sXzI5NzUxZTUwYTVhYTRhNzk4Nzg4MDAwMmVlYTIzNGZiKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzQzODNhYTgwMjJlMTQ0NjI4NzJjOTk0NTNkY2NlODc5LmJpbmRQb3B1cChwb3B1cF9hYTEzMTE0MWY2YjA0Mzc3YWExZTVmMGRjYzA5ZTQ5MSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8zZmE0MTFjMjQ3NjY0OWNhODUwNjAyNWFhNGQyMDViMyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzU0LjQ1MTY2NjcsIDU3Ljk1MjQyOTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDcyNC44MzI4NzE3LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84OGFjN2FmYTQ5NDE0NTYyYWYxYTVhNjFjMzk1YWJlOCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYjdjZmNiY2VhNjE5NDAxNmI2MGM2ZjUwZWY4MzkyNDMgPSAkKGA8ZGl2IGlkPSJodG1sX2I3Y2ZjYmNlYTYxOTQwMTZiNjBjNmY1MGVmODM5MjQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZW5tYXJrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzg4YWM3YWZhNDk0MTQ1NjJhZjFhNWE2MWMzOTVhYmU4LnNldENvbnRlbnQoaHRtbF9iN2NmY2JjZWE2MTk0MDE2YjYwYzZmNTBlZjgzOTI0Myk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8zZmE0MTFjMjQ3NjY0OWNhODUwNjAyNWFhNGQyMDViMy5iaW5kUG9wdXAocG9wdXBfODhhYzdhZmE0OTQxNDU2MmFmMWE1YTYxYzM5NWFiZTgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZDdkMmU4YzQ1NTQ4NGNhNWI0NmZmYWE5Y2ZhMDk1OTUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMC45MTQ5NTQ3LCAxMi43OTIzMDgxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1Ni4xOTQwOTMwOTk5OTk5OTYsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2MwZWNlNWFhNDYzZDQ3NjBiYWQ2Mjg2NDA3NzJmZDk4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yMjNhMDNjODYwZmE0NzI2OGIwNjBmNDEyNGJhY2ZjMCA9ICQoYDxkaXYgaWQ9Imh0bWxfMjIzYTAzYzg2MGZhNDcyNjhiMDYwZjQxMjRiYWNmYzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRqaWJvdXRpPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2MwZWNlNWFhNDYzZDQ3NjBiYWQ2Mjg2NDA3NzJmZDk4LnNldENvbnRlbnQoaHRtbF8yMjNhMDNjODYwZmE0NzI2OGIwNjBmNDEyNGJhY2ZjMCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kN2QyZThjNDU1NDg0Y2E1YjQ2ZmZhYTljZmEwOTU5NS5iaW5kUG9wdXAocG9wdXBfYzBlY2U1YWE0NjNkNDc2MGJhZDYyODY0MDc3MmZkOTgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNGYxMzRlNTQyYzA1NDczNmEzNTVhYTdjNzdlN2M2M2UgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxNS4wMDc0MjA3LCAxNS43ODcyMjIyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNzQuNjE0MTU2NSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYjJmNDU2MmUzNzc3NGZkYTgzMDljNDMwMTMzMmI3MDUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzMyMzQ2NDI5Nzc5NzQ4MGJhNDc2OGYzYTFhMWUxM2IzID0gJChgPGRpdiBpZD0iaHRtbF8zMjM0NjQyOTc3OTc0ODBiYTQ3NjhmM2ExYTFlMTNiMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9taW5pY2E8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjJmNDU2MmUzNzc3NGZkYTgzMDljNDMwMTMzMmI3MDUuc2V0Q29udGVudChodG1sXzMyMzQ2NDI5Nzc5NzQ4MGJhNDc2OGYzYTFhMWUxM2IzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzRmMTM0ZTU0MmMwNTQ3MzZhMzU1YWE3Yzc3ZTdjNjNlLmJpbmRQb3B1cChwb3B1cF9iMmY0NTYyZTM3Nzc0ZmRhODMwOWM0MzAxMzMyYjcwNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9iMDRlZTg3NmE2ZWU0ZmEzOTU1MGFhMTYwMjVhNjVmMCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzE3LjI3MDE3MDgsIDIxLjI5Mjg3NzFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDIxOC4yOTA3ODg4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lMWZmMDY5MjNiMjY0ZWE3YTM4ODdkOTAyNmFjYjQ5OCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTZkMDZjMjMyNWEzNDVhYmI0YTM1YWM3MmU0NDRjZjggPSAkKGA8ZGl2IGlkPSJodG1sXzU2ZDA2YzIzMjVhMzQ1YWJiNGEzNWFjNzJlNDQ0Y2Y4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb21pbmljYW4gUmVwdWJsaWM8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTFmZjA2OTIzYjI2NGVhN2EzODg3ZDkwMjZhY2I0OTguc2V0Q29udGVudChodG1sXzU2ZDA2YzIzMjVhMzQ1YWJiNGEzNWFjNzJlNDQ0Y2Y4KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2IwNGVlODc2YTZlZTRmYTM5NTUwYWExNjAyNWE2NWYwLmJpbmRQb3B1cChwb3B1cF9lMWZmMDY5MjNiMjY0ZWE3YTM4ODdkOTAyNmFjYjQ5OCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9hNzVmYzE0ZjU1MmU0MWM4ODIzYTEwMTg2ZjA5OTI2OCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy01LjAxNTkzMTQsIDEuODgzNTk2NF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjM1LjQwMTczODY5OTk5OTk4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82OTE4ZGFhZGYzNjI0NWI5OTU5N2IxMzBhM2Q3YzZlNSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfN2EyMTEzZjVlMDA0NDQzNmE0MTEzMTE4NTBhMmZhY2MgPSAkKGA8ZGl2IGlkPSJodG1sXzdhMjExM2Y1ZTAwNDQ0MzZhNDExMzExODUwYTJmYWNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FY3VhZG9yPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzY5MThkYWFkZjM2MjQ1Yjk5NTk3YjEzMGEzZDdjNmU1LnNldENvbnRlbnQoaHRtbF83YTIxMTNmNWUwMDQ0NDM2YTQxMTMxMTg1MGEyZmFjYyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9hNzVmYzE0ZjU1MmU0MWM4ODIzYTEwMTg2ZjA5OTI2OC5iaW5kUG9wdXAocG9wdXBfNjkxOGRhYWRmMzYyNDViOTk1OTdiMTMwYTNkN2M2ZTUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfODc5ZjcyZjZmZDg4NDIwN2I5NDI4YjgxODliYThmZjAgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMi45NzYwNDYsIDE0LjQ1MTA0ODhdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDExMC4zOTk4NTc0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mMjExNjgwZmI1ODY0MmM1YmM3NjU5NzI1MDA5ZDFjYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjVmYzE2YzcxYzMxNGYyYjliMzAzMWNhMjU1YjQ2YjggPSAkKGA8ZGl2IGlkPSJodG1sX2Y1ZmMxNmM3MWMzMTRmMmI5YjMwMzFjYTI1NWI0NmI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FbCBTYWx2YWRvcjwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mMjExNjgwZmI1ODY0MmM1YmM3NjU5NzI1MDA5ZDFjYS5zZXRDb250ZW50KGh0bWxfZjVmYzE2YzcxYzMxNGYyYjliMzAzMWNhMjU1YjQ2YjgpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfODc5ZjcyZjZmZDg4NDIwN2I5NDI4YjgxODliYThmZjAuYmluZFBvcHVwKHBvcHVwX2YyMTE2ODBmYjU4NjQyYzViYzc2NTk3MjUwMDlkMWNhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2IxNmM0NWRjOTY4YTRkNjI5NjhlYTY4MzA0MDVmOTAyID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTEuNjczMjE5NiwgMy45ODldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDg5MC43MjQxNTM2MDAwMDAxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF83N2IxMmI5MGM5MDM0MzVjYTBiNzAxYWMzNzRlMGU0NyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMjBhOTE5NmZiMDFmNDhhZGIxOTYyMzU2NTkwMTU2NDQgPSAkKGA8ZGl2IGlkPSJodG1sXzIwYTkxOTZmYjAxZjQ4YWRiMTk2MjM1NjU5MDE1NjQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FcXVhdG9yaWFsIEd1aW5lYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF83N2IxMmI5MGM5MDM0MzVjYTBiNzAxYWMzNzRlMGU0Ny5zZXRDb250ZW50KGh0bWxfMjBhOTE5NmZiMDFmNDhhZGIxOTYyMzU2NTkwMTU2NDQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfYjE2YzQ1ZGM5NjhhNGQ2Mjk2OGVhNjgzMDQwNWY5MDIuYmluZFBvcHVwKHBvcHVwXzc3YjEyYjkwYzkwMzQzNWNhMGI3MDFhYzM3NGUwZTQ3KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2FjYzY3MzdiZmExZTQwZjM4ZDU0MWUzNDk1ZGIxZmYwID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMy4zOTc0NDgsIDE0Ljg5NDA1MzddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDguMzk0MzExNywgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWZkOTcyYzgwY2MzNDIzMDgxNDdhOTY0NjI0MWE4MzkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2MwYTU1NzhmNjE4MjQzZDM5MGE4ODE4MzgxMDQ1NmU0ID0gJChgPGRpdiBpZD0iaHRtbF9jMGE1NTc4ZjYxODI0M2QzOTBhODgxODM4MTA0NTZlNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RXRoaW9waWE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWZkOTcyYzgwY2MzNDIzMDgxNDdhOTY0NjI0MWE4Mzkuc2V0Q29udGVudChodG1sX2MwYTU1NzhmNjE4MjQzZDM5MGE4ODE4MzgxMDQ1NmU0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2FjYzY3MzdiZmExZTQwZjM4ZDU0MWUzNDk1ZGIxZmYwLmJpbmRQb3B1cChwb3B1cF81ZmQ5NzJjODBjYzM0MjMwODE0N2E5NjQ2MjQxYTgzOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8yNzMxNTQzNWRhOGM0MmEwOWE2MjY2ZTJiODVmM2U1MyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzYxLjI4ODA5OTEsIDYyLjQ0NzYxNjJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDExNzIuMTIxMTc1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82MGNmYTMwNTZjNjI0OGNiYWRlOTYwNmUxMGUzYWY4ZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzcwZGU5YjJjZTM3NGQ4Zjk1MjllODNhNDEzZmEzZTEgPSAkKGA8ZGl2IGlkPSJodG1sXzM3MGRlOWIyY2UzNzRkOGY5NTI5ZTgzYTQxM2ZhM2UxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GYXJvZSBJc2xhbmRzPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzYwY2ZhMzA1NmM2MjQ4Y2JhZGU5NjA2ZTEwZTNhZjhlLnNldENvbnRlbnQoaHRtbF8zNzBkZTliMmNlMzc0ZDhmOTUyOWU4M2E0MTNmYTNlMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8yNzMxNTQzNWRhOGM0MmEwOWE2MjY2ZTJiODVmM2U1My5iaW5kUG9wdXAocG9wdXBfNjBjZmEzMDU2YzYyNDhjYmFkZTk2MDZlMTBlM2FmOGUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZGFmN2ExOTU3NGQ0NDQxOGFmNmNjZGNhOTJmMGZhYmQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMjEuOTQzNDI3NCwgLTEyLjI2MTM4NjZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDE0Mi40ODEzMjQ4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hNTNjMjNjYjg3NmM0MzVkOGJjNmIxNTA1NDI2NDNlYiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMTMwZGYxOGYzMDIzNGY0ZWE2NDRkNmQyZjhjYmE1ZDEgPSAkKGA8ZGl2IGlkPSJodG1sXzEzMGRmMThmMzAyMzRmNGVhNjQ0ZDZkMmY4Y2JhNWQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GaWppPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2E1M2MyM2NiODc2YzQzNWQ4YmM2YjE1MDU0MjY0M2ViLnNldENvbnRlbnQoaHRtbF8xMzBkZjE4ZjMwMjM0ZjRlYTY0NGQ2ZDJmOGNiYTVkMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kYWY3YTE5NTc0ZDQ0NDE4YWY2Y2NkY2E5MmYwZmFiZC5iaW5kUG9wdXAocG9wdXBfYTUzYzIzY2I4NzZjNDM1ZDhiYzZiMTUwNTQyNjQzZWIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMDg4MThjZWY0MWU2NDJlMjhjZDYwOTA1ZjJkMzQ3NTcgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1OS40NTQxNTc4LCA3MC4wOTIyOTM5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxMDE2LjQwNDYxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84ZWFiM2YwODIwNzQ0OTVjOWNkZThiOGNkNjAwZjZmNiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODliODNhMjdkMDEwNDlkYTgxNWEzN2M3Nzc2N2EyODYgPSAkKGA8ZGl2IGlkPSJodG1sXzg5YjgzYTI3ZDAxMDQ5ZGE4MTVhMzdjNzc3NjdhMjg2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GaW5sYW5kPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzhlYWIzZjA4MjA3NDQ5NWM5Y2RlOGI4Y2Q2MDBmNmY2LnNldENvbnRlbnQoaHRtbF84OWI4M2EyN2QwMTA0OWRhODE1YTM3Yzc3NzY3YTI4Nik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8wODgxOGNlZjQxZTY0MmUyOGNkNjA5MDVmMmQzNDc1Ny5iaW5kUG9wdXAocG9wdXBfOGVhYjNmMDgyMDc0NDk1YzljZGU4YjhjZDYwMGY2ZjYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMmM5YTE0NDY5MDNkNDkzNzkyMzg2NWNjMjljZjBkMmEgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstNTAuMjE4NzE2OSwgNTEuMjY4MzE4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1MTguNTA0MzQyNCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzBjOTE4MDJhNmZmNDU0Y2JmMjcwZDQxNGI1ODFkMDAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzgxNzM4MGY0YWVjYTQ5NjNiMjQ1MTNhMmRmMTA0OTJhID0gJChgPGRpdiBpZD0iaHRtbF84MTczODBmNGFlY2E0OTYzYjI0NTEzYTJkZjEwNDkyYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RnJhbmNlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzcwYzkxODAyYTZmZjQ1NGNiZjI3MGQ0MTRiNTgxZDAwLnNldENvbnRlbnQoaHRtbF84MTczODBmNGFlY2E0OTYzYjI0NTEzYTJkZjEwNDkyYSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8yYzlhMTQ0NjkwM2Q0OTM3OTIzODY1Y2MyOWNmMGQyYS5iaW5kUG9wdXAocG9wdXBfNzBjOTE4MDJhNmZmNDU0Y2JmMjcwZDQxNGI1ODFkMDApCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZjMwMGY0NmM0NDE5NDBmZmJmZmIyODg3NzEzOTk1NTggPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstNC4xMDEyMjYxLCAyLjMxODIxNzFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDE0MS44MTY1NzQ5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82Njk4MGVmMTRjYWM0OThmYjUzZDI3NTNhNzg2MzNhMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYTFlZDVkMDgwNDk5NGVkYWJkM2RkOTdiOGQ3ZWQ0ZmYgPSAkKGA8ZGl2IGlkPSJodG1sX2ExZWQ1ZDA4MDQ5OTRlZGFiZDNkZDk3YjhkN2VkNGZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HYWJvbjwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82Njk4MGVmMTRjYWM0OThmYjUzZDI3NTNhNzg2MzNhMy5zZXRDb250ZW50KGh0bWxfYTFlZDVkMDgwNDk5NGVkYWJkM2RkOTdiOGQ3ZWQ0ZmYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZjMwMGY0NmM0NDE5NDBmZmJmZmIyODg3NzEzOTk1NTguYmluZFBvcHVwKHBvcHVwXzY2OTgwZWYxNGNhYzQ5OGZiNTNkMjc1M2E3ODYzM2EzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2JkZjE1MTczNTlmODRiYmM4Yjg2N2Q5YjZmNmM2Mjc5ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTMuMDYxLCAxMy44MjUzMTM3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAyNC4xMDk4NDE2OTk5OTk5OTcsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzY0M2EzNTVlOWY2ZTQ3ZTRiOGYyNjA4Mjk4ZjQ4ZDAyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jMjUzZGE3ZDU3OTk0ZDVlODkxMjVmNTBjYjQ3NzYxMSA9ICQoYDxkaXYgaWQ9Imh0bWxfYzI1M2RhN2Q1Nzk5NGQ1ZTg5MTI1ZjUwY2I0Nzc2MTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhbWJpYSwgVGhlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzY0M2EzNTVlOWY2ZTQ3ZTRiOGYyNjA4Mjk4ZjQ4ZDAyLnNldENvbnRlbnQoaHRtbF9jMjUzZGE3ZDU3OTk0ZDVlODkxMjVmNTBjYjQ3NzYxMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9iZGYxNTE3MzU5Zjg0YmJjOGI4NjdkOWI2ZjZjNjI3OS5iaW5kUG9wdXAocG9wdXBfNjQzYTM1NWU5ZjZlNDdlNGI4ZjI2MDgyOThmNDhkMDIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNjljNDgxNDM4MmI3NGVkZDgwZjhkOTZiZmM1OGIyMTQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0LjUzOTI1MjUsIDExLjE3NDg1NjJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQwLjQzNzk3OTc5OTk5OTk5NCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMjdjN2Y2ZWNkYWU4NGJhOWFlOTY4NGEwYjY0YTc2NTMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2I4ZmM4YWE5MDFjODQwMjBhOGNhMWEyNTUyNjg1N2QyID0gJChgPGRpdiBpZD0iaHRtbF9iOGZjOGFhOTAxYzg0MDIwYThjYTFhMjU1MjY4NTdkMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2hhbmE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMjdjN2Y2ZWNkYWU4NGJhOWFlOTY4NGEwYjY0YTc2NTMuc2V0Q29udGVudChodG1sX2I4ZmM4YWE5MDFjODQwMjBhOGNhMWEyNTUyNjg1N2QyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzY5YzQ4MTQzODJiNzRlZGQ4MGY4ZDk2YmZjNThiMjE0LmJpbmRQb3B1cChwb3B1cF8yN2M3ZjZlY2RhZTg0YmE5YWU5Njg0YTBiNjRhNzY1MykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9jODE5MGU3ZGMzYTE0M2EzYWVlZTQwNDJhZDFlMDIyZiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM0LjcwMDYwOTYsIDQxLjc0ODg4NjJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDc1Ni44NTE5MDgyOTk5OTk5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wZDUyOTY5ZmQ5ODQ0NjVmYmE4MzcwOWE5MTFhMmNkYiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzYwYWM3ZWUzNDk5NDdkYWI0NjAzNzc3ZWM3ODZlZjAgPSAkKGA8ZGl2IGlkPSJodG1sXzM2MGFjN2VlMzQ5OTQ3ZGFiNDYwMzc3N2VjNzg2ZWYwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HcmVlY2U8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMGQ1Mjk2OWZkOTg0NDY1ZmJhODM3MDlhOTExYTJjZGIuc2V0Q29udGVudChodG1sXzM2MGFjN2VlMzQ5OTQ3ZGFiNDYwMzc3N2VjNzg2ZWYwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2M4MTkwZTdkYzNhMTQzYTNhZWVlNDA0MmFkMWUwMjJmLmJpbmRQb3B1cChwb3B1cF8wZDUyOTY5ZmQ5ODQ0NjVmYmE4MzcwOWE5MTFhMmNkYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV84MGJhMGQ5Yzk1Y2U0NDhjOGU4MzUwYjVmYzUyYzZjYyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzU5LjUxNTM4NywgODMuODc1MTcyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxMjQ0LjAzNDEwMSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTY1ZjQ4NjVkMTBkNGIyM2EyYmYxNzgxNjQwY2M0MmMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzk1NTBjYjBhZTRkYjQyOGJiYWY1NzBlZTZhNjY1ZWI0ID0gJChgPGRpdiBpZD0iaHRtbF85NTUwY2IwYWU0ZGI0MjhiYmFmNTcwZWU2YTY2NWViNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3JlZW5sYW5kPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U2NWY0ODY1ZDEwZDRiMjNhMmJmMTc4MTY0MGNjNDJjLnNldENvbnRlbnQoaHRtbF85NTUwY2IwYWU0ZGI0MjhiYmFmNTcwZWU2YTY2NWViNCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV84MGJhMGQ5Yzk1Y2U0NDhjOGU4MzUwYjVmYzUyYzZjYy5iaW5kUG9wdXAocG9wdXBfZTY1ZjQ4NjVkMTBkNGIyM2EyYmYxNzgxNjQwY2M0MmMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZTNkYWY4NjEzZGUyNGI2NDkzYmNjZjk3OTU1YmM3NmYgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMS43ODYsIDEyLjU5NjY1MzJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDI0MC44MTM3NDMxOTk5OTk5OCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzY3ZjAyYWQyOTYwNDYxOTg3MzNkNzBiNzk2NWM2ODAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzVhYzllZTljNjA5MjQ1NTc4MWU5NzQ4MzUyOTBhMmY3ID0gJChgPGRpdiBpZD0iaHRtbF81YWM5ZWU5YzYwOTI0NTU3ODFlOTc0ODM1MjkwYTJmNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3JlbmFkYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jNjdmMDJhZDI5NjA0NjE5ODczM2Q3MGI3OTY1YzY4MC5zZXRDb250ZW50KGh0bWxfNWFjOWVlOWM2MDkyNDU1NzgxZTk3NDgzNTI5MGEyZjcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZTNkYWY4NjEzZGUyNGI2NDkzYmNjZjk3OTU1YmM3NmYuYmluZFBvcHVwKHBvcHVwX2M2N2YwMmFkMjk2MDQ2MTk4NzMzZDcwYjc5NjVjNjgwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzM5ZjNhYTg3Mzg0ZDQ3Zjk4N2Q2OGEyZTA2OGEzODRkID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTMuNjM0NTgwNCwgMTcuODE2NTk0N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNzQuODA1NTA1NTk5OTk5OTksICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M5NDBlNDc5NDUwNTQ5MjZiM2RjZTAzMGNmNDc5NTFjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hNDkzODRhZjVmNDU0NGQwOWM4NTQ3ZDJlMDYzY2Q4NCA9ICQoYDxkaXYgaWQ9Imh0bWxfYTQ5Mzg0YWY1ZjQ1NDRkMDljODU0N2QyZTA2M2NkODQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1YXRlbWFsYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jOTQwZTQ3OTQ1MDU0OTI2YjNkY2UwMzBjZjQ3OTUxYy5zZXRDb250ZW50KGh0bWxfYTQ5Mzg0YWY1ZjQ1NDRkMDljODU0N2QyZTA2M2NkODQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMzlmM2FhODczODRkNDdmOTg3ZDY4YTJlMDY4YTM4NGQuYmluZFBvcHVwKHBvcHVwX2M5NDBlNDc5NDUwNTQ5MjZiM2RjZTAzMGNmNDc5NTFjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2VhNDE2YWZhZWE4MTQyZWU5ZDAxM2NmYWZmODc1Mjg3ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNy4xOTA2MDQ1LCAxMi42NzU2M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjIuOTQyMzQyNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZmY4MTAyY2JkMDJmNDQyODk5ZjljZDY2YjRiNzAyMDQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzY2MTc1ZWFjMDIwMjRiOTdhZGRmYWRhZjkyYTAyNmEyID0gJChgPGRpdiBpZD0iaHRtbF82NjE3NWVhYzAyMDI0Yjk3YWRkZmFkYWY5MmEwMjZhMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3VpbmVhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2ZmODEwMmNiZDAyZjQ0Mjg5OWY5Y2Q2NmI0YjcwMjA0LnNldENvbnRlbnQoaHRtbF82NjE3NWVhYzAyMDI0Yjk3YWRkZmFkYWY5MmEwMjZhMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9lYTQxNmFmYWVhODE0MmVlOWQwMTNjZmFmZjg3NTI4Ny5iaW5kUG9wdXAocG9wdXBfZmY4MTAyY2JkMDJmNDQyODk5ZjljZDY2YjRiNzAyMDQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZjBkODMzOGM5NzJmNDEyMzhkMWIyOGEyNzMyM2RjZDMgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMC42NTE0MjE1LCAxMi42ODYyMzg0XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNC42ODEwNjgzLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zYTA3MjhmZjAxYzI0NjEwODRiMzljODgyYzNmZDExNSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOGM3MWU0OWMyNGNiNGNjNGE1YThmYWY4M2ExMDYwM2IgPSAkKGA8ZGl2IGlkPSJodG1sXzhjNzFlNDljMjRjYjRjYzRhNWE4ZmFmODNhMTA2MDNiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HdWluZWEtQmlzc2F1PC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzNhMDcyOGZmMDFjMjQ2MTA4NGIzOWM4ODJjM2ZkMTE1LnNldENvbnRlbnQoaHRtbF84YzcxZTQ5YzI0Y2I0Y2M0YTVhOGZhZjgzYTEwNjAzYik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9mMGQ4MzM4Yzk3MmY0MTIzOGQxYjI4YTI3MzIzZGNkMy5iaW5kUG9wdXAocG9wdXBfM2EwNzI4ZmYwMWMyNDYxMDg0YjM5Yzg4MmMzZmQxMTUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMzlkZmZkMTZhN2U3NDIxN2IwZGRmNGIzZWE5MDgyNmUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxLjE3MTAwMTcsIDguNjAzODg0Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjM1Ljc3MjIwMjgsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzJiMTAxOGI0MDk0ZTRjMjg5MmVlZTVmOWI5M2NkYzBjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wOGM2ZjJmYzQ1MDQ0YzI0OWFhODQ5ZGRjN2VlZGJkMiA9ICQoYDxkaXYgaWQ9Imh0bWxfMDhjNmYyZmM0NTA0NGMyNDlhYTg0OWRkYzdlZWRiZDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1eWFuYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yYjEwMThiNDA5NGU0YzI4OTJlZWU1ZjliOTNjZGMwYy5zZXRDb250ZW50KGh0bWxfMDhjNmYyZmM0NTA0NGMyNDlhYTg0OWRkYzdlZWRiZDIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMzlkZmZkMTZhN2U3NDIxN2IwZGRmNGIzZWE5MDgyNmUuYmluZFBvcHVwKHBvcHVwXzJiMTAxOGI0MDk0ZTRjMjg5MmVlZTVmOWI5M2NkYzBjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2Q2MzQ5NjMwZjIzYzQzYWZhNWM3Y2QzNTk4NjVhNGQwID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTcuOTA5OTI5MSwgMjAuMjE4MTM2OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjEuNzk2MjA0OCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMmJmYzcwZWE5OTVlNGY5NzhlMTJjNjk1MmViMzczYzkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2M2MDg4Y2FhYzAxYzRkMDdiOTY1MzRkY2ZhZjZhZWQzID0gJChgPGRpdiBpZD0iaHRtbF9jNjA4OGNhYWMwMWM0ZDA3Yjk2NTM0ZGNmYWY2YWVkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFpdGk8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMmJmYzcwZWE5OTVlNGY5NzhlMTJjNjk1MmViMzczYzkuc2V0Q29udGVudChodG1sX2M2MDg4Y2FhYzAxYzRkMDdiOTY1MzRkY2ZhZjZhZWQzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2Q2MzQ5NjMwZjIzYzQzYWZhNWM3Y2QzNTk4NjVhNGQwLmJpbmRQb3B1cChwb3B1cF8yYmZjNzBlYTk5NWU0Zjk3OGUxMmM2OTUyZWIzNzNjOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8wMzYyN2VlZjZkYjg0ZDAxYTgwZjRiNTIwMzQxNDQ4OSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzEyLjk4MDg0ODUsIDE3LjYxOTUyNl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTEwLjM3NDUwNTMsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2MwZDJhYmE5OTc2ZDQ3ZjRiMmU5YTk2NmVmNGI0MzRiID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yMjg5OTQ1MWMwOGM0YzA0OWFhNGU0N2MzMmUxOGNhZSA9ICQoYDxkaXYgaWQ9Imh0bWxfMjI4OTk0NTFjMDhjNGMwNDlhYTRlNDdjMzJlMThjYWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhvbmR1cmFzPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2MwZDJhYmE5OTc2ZDQ3ZjRiMmU5YTk2NmVmNGI0MzRiLnNldENvbnRlbnQoaHRtbF8yMjg5OTQ1MWMwOGM0YzA0OWFhNGU0N2MzMmUxOGNhZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8wMzYyN2VlZjZkYjg0ZDAxYTgwZjRiNTIwMzQxNDQ4OS5iaW5kUG9wdXAocG9wdXBfYzBkMmFiYTk5NzZkNDdmNGIyZTlhOTY2ZWY0YjQzNGIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYzRjMTIwYjNlY2U5NGY3M2I3OWVkOGNiOThkNWM2MzUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0NS43MzcxMjgsIDQ4LjU4NTI1N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDg2LjI5ODk5MzEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzJmM2QyNGU0Y2JhNTRlMzFiNWNkMDQ0YjI3MDM4MDBkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF83OTY0N2Q3YTA3OGQ0YWQ0ODk5MGRlOTEzYmJiODgwYyA9ICQoYDxkaXYgaWQ9Imh0bWxfNzk2NDdkN2EwNzhkNGFkNDg5OTBkZTkxM2JiYjg4MGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bmdhcnk8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMmYzZDI0ZTRjYmE1NGUzMWI1Y2QwNDRiMjcwMzgwMGQuc2V0Q29udGVudChodG1sXzc5NjQ3ZDdhMDc4ZDRhZDQ4OTkwZGU5MTNiYmI4ODBjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2M0YzEyMGIzZWNlOTRmNzNiNzllZDhjYjk4ZDVjNjM1LmJpbmRQb3B1cChwb3B1cF8yZjNkMjRlNGNiYTU0ZTMxYjVjZDA0NGIyNzAzODAwZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xYThkNTg0NjY4Nzg0MDgzYWM1YzNlZTAwZTVmMTQ3MSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzYzLjA4NTkxNzcsIDY3LjM1M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNTg5LjY4Mjg5NzksICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzMxNDYzMjE4OGEwYzQzMjY5MGQ0NmU5ZTRmZWNlN2UxID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yZmU1MGJkODA3OTk0Nzc0OGFmYzAyYjIwOWFlOWZlMiA9ICQoYDxkaXYgaWQ9Imh0bWxfMmZlNTBiZDgwNzk5NDc3NDhhZmMwMmIyMDlhZTlmZTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkljZWxhbmQ8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMzE0NjMyMTg4YTBjNDMyNjkwZDQ2ZTllNGZlY2U3ZTEuc2V0Q29udGVudChodG1sXzJmZTUwYmQ4MDc5OTQ3NzQ4YWZjMDJiMjA5YWU5ZmUyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzFhOGQ1ODQ2Njg3ODQwODNhYzVjM2VlMDBlNWYxNDcxLmJpbmRQb3B1cChwb3B1cF8zMTQ2MzIxODhhMGM0MzI2OTBkNDZlOWU0ZmVjZTdlMSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9hNTQ3MDIwN2IxMjM0NjhmOTY3MmJmMDUxMGE5MjM5OCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzYuMjMyNTI3NCwgMzUuNjc0NTQ1N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTY2LjI4NzM0ODMsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzBiMDVmNmZkMGQ3NTRlM2U4NTMzNzViZDA4NTY2MDZlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kYjg3YWMwYjJmMjY0ZDBlYjg1MGI2NjAwNjQ1MDBkOCA9ICQoYDxkaXYgaWQ9Imh0bWxfZGI4N2FjMGIyZjI2NGQwZWI4NTBiNjYwMDY0NTAwZDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzBiMDVmNmZkMGQ3NTRlM2U4NTMzNzViZDA4NTY2MDZlLnNldENvbnRlbnQoaHRtbF9kYjg3YWMwYjJmMjY0ZDBlYjg1MGI2NjAwNjQ1MDBkOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9hNTQ3MDIwN2IxMjM0NjhmOTY3MmJmMDUxMGE5MjM5OC5iaW5kUG9wdXAocG9wdXBfMGIwNWY2ZmQwZDc1NGUzZTg1MzM3NWJkMDg1NjYwNmUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYWJlZTc3MjY3YzRjNGEyZThmNTM2NjI1NmQzNGNkMGEgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMTEuMjA4NTY2OSwgNi4yNzQ0NDk2XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAyMzAuMzc4MDk4MzAwMDAwMDMsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzhhNWI2ZWU0ODE3ZTQyMDE4MGU0ZjI5ZmE0Y2IwN2E0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lZWY2OWE0M2JhOGU0Zjc3YjZmNTZiNzg2M2I1YWQyNCA9ICQoYDxkaXYgaWQ9Imh0bWxfZWVmNjlhNDNiYThlNGY3N2I2ZjU2Yjc4NjNiNWFkMjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZG9uZXNpYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84YTViNmVlNDgxN2U0MjAxODBlNGYyOWZhNGNiMDdhNC5zZXRDb250ZW50KGh0bWxfZWVmNjlhNDNiYThlNGY3N2I2ZjU2Yjc4NjNiNWFkMjQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfYWJlZTc3MjY3YzRjNGEyZThmNTM2NjI1NmQzNGNkMGEuYmluZFBvcHVwKHBvcHVwXzhhNWI2ZWU0ODE3ZTQyMDE4MGU0ZjI5ZmE0Y2IwN2E0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2I3NjFjYzY0NjY4ZTRlODdiZjhkYTRlY2UyM2UzZmVhID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMjkuMDU4NTY2MSwgMzcuMzgwNjY4N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDIwLjE2MzUxMjk5OTk5OTk3LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84MzEyOGMyYzcwMjY0MGE3ODc4YTFmOTQ2MGY0NDFiMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjE0NmVhZWFmOGRhNDc4NDlkNTJkYTAyYTI2ZjkxMjMgPSAkKGA8ZGl2IGlkPSJodG1sX2YxNDZlYWVhZjhkYTQ3ODQ5ZDUyZGEwMmEyNmY5MTIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5JcmFxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzgzMTI4YzJjNzAyNjQwYTc4NzhhMWY5NDYwZjQ0MWIzLnNldENvbnRlbnQoaHRtbF9mMTQ2ZWFlYWY4ZGE0Nzg0OWQ1MmRhMDJhMjZmOTEyMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9iNzYxY2M2NDY2OGU0ZTg3YmY4ZGE0ZWNlMjNlM2ZlYS5iaW5kUG9wdXAocG9wdXBfODMxMjhjMmM3MDI2NDBhNzg3OGExZjk0NjBmNDQxYjMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMjhmZTkxZmJmYmJkNDRhOGEyZDgzMzQzMmI4Y2JkN2YgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1MS4yMjIsIDU1LjYzNl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNzg4LjA3NTkzMjYwMDAwMDEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2UwNjgwMjAxYTRjZjQ5MDA4Y2VhMjQ5Y2Y0YzM5OTk0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jYmFiOTNjMmZhOGY0MzRkYTdkYmViZjgwMTYwNzc0YSA9ICQoYDxkaXYgaWQ9Imh0bWxfY2JhYjkzYzJmYThmNDM0ZGE3ZGJlYmY4MDE2MDc3NGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPklyZWxhbmQ8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTA2ODAyMDFhNGNmNDkwMDhjZWEyNDljZjRjMzk5OTQuc2V0Q29udGVudChodG1sX2NiYWI5M2MyZmE4ZjQzNGRhN2RiZWJmODAxNjA3NzRhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzI4ZmU5MWZiZmJiZDQ0YThhMmQ4MzM0MzJiOGNiZDdmLmJpbmRQb3B1cChwb3B1cF9lMDY4MDIwMWE0Y2Y0OTAwOGNlYTI0OWNmNGMzOTk5NCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9kYzkwMWEyNTYwNTc0MWY5YjNlYTcwYThmNTVlYmQxMiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzI5LjQ1MzM3OTYsIDMzLjMzNTYzMTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDg5NS4yNDEzNTMxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iOGJjOTUwYzFmODY0N2Y1ODZiNjJlNzAzMDZkODBhMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOGVmYzkyOTk4ZGJiNDlmM2JkYjg3MTUzMzY0Mzg2YTUgPSAkKGA8ZGl2IGlkPSJodG1sXzhlZmM5Mjk5OGRiYjQ5ZjNiZGI4NzE1MzM2NDM4NmE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc3JhZWw8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjhiYzk1MGMxZjg2NDdmNTg2YjYyZTcwMzA2ZDgwYTMuc2V0Q29udGVudChodG1sXzhlZmM5Mjk5OGRiYjQ5ZjNiZGI4NzE1MzM2NDM4NmE1KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2RjOTAxYTI1NjA1NzQxZjliM2VhNzBhOGY1NWViZDEyLmJpbmRQb3B1cChwb3B1cF9iOGJjOTUwYzFmODY0N2Y1ODZiNjJlNzAzMDZkODBhMykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xYjNmM2Y2Njk4NWQ0ZDc3ODBkZDg0ODZlMTVmMDI4MyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM1LjI4ODk2MTYsIDQ3LjA5MjE0NjJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDY3MC4yNTU3NjE0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hZmNkNTA3OTg2OGE0OTZkOTExNDM3NmFlNzFlMjE2YyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfN2ZmYzgyYzM4Yzg2NDY4ZmFjYzI2YTQ0YWNkNjRmMmYgPSAkKGA8ZGl2IGlkPSJodG1sXzdmZmM4MmMzOGM4NjQ2OGZhY2MyNmE0NGFjZDY0ZjJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5JdGFseTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hZmNkNTA3OTg2OGE0OTZkOTExNDM3NmFlNzFlMjE2Yy5zZXRDb250ZW50KGh0bWxfN2ZmYzgyYzM4Yzg2NDY4ZmFjYzI2YTQ0YWNkNjRmMmYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMWIzZjNmNjY5ODVkNGQ3NzgwZGQ4NDg2ZTE1ZjAyODMuYmluZFBvcHVwKHBvcHVwX2FmY2Q1MDc5ODY4YTQ5NmQ5MTE0Mzc2YWU3MWUyMTZjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2U4YjQ2MjdiNDU5ZDQwN2U5YWJlOWM1ZTM0YzdmYmY0ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTYuNTg5OTQ0MywgMTguNzI1NjM5NF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjg3LjI2NTU2OTMsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzE1YWNhZTU3OTQ0NzQ2Y2FiNTA4ZmUxOTgyM2QyYWEwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85OWFkZmM5MTVjZmY0MmE3OGEwM2JjYjNkNjEzZDQyOSA9ICQoYDxkaXYgaWQ9Imh0bWxfOTlhZGZjOTE1Y2ZmNDJhNzhhMDNiY2IzZDYxM2Q0MjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkphbWFpY2E8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMTVhY2FlNTc5NDQ3NDZjYWI1MDhmZTE5ODIzZDJhYTAuc2V0Q29udGVudChodG1sXzk5YWRmYzkxNWNmZjQyYTc4YTAzYmNiM2Q2MTNkNDI5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2U4YjQ2MjdiNDU5ZDQwN2U5YWJlOWM1ZTM0YzdmYmY0LmJpbmRQb3B1cChwb3B1cF8xNWFjYWU1Nzk0NDc0NmNhYjUwOGZlMTk4MjNkMmFhMCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV82ZTM2Y2Q4N2M3ZGI0OTgxOTE0NDE0M2Y2MGFkMzc2YyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzIwLjIxNDU4MTEsIDQ1LjcxMTIwNDZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDkyOS4xODM0MzAyOTk5OTk5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lNDlkY2UxYzRkY2Y0OWVjYTAzN2EzNWU3NmEwZjY4YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzYxMDJhYjc1MGUyNDljYWFlZjUzNWRjNjRiZDlhNGIgPSAkKGA8ZGl2IGlkPSJodG1sXzM2MTAyYWI3NTBlMjQ5Y2FhZWY1MzVkYzY0YmQ5YTRiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5KYXBhbjwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lNDlkY2UxYzRkY2Y0OWVjYTAzN2EzNWU3NmEwZjY4YS5zZXRDb250ZW50KGh0bWxfMzYxMDJhYjc1MGUyNDljYWFlZjUzNWRjNjRiZDlhNGIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNmUzNmNkODdjN2RiNDk4MTkxNDQxNDNmNjBhZDM3NmMuYmluZFBvcHVwKHBvcHVwX2U0OWRjZTFjNGRjZjQ5ZWNhMDM3YTM1ZTc2YTBmNjhhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2YxMmY3OThhOGQxNDQ0OWQ4Y2QxZTRhNGZlOGJlMTU3ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMjkuMTgzNDAxLCAzMy4zNzUwNjE3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzNjAuMTE0NzA2NCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMWRmZGFjMjQzNmJjNDVkZDliODE1ZTBhNTE2Nzg0MTIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Y0YmM1YWI2MDFiYzRmZmU4NjVkZjFiNTBjNDcxZDRiID0gJChgPGRpdiBpZD0iaHRtbF9mNGJjNWFiNjAxYmM0ZmZlODY1ZGYxYjUwYzQ3MWQ0YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Sm9yZGFuPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzFkZmRhYzI0MzZiYzQ1ZGQ5YjgxNWUwYTUxNjc4NDEyLnNldENvbnRlbnQoaHRtbF9mNGJjNWFiNjAxYmM0ZmZlODY1ZGYxYjUwYzQ3MWQ0Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9mMTJmNzk4YThkMTQ0NDlkOGNkMWU0YTRmZThiZTE1Ny5iaW5kUG9wdXAocG9wdXBfMWRmZGFjMjQzNmJjNDVkZDliODE1ZTBhNTE2Nzg0MTIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfY2Q2MDZjZTg3YjdiNDYyMWIxMzVhY2UzMmMyOTVhMzggPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstNC44OTk1MjA0LCA0LjYyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzMi43NTY5MTcxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82M2M3ZDU0MTMwNGM0MTdjYTQ2NDc1Y2VlZmEyZjAyYiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTVkMzUwMDcyMDY2NDU3M2E1ZTliMWZiMmYwODA4ODEgPSAkKGA8ZGl2IGlkPSJodG1sXzU1ZDM1MDA3MjA2NjQ1NzNhNWU5YjFmYjJmMDgwODgxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW55YTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82M2M3ZDU0MTMwNGM0MTdjYTQ2NDc1Y2VlZmEyZjAyYi5zZXRDb250ZW50KGh0bWxfNTVkMzUwMDcyMDY2NDU3M2E1ZTliMWZiMmYwODA4ODEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfY2Q2MDZjZTg3YjdiNDYyMWIxMzVhY2UzMmMyOTVhMzguYmluZFBvcHVwKHBvcHVwXzYzYzdkNTQxMzA0YzQxN2NhNDY0NzVjZWVmYTJmMDJiKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2MwMzYwYzQ2MjMzZjRjZGZhZGYyZThkYzZiOTZlZDlkID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMzMuMDQ3OTg1OCwgMzQuNjkyMzU0M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDY2LjgzMTI1MDgwMDAwMDEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzYwZGUyYWFiN2U4ODRmODZiMmMxOGNiZmM4ZTIyMjYyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wYjQyZDRmZDYwNTk0Yjk5OTgwYWM5MzE3NjdiNTZmOSA9ICQoYDxkaXYgaWQ9Imh0bWxfMGI0MmQ0ZmQ2MDU5NGI5OTk4MGFjOTMxNzY3YjU2ZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxlYmFub248L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjBkZTJhYWI3ZTg4NGY4NmIyYzE4Y2JmYzhlMjIyNjIuc2V0Q29udGVudChodG1sXzBiNDJkNGZkNjA1OTRiOTk5ODBhYzkzMTc2N2I1NmY5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2MwMzYwYzQ2MjMzZjRjZGZhZGYyZThkYzZiOTZlZDlkLmJpbmRQb3B1cChwb3B1cF82MGRlMmFhYjdlODg0Zjg2YjJjMThjYmZjOGUyMjI2MikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8zNTIzODY5MjhjZjk0ZGY2YWYwNTQ0ZWI4ZjE0NThiOSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQuMTU1NTkwNywgOC41NTE5ODYxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAyMS44NDI1MDEyLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8xMGU1MmNiNzkzMzM0OWE1Yjc5ZDE0NjlkYzFkNjhkZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYjFiNzg0MWZiZThkNDk3YjgwOTBhNjE1M2MzMzQwZTggPSAkKGA8ZGl2IGlkPSJodG1sX2IxYjc4NDFmYmU4ZDQ5N2I4MDkwYTYxNTNjMzM0MGU4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MaWJlcmlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzEwZTUyY2I3OTMzMzQ5YTViNzlkMTQ2OWRjMWQ2OGRmLnNldENvbnRlbnQoaHRtbF9iMWI3ODQxZmJlOGQ0OTdiODA5MGE2MTUzYzMzNDBlOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8zNTIzODY5MjhjZjk0ZGY2YWYwNTQ0ZWI4ZjE0NThiOS5iaW5kUG9wdXAocG9wdXBfMTBlNTJjYjc5MzMzNDlhNWI3OWQxNDY5ZGMxZDY4ZGYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfOWRlYjYyNmVhZGRkNDE3Y2E4NDNiZDNmNzU2MTVkZmIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxOS41MDA4MTM4LCAzMy4zNTQ1ODk4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA2MjAuNDkxNDM0NCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMmM4ZGMxZjY2NDEwNGZlZWFjYjAxMzZhNWNkNWM0NTIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzUwM2MzOGFhNGUyZTQwMGZhZDcyNDE3ZTUxM2UzOGMxID0gJChgPGRpdiBpZD0iaHRtbF81MDNjMzhhYTRlMmU0MDBmYWQ3MjQxN2U1MTNlMzhjMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGlieWE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMmM4ZGMxZjY2NDEwNGZlZWFjYjAxMzZhNWNkNWM0NTIuc2V0Q29udGVudChodG1sXzUwM2MzOGFhNGUyZTQwMGZhZDcyNDE3ZTUxM2UzOGMxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzlkZWI2MjZlYWRkZDQxN2NhODQzYmQzZjc1NjE1ZGZiLmJpbmRQb3B1cChwb3B1cF8yYzhkYzFmNjY0MTA0ZmVlYWNiMDEzNmE1Y2Q1YzQ1MikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV84Zjc3ZThmMjc5Y2I0YjliYjgzYmZmYTM3ZDMxMDVhNCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQ5LjQ0Nzg1MzksIDUwLjE4Mjc5NThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDIwODkuNzgxMTY5OTk5OTk5NywgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTg1ZTIxODlkYmQ3NDNlNWJkOGZkNmM4ZTg0YmUxMzAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzYzMmE0NTI4ODdhMTQxZTc4M2YwZTcwOTU1M2EwYzg2ID0gJChgPGRpdiBpZD0iaHRtbF82MzJhNDUyODg3YTE0MWU3ODNmMGU3MDk1NTNhMGM4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+THV4ZW1ib3VyZzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lODVlMjE4OWRiZDc0M2U1YmQ4ZmQ2YzhlODRiZTEzMC5zZXRDb250ZW50KGh0bWxfNjMyYTQ1Mjg4N2ExNDFlNzgzZjBlNzA5NTUzYTBjODYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfOGY3N2U4ZjI3OWNiNGI5YmI4M2JmZmEzN2QzMTA1YTQuYmluZFBvcHVwKHBvcHVwX2U4NWUyMTg5ZGJkNzQzZTViZDhmZDZjOGU4NGJlMTMwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzcxNWQ1ODcyNjQ1ZDQ2ZGM5ZGFmNjRkYzU1OWFjMzU3ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTI1Ljc4NDAyMSwgLTExLjczMjg4OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTEuMjk5MjgwNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMWNkMzhlMGQ4ZjBmNDhlN2EyYjhkZmViYmQzM2VjYTMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2I0NjBjNTZjZjZiZDRkMmM5ZGM2ZWY5YmFmZTlmNGJiID0gJChgPGRpdiBpZD0iaHRtbF9iNDYwYzU2Y2Y2YmQ0ZDJjOWRjNmVmOWJhZmU5ZjRiYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFkYWdhc2NhcjwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8xY2QzOGUwZDhmMGY0OGU3YTJiOGRmZWJiZDMzZWNhMy5zZXRDb250ZW50KGh0bWxfYjQ2MGM1NmNmNmJkNGQyYzlkYzZlZjliYWZlOWY0YmIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNzE1ZDU4NzI2NDVkNDZkYzlkYWY2NGRjNTU5YWMzNTcuYmluZFBvcHVwKHBvcHVwXzFjZDM4ZTBkOGYwZjQ4ZTdhMmI4ZGZlYmJkMzNlY2EzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzEzMWM2NWFjOTBjODQ3MzdhOGQ3NjM2MTc1OGFhMjlhID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTAuMTQ3ODExLCAyNS4wMDEwODRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDcuOTk1NjM3OTAwMDAwMDAxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hZGI5YjI3ZTk0Mzg0NWQ2YjQ0ZWM4OGI5N2U3NDhlYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYjFhMDcwYTg0M2IyNDY3MTllMjZkMGIyMTYyZjRkMmYgPSAkKGA8ZGl2IGlkPSJodG1sX2IxYTA3MGE4NDNiMjQ2NzE5ZTI2ZDBiMjE2MmY0ZDJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYWxpPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FkYjliMjdlOTQzODQ1ZDZiNDRlYzg4Yjk3ZTc0OGVhLnNldENvbnRlbnQoaHRtbF9iMWEwNzBhODQzYjI0NjcxOWUyNmQwYjIxNjJmNGQyZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8xMzFjNjVhYzkwYzg0NzM3YThkNzYzNjE3NThhYTI5YS5iaW5kUG9wdXAocG9wdXBfYWRiOWIyN2U5NDM4NDVkNmI0NGVjODhiOTdlNzQ4ZWEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfOGM2OGJhNTYyZDM3NDYxMTg0NmIzZWRkZDAzYWZmMGIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszNS42MDI5Njk2LCAzNi4yODUyNzA2XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA2MDMuNDMyMTYzOSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfODQ1MjA3MTI1ZTk0NDc3NjhmNGJjMjM3NjJjODAzNDUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzJmZDFiOTA1NzI3NTQ0MzlhNmIxMDI5MTNhMjU5MWQzID0gJChgPGRpdiBpZD0iaHRtbF8yZmQxYjkwNTcyNzU0NDM5YTZiMTAyOTEzYTI1OTFkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFsdGE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfODQ1MjA3MTI1ZTk0NDc3NjhmNGJjMjM3NjJjODAzNDUuc2V0Q29udGVudChodG1sXzJmZDFiOTA1NzI3NTQ0MzlhNmIxMDI5MTNhMjU5MWQzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzhjNjhiYTU2MmQzNzQ2MTE4NDZiM2VkZGQwM2FmZjBiLmJpbmRQb3B1cChwb3B1cF84NDUyMDcxMjVlOTQ0Nzc2OGY0YmMyMzc2MmM4MDM0NSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xMzVmOGI0MzBhZTQ0NWQxYTljNzVkOWRjN2YwMmRiNiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzE0LjcyMDk5MDksIDI3LjMxNDk0Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNjIuNzIyNTQ2NCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzRmYWUwYzY1NzRmNDljNTllMDIxYmE5MmY5YjE5YjMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzAzODA4Yjg0OTYzOTQyMjI4NTRjODNlOWI0ODkyMzlkID0gJChgPGRpdiBpZD0iaHRtbF8wMzgwOGI4NDk2Mzk0MjIyODU0YzgzZTliNDg5MjM5ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWF1cml0YW5pYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jNGZhZTBjNjU3NGY0OWM1OWUwMjFiYTkyZjliMTliMy5zZXRDb250ZW50KGh0bWxfMDM4MDhiODQ5NjM5NDIyMjg1NGM4M2U5YjQ4OTIzOWQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMTM1ZjhiNDMwYWU0NDVkMWE5Yzc1ZDlkYzdmMDJkYjYuYmluZFBvcHVwKHBvcHVwX2M0ZmFlMGM2NTc0ZjQ5YzU5ZTAyMWJhOTJmOWIxOWIzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzJiZGI2ZGRkYjBiZDQxMmZhN2VmZmRkZThhNDY3YzVhID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTIwLjcyNSwgLTEwLjEzOF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMzEyLjcwNzA4MTcsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzE4ZjU0MjhiMjVlODQ5ZDY4NzM1OTY1NzY1MmVlMTQ3ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mZGQ3NDg1ZDAyNmM0MzBhODdiZWVhMWNlM2NkMzdkYyA9ICQoYDxkaXYgaWQ9Imh0bWxfZmRkNzQ4NWQwMjZjNDMwYTg3YmVlYTFjZTNjZDM3ZGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hdXJpdGl1czwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8xOGY1NDI4YjI1ZTg0OWQ2ODczNTk2NTc2NTJlZTE0Ny5zZXRDb250ZW50KGh0bWxfZmRkNzQ4NWQwMjZjNDMwYTg3YmVlYTFjZTNjZDM3ZGMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMmJkYjZkZGRiMGJkNDEyZmE3ZWZmZGRlOGE0NjdjNWEuYmluZFBvcHVwKHBvcHVwXzE4ZjU0MjhiMjVlODQ5ZDY4NzM1OTY1NzY1MmVlMTQ3KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2ZlY2EwN2ZlNGNjZTQzNmViMTIzMTU0ZDFhNmExZWZkID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTQuMzg4NjI0MywgMzIuNzE4NjU1M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMzg3LjYxMDc2MTEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzc2NGRkMjc5MWMyODQ4NWU4OGQ5ODMwZDI3YjZjYmYyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mYzdlOTU5ZjEyOGU0MGYyYmRmMTNmN2NjOTdiZDYxNyA9ICQoYDxkaXYgaWQ9Imh0bWxfZmM3ZTk1OWYxMjhlNDBmMmJkZjEzZjdjYzk3YmQ2MTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1leGljbzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF83NjRkZDI3OTFjMjg0ODVlODhkOTgzMGQyN2I2Y2JmMi5zZXRDb250ZW50KGh0bWxfZmM3ZTk1OWYxMjhlNDBmMmJkZjEzZjdjYzk3YmQ2MTcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZmVjYTA3ZmU0Y2NlNDM2ZWIxMjMxNTRkMWE2YTFlZmQuYmluZFBvcHVwKHBvcHVwXzc2NGRkMjc5MWMyODQ4NWU4OGQ5ODMwZDI3YjZjYmYyKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzY2ODc1ZTc2NzIwZDQ2NjVhYTBjYzdhYTIwYWViOTIxID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNDEuNTgwMDI3NiwgNTIuMTQ5Nl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNjkxLjUxNDY1MzEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzljNGYyOTQ1MmE2MTQ0NzVhNDliNzAzZmQ5ZDc3MzhkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mYzIwNmJkODU5MDk0MDJmYmM0ZWZkZWE3NDFkZDJhNSA9ICQoYDxkaXYgaWQ9Imh0bWxfZmMyMDZiZDg1OTA5NDAyZmJjNGVmZGVhNzQxZGQyYTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vbmdvbGlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzljNGYyOTQ1MmE2MTQ0NzVhNDliNzAzZmQ5ZDc3MzhkLnNldENvbnRlbnQoaHRtbF9mYzIwNmJkODU5MDk0MDJmYmM0ZWZkZWE3NDFkZDJhNSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV82Njg3NWU3NjcyMGQ0NjY1YWEwY2M3YWEyMGFlYjkyMS5iaW5kUG9wdXAocG9wdXBfOWM0ZjI5NDUyYTYxNDQ3NWE0OWI3MDNmZDlkNzczOGQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMDI1N2FkNzAyNTRiNGQyNmFmZWQ5NDVmMzAyNTMyNjQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsyMS4zMzY1MzIxLCAzNi4wNTA1MjY5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNzMuNzkxNTU0Njk5OTk5OTgsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzE0NzVkZDcxZjliYzQyMmJiMzcyYjQ4NDVhNGU0MWQ4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zNzRhYjk3YWU4Njk0MWI2YWJkYzA3OTM1MjNhZmNjZSA9ICQoYDxkaXYgaWQ9Imh0bWxfMzc0YWI5N2FlODY5NDFiNmFiZGMwNzkzNTIzYWZjY2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vcm9jY288L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMTQ3NWRkNzFmOWJjNDIyYmIzNzJiNDg0NWE0ZTQxZDguc2V0Q29udGVudChodG1sXzM3NGFiOTdhZTg2OTQxYjZhYmRjMDc5MzUyM2FmY2NlKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzAyNTdhZDcwMjU0YjRkMjZhZmVkOTQ1ZjMwMjUzMjY0LmJpbmRQb3B1cChwb3B1cF8xNDc1ZGQ3MWY5YmM0MjJiYjM3MmI0ODQ1YTRlNDFkOCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9iMGJiYjdjMGVmYmY0YTk3YjQxZDM4N2Q4MjMwNGQxMiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0yNi45MjA5NDI3LCAtMTAuMzI1MjE0OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTMuMTE4OTgwNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzgyODkyOGM4NDY0NGYwZWFlNGRhZGFlZjgzYzQ0MzYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2I4MWM2NTIyZDY0YjQ1OWFhODA1ZGUxNTRiN2UwYmZkID0gJChgPGRpdiBpZD0iaHRtbF9iODFjNjUyMmQ2NGI0NTlhYTgwNWRlMTU0YjdlMGJmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TW96YW1iaXF1ZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jODI4OTI4Yzg0NjQ0ZjBlYWU0ZGFkYWVmODNjNDQzNi5zZXRDb250ZW50KGh0bWxfYjgxYzY1MjJkNjRiNDU5YWE4MDVkZTE1NGI3ZTBiZmQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfYjBiYmI3YzBlZmJmNGE5N2I0MWQzODdkODIzMDRkMTIuYmluZFBvcHVwKHBvcHVwX2M4Mjg5MjhjODQ2NDRmMGVhZTRkYWRhZWY4M2M0NDM2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzI0N2YxZDA5YjE0OTQyNDJhNjFhODcwMDVhNWQ0YWUyID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbOS40Mzk5NDMyLCAyOC41NDc4MzVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDIwLjAyODUyMTIsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2RjZmM4YWJlOWY5MDQyMmFiMTRhYjMzYWE3NjQxNGMzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82ZjE0Y2I3NDA3YjU0YzhiYTg3ZjJiYjNmMDUxNDdlZSA9ICQoYDxkaXYgaWQ9Imh0bWxfNmYxNGNiNzQwN2I1NGM4YmE4N2YyYmIzZjA1MTQ3ZWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk15YW5tYXI8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZGNmYzhhYmU5ZjkwNDIyYWIxNGFiMzNhYTc2NDE0YzMuc2V0Q29udGVudChodG1sXzZmMTRjYjc0MDdiNTRjOGJhODdmMmJiM2YwNTE0N2VlKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzI0N2YxZDA5YjE0OTQyNDJhNjFhODcwMDVhNWQ0YWUyLmJpbmRQb3B1cChwb3B1cF9kY2ZjOGFiZTlmOTA0MjJhYjE0YWIzM2FhNzY0MTRjMykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8zMWQ0MTYzOTUwYjQ0NmY1OWM2NTFlMTkyNGFiM2UzZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzI2LjM0Nzc1ODEsIDMwLjQ0Njk0NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTUuOTQ3NDQ1NDk5OTk5OTk5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF83MDdiNzIzYWFjZGQ0M2E4YTRlYzVjMTVmNDJhM2IxMCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNmNhYWQ3ZTAzZjU5NDZlMmFlOGFkNDcwMDU2OTNjZDUgPSAkKGA8ZGl2IGlkPSJodG1sXzZjYWFkN2UwM2Y1OTQ2ZTJhZThhZDQ3MDA1NjkzY2Q1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OZXBhbDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF83MDdiNzIzYWFjZGQ0M2E4YTRlYzVjMTVmNDJhM2IxMC5zZXRDb250ZW50KGh0bWxfNmNhYWQ3ZTAzZjU5NDZlMmFlOGFkNDcwMDU2OTNjZDUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMzFkNDE2Mzk1MGI0NDZmNTljNjUxZTE5MjRhYjNlM2UuYmluZFBvcHVwKHBvcHVwXzcwN2I3MjNhYWNkZDQzYThhNGVjNWMxNWY0MmEzYjEwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2I3MDIyMjRiY2YxMzQxMzE4OGViMGE1NmMxMDk4ZDE4ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMTEuNzc3LCA1My43NDQzODldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDEwMDYuNDQ4OTc0MDAwMDAwMSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfN2I5ODdiOTk2MWMyNDExZmJmMTQwYjgyMTZlY2ExODUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FmOTMxYWZkZmI4OTQxZWI4ZDY2OWQ2ODRiNWU5OTZjID0gJChgPGRpdiBpZD0iaHRtbF9hZjkzMWFmZGZiODk0MWViOGQ2NjlkNjg0YjVlOTk2YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV0aGVybGFuZHM8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfN2I5ODdiOTk2MWMyNDExZmJmMTQwYjgyMTZlY2ExODUuc2V0Q29udGVudChodG1sX2FmOTMxYWZkZmI4OTQxZWI4ZDY2OWQ2ODRiNWU5OTZjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2I3MDIyMjRiY2YxMzQxMzE4OGViMGE1NmMxMDk4ZDE4LmJpbmRQb3B1cChwb3B1cF83Yjk4N2I5OTYxYzI0MTFmYmYxNDBiODIxNmVjYTE4NSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV80ZWY1Yjk1MjlkNGU0NDRhYTk3MjM1OTMwYWJhNTZhZiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy01Mi44MjEzNjg3LCAtMjkuMDMwMzMwM10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNzEyLjQwNTA4NjcsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQ1YzlmZDhkMTJiMTRiNzliMzM0ZGFlN2Y4NTUxZGYxID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xMzI1ZjI4ODk5NTA0NTE4YTk2MDI2N2JmODRkOTliMSA9ICQoYDxkaXYgaWQ9Imh0bWxfMTMyNWYyODg5OTUwNDUxOGE5NjAyNjdiZjg0ZDk5YjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5ldyBaZWFsYW5kPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzQ1YzlmZDhkMTJiMTRiNzliMzM0ZGFlN2Y4NTUxZGYxLnNldENvbnRlbnQoaHRtbF8xMzI1ZjI4ODk5NTA0NTE4YTk2MDI2N2JmODRkOTliMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV80ZWY1Yjk1MjlkNGU0NDRhYTk3MjM1OTMwYWJhNTZhZi5iaW5kUG9wdXAocG9wdXBfNDVjOWZkOGQxMmIxNGI3OWIzMzRkYWU3Zjg1NTFkZjEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfY2E1YjZiNzEwNzBhNDliM2FlZDM3MjhlZTAyMmE5NmIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMC43MDc2NTY1LCAxNS4wMzMxMTgzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA4NC4zNTQxOTU0OTk5OTk5OSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzU4ZjI0MjAwOWNmNDJjM2JjM2M3YjM1YmJkMjAxMTEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzkzMjU4OTJiNDJjZTRkNjViMDQzNTRlZWY0Zjk1NmMzID0gJChgPGRpdiBpZD0iaHRtbF85MzI1ODkyYjQyY2U0ZDY1YjA0MzU0ZWVmNGY5NTZjMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmljYXJhZ3VhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M1OGYyNDIwMDljZjQyYzNiYzNjN2IzNWJiZDIwMTExLnNldENvbnRlbnQoaHRtbF85MzI1ODkyYjQyY2U0ZDY1YjA0MzU0ZWVmNGY5NTZjMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9jYTViNmI3MTA3MGE0OWIzYWVkMzcyOGVlMDIyYTk2Yi5iaW5kUG9wdXAocG9wdXBfYzU4ZjI0MjAwOWNmNDJjM2JjM2M3YjM1YmJkMjAxMTEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNzdiOTFjMzFjN2IwNGQ0OWFjNDkxYmYyMjI0N2VmYzAgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMS42OTM3NTYsIDIzLjUxNzE3OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogOC4zOTU4MTc4MDAwMDAwMDEsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzM5MjFkM2VmMTc5ODQ0ZjY5N2Q3NDZjZWQzZTQzYTc0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xNDAzZTRlOWQxMmE0NzEzYTk0MzJmYjg3OTFiZWM2YiA9ICQoYDxkaXYgaWQ9Imh0bWxfMTQwM2U0ZTlkMTJhNDcxM2E5NDMyZmI4NzkxYmVjNmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5pZ2VyPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzM5MjFkM2VmMTc5ODQ0ZjY5N2Q3NDZjZWQzZTQzYTc0LnNldENvbnRlbnQoaHRtbF8xNDAzZTRlOWQxMmE0NzEzYTk0MzJmYjg3OTFiZWM2Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV83N2I5MWMzMWM3YjA0ZDQ5YWM0OTFiZjIyMjQ3ZWZjMC5iaW5kUG9wdXAocG9wdXBfMzkyMWQzZWYxNzk4NDRmNjk3ZDc0NmNlZDNlNDNhNzQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfOTI2ZDQ0Mjk3ZWQxNDg1Y2IyZjBlZjcyNGQxODY5ZDUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0LjA2OTA5NTksIDEzLjg4NTY0NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNTMuNzQ5NzU2MDAwMDAwMDA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mYTA5NGQwOGU5MjE0MzM4OWU1MGRhZWZjYTg1ZGZjNyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjYyYmE0MzYxYzA0NDcwNjg3ZmQ4MDNiNzg3NmZiY2UgPSAkKGA8ZGl2IGlkPSJodG1sXzY2MmJhNDM2MWMwNDQ3MDY4N2ZkODAzYjc4NzZmYmNlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OaWdlcmlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2ZhMDk0ZDA4ZTkyMTQzMzg5ZTUwZGFlZmNhODVkZmM3LnNldENvbnRlbnQoaHRtbF82NjJiYTQzNjFjMDQ0NzA2ODdmZDgwM2I3ODc2ZmJjZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV85MjZkNDQyOTdlZDE0ODVjYjJmMGVmNzI0ZDE4NjlkNS5iaW5kUG9wdXAocG9wdXBfZmEwOTRkMDhlOTIxNDMzODllNTBkYWVmY2E4NWRmYzcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZTRkNmU0OGZlMWI1NGEzN2JlYTRlYWRkNGJkMzM2YmUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstNTQuNjU0LCA4MS4wMjgwNzZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDkxOS4yODc5MDY5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mZTJkZWI4NTZjY2Q0ZjgwOGYwN2IwZjU5MGNmMDBjZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNzBiYmVlYmFkMmNlNDA3NTljY2EyODI0MjQ2ZTNlODAgPSAkKGA8ZGl2IGlkPSJodG1sXzcwYmJlZWJhZDJjZTQwNzU5Y2NhMjgyNDI0NmUzZTgwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J3YXk8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZmUyZGViODU2Y2NkNGY4MDhmMDdiMGY1OTBjZjAwY2Uuc2V0Q29udGVudChodG1sXzcwYmJlZWJhZDJjZTQwNzU5Y2NhMjgyNDI0NmUzZTgwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2U0ZDZlNDhmZTFiNTRhMzdiZWE0ZWFkZDRiZDMzNmJlLmJpbmRQb3B1cChwb3B1cF9mZTJkZWI4NTZjY2Q0ZjgwOGYwN2IwZjU5MGNmMDBjZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xMDU5MmZhODE3OGM0NTBjYmQxNmI5MDgzYmJhYzdkZiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzIzLjUzOTM5MTYsIDM3LjA4NDEwN10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogOTQuMTE3MTEyNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMmIwZTcyNGRhODRlNDUxODhlZmUwNzY4MDBkZDJkYTkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2UwNTAyYTUzYzMyNjQxOTU4OTFiYTU0OWRlNjEyZTc4ID0gJChgPGRpdiBpZD0iaHRtbF9lMDUwMmE1M2MzMjY0MTk1ODkxYmE1NDlkZTYxMmU3OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFraXN0YW48L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMmIwZTcyNGRhODRlNDUxODhlZmUwNzY4MDBkZDJkYTkuc2V0Q29udGVudChodG1sX2UwNTAyYTUzYzMyNjQxOTU4OTFiYTU0OWRlNjEyZTc4KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzEwNTkyZmE4MTc4YzQ1MGNiZDE2YjkwODNiYmFjN2RmLmJpbmRQb3B1cChwb3B1cF8yYjBlNzI0ZGE4NGU0NTE4OGVmZTA3NjgwMGRkMmRhOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV84MTg4MmQ4ZmJhMDk0NDMwOWY5MjczZDczMjYyMzFhMyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzIuNzQ4LCA4LjIyMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTA4NS41NDMwNDYsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2E3MTJiNWQyMWE2ZjRkODg4MGQyYTdiOTE1NGZhYjI4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82NmNhYjUyNDhlMjE0ZjY0YjE2ODc2YjI4MjlmOTFhYSA9ICQoYDxkaXYgaWQ9Imh0bWxfNjZjYWI1MjQ4ZTIxNGY2NGIxNjg3NmIyODI5ZjkxYWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhbGF1PC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2E3MTJiNWQyMWE2ZjRkODg4MGQyYTdiOTE1NGZhYjI4LnNldENvbnRlbnQoaHRtbF82NmNhYjUyNDhlMjE0ZjY0YjE2ODc2YjI4MjlmOTFhYSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV84MTg4MmQ4ZmJhMDk0NDMwOWY5MjczZDczMjYyMzFhMy5iaW5kUG9wdXAocG9wdXBfYTcxMmI1ZDIxYTZmNGQ4ODgwZDJhN2I5MTU0ZmFiMjgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfODU3NDIzMzY1ZDU1NDNkYzk4N2EzZWM0NjNlZDZjNDEgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs3LjAzMzg2NzksIDkuODcwMTc1N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjYyLjUyNzYyNDQwMDAwMDA0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81YzMyNDc4YWU4YTQ0MzIzOGY1MGY5Mjc0NjA4MWNlZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNzUxYTM4MzU4ODc2NDlmYjgxNWRiZTIzMjEyODhjNmQgPSAkKGA8ZGl2IGlkPSJodG1sXzc1MWEzODM1ODg3NjQ5ZmI4MTVkYmUyMzIxMjg4YzZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYW5hbWE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWMzMjQ3OGFlOGE0NDMyMzhmNTBmOTI3NDYwODFjZWQuc2V0Q29udGVudChodG1sXzc1MWEzODM1ODg3NjQ5ZmI4MTVkYmUyMzIxMjg4YzZkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzg1NzQyMzM2NWQ1NTQzZGM5ODdhM2VjNDYzZWQ2YzQxLmJpbmRQb3B1cChwb3B1cF81YzMyNDc4YWU4YTQ0MzIzOGY1MGY5Mjc0NjA4MWNlZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV82MjY0ODdjNDA5ZTY0ZDlkYmUwZjZiNGZlY2FiMzU4NCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0xMS44NTU1NzM5LCAtMC41NTczNTc2XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA3NC42ODk1MjM0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80MjIwMjYyMTU3M2M0NmY1OTA2NGUwMDNkM2I0MDVmNSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZDJiNTc1MjY3OTkzNDA4MmI5ODg4NTIxYjk4MThhZDcgPSAkKGA8ZGl2IGlkPSJodG1sX2QyYjU3NTI2Nzk5MzQwODJiOTg4ODUyMWI5ODE4YWQ3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXB1YSBOZXcgR3VpbmVhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzQyMjAyNjIxNTczYzQ2ZjU5MDY0ZTAwM2QzYjQwNWY1LnNldENvbnRlbnQoaHRtbF9kMmI1NzUyNjc5OTM0MDgyYjk4ODg1MjFiOTgxOGFkNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV82MjY0ODdjNDA5ZTY0ZDlkYmUwZjZiNGZlY2FiMzU4NC5iaW5kUG9wdXAocG9wdXBfNDIyMDI2MjE1NzNjNDZmNTkwNjRlMDAzZDNiNDA1ZjUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNDZkYzZkMWE4MDk3NGM5M2JhMzE2YzFjYjI3MDdhYTcgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMjcuNjA2MzkzNSwgLTE5LjI4NzY0NzJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDg0LjE5MTUyNDIsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2JkY2YzODMxZjlkZTQ4NzE4MzAyMzk1ZGUyYTU4ZGM5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zMWQ5MDQyMTY2YTI0OWQyODYwNzM0ZWU4ZDI1OTczOCA9ICQoYDxkaXYgaWQ9Imh0bWxfMzFkOTA0MjE2NmEyNDlkMjg2MDczNGVlOGQyNTk3MzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmFndWF5PC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2JkY2YzODMxZjlkZTQ4NzE4MzAyMzk1ZGUyYTU4ZGM5LnNldENvbnRlbnQoaHRtbF8zMWQ5MDQyMTY2YTI0OWQyODYwNzM0ZWU4ZDI1OTczOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV80NmRjNmQxYTgwOTc0YzkzYmEzMTZjMWNiMjcwN2FhNy5iaW5kUG9wdXAocG9wdXBfYmRjZjM4MzFmOWRlNDg3MTgzMDIzOTVkZTJhNThkYzkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZGVmZTk1YzBjZWExNGYxOThlMzFlY2QzM2Y2ODliOTIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMjAuMTk4NDQ3MiwgLTAuMDM5MjgxOF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTc4LjMyMzMxNDQsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2MyNjZjOWU5ZThmMDQ5MmY5MGJiMDRjYzlkMjUzMTNiID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF83MTc0YzlmNDUwMDA0NWQ3YjNkN2ExZmY2N2U3OTdlMSA9ICQoYDxkaXYgaWQ9Imh0bWxfNzE3NGM5ZjQ1MDAwNDVkN2IzZDdhMWZmNjdlNzk3ZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBlcnU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzI2NmM5ZTllOGYwNDkyZjkwYmIwNGNjOWQyNTMxM2Iuc2V0Q29udGVudChodG1sXzcxNzRjOWY0NTAwMDQ1ZDdiM2Q3YTFmZjY3ZTc5N2UxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2RlZmU5NWMwY2VhMTRmMTk4ZTMxZWNkMzNmNjg5YjkyLmJpbmRQb3B1cChwb3B1cF9jMjY2YzllOWU4ZjA0OTJmOTBiYjA0Y2M5ZDI1MzEzYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV80YjBhNjhiMDgzYjE0Yzc3YTVhNGM2ZjkxMDNmNjcyNSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQ5LjAwMjA0NjgsIDU1LjAzMzY5NjNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDgzMy41Nzg2NzA2LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lNDM2NDZmMzZiM2M0YjRiYTI2NGE0NTU3MDRmMDJlNSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYzVmMGQ0YTI0Y2Y0NDZmNDhjNTQ2ZWZlYjJkN2M5OWQgPSAkKGA8ZGl2IGlkPSJodG1sX2M1ZjBkNGEyNGNmNDQ2ZjQ4YzU0NmVmZWIyZDdjOTlkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Qb2xhbmQ8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTQzNjQ2ZjM2YjNjNGI0YmEyNjRhNDU1NzA0ZjAyZTUuc2V0Q29udGVudChodG1sX2M1ZjBkNGEyNGNmNDQ2ZjQ4YzU0NmVmZWIyZDdjOTlkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzRiMGE2OGIwODNiMTRjNzdhNWE0YzZmOTEwM2Y2NzI1LmJpbmRQb3B1cChwb3B1cF9lNDM2NDZmMzZiM2M0YjRiYTI2NGE0NTU3MDRmMDJlNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9jZDJmN2Q4ZGVlYWE0NjliYjE5N2NjYmM5MTdiYzliYSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzI5LjgyODgwMjEsIDQyLjE1NDMxMTJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQ3MC45ODQ5NjIzOTk5OTk5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84MmVjYmY3YTZjYTQ0MDhkOWM3NTVmYTNkOWIwODllNiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNzdhZTI5NzhmNWU0NDMxMjgzMjU5NWRjYThkZDMzYmIgPSAkKGA8ZGl2IGlkPSJodG1sXzc3YWUyOTc4ZjVlNDQzMTI4MzI1OTVkY2E4ZGQzM2JiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Qb3J0dWdhbDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84MmVjYmY3YTZjYTQ0MDhkOWM3NTVmYTNkOWIwODllNi5zZXRDb250ZW50KGh0bWxfNzdhZTI5NzhmNWU0NDMxMjgzMjU5NWRjYThkZDMzYmIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfY2QyZjdkOGRlZWFhNDY5YmIxOTdjY2JjOTE3YmM5YmEuYmluZFBvcHVwKHBvcHVwXzgyZWNiZjdhNmNhNDQwOGQ5Yzc1NWZhM2Q5YjA4OWU2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzIzZTgzNjIyMjlkNjRmYzJiNjVjYTU5OTc1NTIxOGZkID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMjQuNDcwNzUzNCwgMjYuMzgzMDIxMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDQwMS44OTI2MzcsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2MyMDRhZjg3ZTY2YjQzMzRhMjhlYTZiN2EwMTBmMmIwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hNmQwZjczOWMxZTY0MjM0YTFlMTNkNGFhYjJmZmY4YiA9ICQoYDxkaXYgaWQ9Imh0bWxfYTZkMGY3MzljMWU2NDIzNGExZTEzZDRhYWIyZmZmOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlFhdGFyPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2MyMDRhZjg3ZTY2YjQzMzRhMjhlYTZiN2EwMTBmMmIwLnNldENvbnRlbnQoaHRtbF9hNmQwZjczOWMxZTY0MjM0YTFlMTNkNGFhYjJmZmY4Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8yM2U4MzYyMjI5ZDY0ZmMyYjY1Y2E1OTk3NTUyMThmZC5iaW5kUG9wdXAocG9wdXBfYzIwNGFmODdlNjZiNDMzNGEyOGVhNmI3YTAxMGYyYjApCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMWZjMTgwNDk3YzljNDE4MTk3NTU3ZWU1ODEyMDFiN2IgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0My42MTg2ODIsIDQ4LjI2NTM5NjRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQyMS4wNTYwMTI0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jNTgwMGJjNTBjYzc0MTU3OTg3NGEyOTM0ZjdiYzEzZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjJiZWE3YjVmZjVhNDE2ODg3YjZlMzRmZWFiNzI3NmQgPSAkKGA8ZGl2IGlkPSJodG1sXzYyYmVhN2I1ZmY1YTQxNjg4N2I2ZTM0ZmVhYjcyNzZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb21hbmlhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M1ODAwYmM1MGNjNzQxNTc5ODc0YTI5MzRmN2JjMTNmLnNldENvbnRlbnQoaHRtbF82MmJlYTdiNWZmNWE0MTY4ODdiNmUzNGZlYWI3Mjc2ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8xZmMxODA0OTdjOWM0MTgxOTc1NTdlZTU4MTIwMWI3Yi5iaW5kUG9wdXAocG9wdXBfYzU4MDBiYzUwY2M3NDE1Nzk4NzRhMjkzNGY3YmMxM2YpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZGQ4Njk2ZmNhZTkyNDY5ZTk4YTAzYjllMjNlNzQ1NTQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMi44Mzk3NTgxLCAtMS4wNDc0MDgzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA2LjI4NzQxOTcwMDAwMDAwMSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWMyMThjMGZiMzIzNDI3YmJhNDMyMTU0NGRiM2I2MDMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzU2OGYxYWM4NDRlYjRkMTFiYjgxNjU4NWE3MzI5OTQzID0gJChgPGRpdiBpZD0iaHRtbF81NjhmMWFjODQ0ZWI0ZDExYmI4MTY1ODVhNzMyOTk0MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UndhbmRhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzVjMjE4YzBmYjMyMzQyN2JiYTQzMjE1NDRkYjNiNjAzLnNldENvbnRlbnQoaHRtbF81NjhmMWFjODQ0ZWI0ZDExYmI4MTY1ODVhNzMyOTk0Myk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kZDg2OTZmY2FlOTI0NjllOThhMDNiOWUyM2U3NDU1NC5iaW5kUG9wdXAocG9wdXBfNWMyMThjMGZiMzIzNDI3YmJhNDMyMTU0NGRiM2I2MDMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZTM2NTVlNDQ5NDk2NDE5NjljN2I1MWFmZTlmMTdkODEgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMTQuMjc3MDkxNiwgLTEzLjIzODE4OTJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDEyNS4yMTEwMDc1OTk5OTk5OSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMDBlYTZiYTFjNWNlNDA3NzkyMDFhMzgzMmRkNjZlMWIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2M3NDU4N2ZiMTQ1MjRmYjE5M2Y3NDJlZDMzOTE3OWViID0gJChgPGRpdiBpZD0iaHRtbF9jNzQ1ODdmYjE0NTI0ZmIxOTNmNzQyZWQzMzkxNzllYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2Ftb2E8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMDBlYTZiYTFjNWNlNDA3NzkyMDFhMzgzMmRkNjZlMWIuc2V0Q29udGVudChodG1sX2M3NDU4N2ZiMTQ1MjRmYjE5M2Y3NDJlZDMzOTE3OWViKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2UzNjU1ZTQ0OTQ5NjQxOTY5YzdiNTFhZmU5ZjE3ZDgxLmJpbmRQb3B1cChwb3B1cF8wMGVhNmJhMWM1Y2U0MDc3OTIwMWEzODMyZGQ2NmUxYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9jZjM5NzNjZWQ4ZDI0ZjUzYmI2NThlZDRhYzRhNTVjZCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0wLjIxMzUxMzcsIDEuOTI1NzYwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNTguNzkwOTI1NiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzQ3MTI2ODE1YTgxNDQxMTg2ZjIxNzgzNTAxYjdkNGIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzRmOWZkNzJkYmY1NDQ1NjU4MjU2Y2U1YTA1NjMwOTJmID0gJChgPGRpdiBpZD0iaHRtbF80ZjlmZDcyZGJmNTQ0NTY1ODI1NmNlNWEwNTYzMDkyZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2FvIFRvbWUgYW5kIFByaW5jaXBlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M0NzEyNjgxNWE4MTQ0MTE4NmYyMTc4MzUwMWI3ZDRiLnNldENvbnRlbnQoaHRtbF80ZjlmZDcyZGJmNTQ0NTY1ODI1NmNlNWEwNTYzMDkyZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9jZjM5NzNjZWQ4ZDI0ZjUzYmI2NThlZDRhYzRhNTVjZC5iaW5kUG9wdXAocG9wdXBfYzQ3MTI2ODE1YTgxNDQxMTg2ZjIxNzgzNTAxYjdkNGIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYWM0YjllYzk0YmVjNDA0NDhhNDQ4NmYzMjg2MWYxYWMgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxNi4yOSwgMzIuMTU0MzM3N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTgwNy4yNDUwNzIsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzM4Y2E3NTkwZTNjOTRiOTY4NWQ4Yzg3YjcxNDBjNjk0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85NjM5NzAzNGEzNDM0ODEzOWY1NjMxODExYTI0ZTQ5YyA9ICQoYDxkaXYgaWQ9Imh0bWxfOTYzOTcwMzRhMzQzNDgxMzlmNTYzMTgxMWEyNGU0OWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNhdWRpIEFyYWJpYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8zOGNhNzU5MGUzYzk0Yjk2ODVkOGM4N2I3MTQwYzY5NC5zZXRDb250ZW50KGh0bWxfOTYzOTcwMzRhMzQzNDgxMzlmNTYzMTgxMWEyNGU0OWMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfYWM0YjllYzk0YmVjNDA0NDhhNDQ4NmYzMjg2MWYxYWMuYmluZFBvcHVwKHBvcHVwXzM4Y2E3NTkwZTNjOTRiOTY4NWQ4Yzg3YjcxNDBjNjk0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzMxOTgzMTIwOWJkNDRlZWE5OGYyYmExYjc4Njk5YWQyID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNi43NTUsIDkuOTk5OTczXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNS4yMDQ0MTY4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mOGMzYTEzMWRjYzY0YmIwOWQ2ZGQ0NWZmMmRhNGM5NyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZGY0YWRlYmFjNmM1NGU3NmI4MGJkMGMyZmNjN2U0NTMgPSAkKGA8ZGl2IGlkPSJodG1sX2RmNGFkZWJhYzZjNTRlNzZiODBiZDBjMmZjYzdlNDUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TaWVycmEgTGVvbmU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjhjM2ExMzFkY2M2NGJiMDlkNmRkNDVmZjJkYTRjOTcuc2V0Q29udGVudChodG1sX2RmNGFkZWJhYzZjNTRlNzZiODBiZDBjMmZjYzdlNDUzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzMxOTgzMTIwOWJkNDRlZWE5OGYyYmExYjc4Njk5YWQyLmJpbmRQb3B1cChwb3B1cF9mOGMzYTEzMWRjYzY0YmIwOWQ2ZGQ0NWZmMmRhNGM5NykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9hMmZkZmI1YWMyNzU0Zjc4OGU2YzJjYzMxYzY2YzM2NyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzEuMTMwMzYxMSwgMS41MTMxNjAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA0MzIuMDE2MTQzNjk5OTk5OTMsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2MwMTQ4NDBhODMzZjQ1YWJiMDI1ZjI1NGQ4MmJhNDVhID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xZDA4NmYyMzQ1OGI0ODE0OTQ0ZTcwODYyZjllOWI3NCA9ICQoYDxkaXYgaWQ9Imh0bWxfMWQwODZmMjM0NThiNDgxNDk0NGU3MDg2MmY5ZTliNzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNpbmdhcG9yZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jMDE0ODQwYTgzM2Y0NWFiYjAyNWYyNTRkODJiYTQ1YS5zZXRDb250ZW50KGh0bWxfMWQwODZmMjM0NThiNDgxNDk0NGU3MDg2MmY5ZTliNzQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfYTJmZGZiNWFjMjc1NGY3ODhlNmMyY2MzMWM2NmMzNjcuYmluZFBvcHVwKHBvcHVwX2MwMTQ4NDBhODMzZjQ1YWJiMDI1ZjI1NGQ4MmJhNDVhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzg4OTM4NmZjMGQ2ZTRjZDI4NmUwYzNhZThkOWU1NDNmID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTEzLjI0MjQyOTgsIC00LjgxMDg1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzNi44MzA0MTY4OTk5OTk5OTYsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzIzYzczMmIwNjY2ZDRjODhiOGYxN2EyYmYyNTUzYTI0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jY2ExMjMzZGRjZTE0ODNiYWQ4NTczYzA0NTIyZTQ1NCA9ICQoYDxkaXYgaWQ9Imh0bWxfY2NhMTIzM2RkY2UxNDgzYmFkODU3M2MwNDUyMmU0NTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvbG9tb24gSXNsYW5kczwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yM2M3MzJiMDY2NmQ0Yzg4YjhmMTdhMmJmMjU1M2EyNC5zZXRDb250ZW50KGh0bWxfY2NhMTIzM2RkY2UxNDgzYmFkODU3M2MwNDUyMmU0NTQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfODg5Mzg2ZmMwZDZlNGNkMjg2ZTBjM2FlOGQ5ZTU0M2YuYmluZFBvcHVwKHBvcHVwXzIzYzczMmIwNjY2ZDRjODhiOGYxN2EyYmYyNTUzYTI0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzNjNmM0NGUxNWU3MzRkNmRiMTliZGRmNDQ3YjVkZjU0ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTQ3LjE3ODgzMzUsIC0yMi4xMjUwMzAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA5MjUuNzIxNjQ3MiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzE3ODIzNmE5ODk3NDM5NWI4OTZhZDdjYjI3Y2IxYzEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2YzMzAxYTljNzRkOTRlZGI5Yjk4Y2ZjYTZlZGIwY2ZmID0gJChgPGRpdiBpZD0iaHRtbF9mMzMwMWE5Yzc0ZDk0ZWRiOWI5OGNmY2E2ZWRiMGNmZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggQWZyaWNhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2MxNzgyMzZhOTg5NzQzOTViODk2YWQ3Y2IyN2NiMWMxLnNldENvbnRlbnQoaHRtbF9mMzMwMWE5Yzc0ZDk0ZWRiOWI5OGNmY2E2ZWRiMGNmZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8zYzZjNDRlMTVlNzM0ZDZkYjE5YmRkZjQ0N2I1ZGY1NC5iaW5kUG9wdXAocG9wdXBfYzE3ODIzNmE5ODk3NDM5NWI4OTZhZDdjYjI3Y2IxYzEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNzRhMGI1N2JlZGY1NGFjZTg0ODRlZTg2YmNkNmZkNGYgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsyNy40MzM1NDI2LCA0My45OTMzMDg4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1NzkuMDc2NDIzNCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTgzZjU3YzI1MTY1NDVkOTliMjUwNDFjMzdhNDRiYmUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzA0MDgxYjcyZDRhYzQyMjJhYTc3YzE5MDY2ZDA4NTlhID0gJChgPGRpdiBpZD0iaHRtbF8wNDA4MWI3MmQ0YWM0MjIyYWE3N2MxOTA2NmQwODU5YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3BhaW48L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTgzZjU3YzI1MTY1NDVkOTliMjUwNDFjMzdhNDRiYmUuc2V0Q29udGVudChodG1sXzA0MDgxYjcyZDRhYzQyMjJhYTc3YzE5MDY2ZDA4NTlhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzc0YTBiNTdiZWRmNTRhY2U4NDg0ZWU4NmJjZDZmZDRmLmJpbmRQb3B1cChwb3B1cF9lODNmNTdjMjUxNjU0NWQ5OWIyNTA0MWMzN2E0NGJiZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8zNGU0MzZhZTFkOGE0M2YwYWFmNTJkMzEyNDY1ZTJlOCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzUuNzE5LCAxMC4wMzVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDcyLjk5MjA4NCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfODJkYThkNTM4YzhiNDEzOWJiY2UyOGU3YzhmNjFkOWQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2NiZGI5ZGI0ZDAzMjQ2ZGNiZmUzMTFiNmIzZmRlNmYxID0gJChgPGRpdiBpZD0iaHRtbF9jYmRiOWRiNGQwMzI0NmRjYmZlMzExYjZiM2ZkZTZmMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3JpIExhbmthPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzgyZGE4ZDUzOGM4YjQxMzliYmNlMjhlN2M4ZjYxZDlkLnNldENvbnRlbnQoaHRtbF9jYmRiOWRiNGQwMzI0NmRjYmZlMzExYjZiM2ZkZTZmMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8zNGU0MzZhZTFkOGE0M2YwYWFmNTJkMzEyNDY1ZTJlOC5iaW5kUG9wdXAocG9wdXBfODJkYThkNTM4YzhiNDEzOWJiY2UyOGU3YzhmNjFkOWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYjZhMThjNzI2NTAzNDJjZTk3OTU1NThmMzNjNWQwNGMgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxNi44OTUsIDE3LjYxNTgxNDZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUwNS4wOTY0MTg3LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80ZDM5ZTFhZWJiODk0YjUwOWU5ZTNiZmQzMWUyYzYyMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNWQ5Y2ZkZDRmM2Q3NDVhOTkzNDBlODkyOTdjZjc4MWUgPSAkKGA8ZGl2IGlkPSJodG1sXzVkOWNmZGQ0ZjNkNzQ1YTk5MzQwZTg5Mjk3Y2Y3ODFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gS2l0dHMgYW5kIE5ldmlzPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzRkMzllMWFlYmI4OTRiNTA5ZTllM2JmZDMxZTJjNjIzLnNldENvbnRlbnQoaHRtbF81ZDljZmRkNGYzZDc0NWE5OTM0MGU4OTI5N2NmNzgxZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9iNmExOGM3MjY1MDM0MmNlOTc5NTU1OGYzM2M1ZDA0Yy5iaW5kUG9wdXAocG9wdXBfNGQzOWUxYWViYjg5NGI1MDllOWUzYmZkMzFlMmM2MjMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZDY3OTIwODcyODJjNGM4ZTg5NDlmMGRiNTE3ZWE1ZWMgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMy41MDgsIDE0LjI3MjVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDIyNy4wNDIzNTg4MDAwMDAwNCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZWY5ZWM1ZTg5ZmZmNDg3MjlhNGIwMWMwNzNlODZmYWMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2UwMTAxYzQxZjA4MTRkNThhNmY4ODJiMjc1MDllNmM2ID0gJChgPGRpdiBpZD0iaHRtbF9lMDEwMWM0MWYwODE0ZDU4YTZmODgyYjI3NTA5ZTZjNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEx1Y2lhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2VmOWVjNWU4OWZmZjQ4NzI5YTRiMDFjMDczZTg2ZmFjLnNldENvbnRlbnQoaHRtbF9lMDEwMWM0MWYwODE0ZDU4YTZmODgyYjI3NTA5ZTZjNik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kNjc5MjA4NzI4MmM0YzhlODk0OWYwZGI1MTdlYTVlYy5iaW5kUG9wdXAocG9wdXBfZWY5ZWM1ZTg5ZmZmNDg3MjlhNGIwMWMwNzNlODZmYWMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfOGFlOWMwMGJmYTgzNGE1NjlhOWU0YjUyMGY1MGMxNGQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxMi41MTY2NTQ4LCAxMy41ODNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDIxNy45OTIzMzU5LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hMmE4YzE1NmYyODA0OTRjODNiZGFkODliYzUzNmVlZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZGExNjM1ODQyNDdlNGM3ZDkwZDNhODZjNWEwNjMxZDggPSAkKGA8ZGl2IGlkPSJodG1sX2RhMTYzNTg0MjQ3ZTRjN2Q5MGQzYTg2YzVhMDYzMWQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gVmluY2VudCBhbmQgdGhlIEdyZW5hZGluZXM8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYTJhOGMxNTZmMjgwNDk0YzgzYmRhZDg5YmM1MzZlZWUuc2V0Q29udGVudChodG1sX2RhMTYzNTg0MjQ3ZTRjN2Q5MGQzYTg2YzVhMDYzMWQ4KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzhhZTljMDBiZmE4MzRhNTY5YTllNGI1MjBmNTBjMTRkLmJpbmRQb3B1cChwb3B1cF9hMmE4YzE1NmYyODA0OTRjODNiZGFkODliYzUzNmVlZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9kYmZhMjA5ZGJhNWY0ODc1OTY4MDc1YzFiOWRhYTUyYSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzguNjg1Mjc4LCAyMi4yMjQ5MThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDM0Ljk1NDg3MjQsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2E5NDY2MGNhNDQ0MTQ4MjZhYTBjZmE2ZTY2MTcxNWI1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hNWYwNWY1ZmFjNTI0NWQwYTYxYzJjNzM1MTM1YWJkNCA9ICQoYDxkaXYgaWQ9Imh0bWxfYTVmMDVmNWZhYzUyNDVkMGE2MWMyYzczNTEzNWFiZDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1ZGFuPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2E5NDY2MGNhNDQ0MTQ4MjZhYTBjZmE2ZTY2MTcxNWI1LnNldENvbnRlbnQoaHRtbF9hNWYwNWY1ZmFjNTI0NWQwYTYxYzJjNzM1MTM1YWJkNCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kYmZhMjA5ZGJhNWY0ODc1OTY4MDc1YzFiOWRhYTUyYS5iaW5kUG9wdXAocG9wdXBfYTk0NjYwY2E0NDQxNDgyNmFhMGNmYTZlNjYxNzE1YjUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZjEzODE3NTkyYzllNDQzNmFiODg2ZTMwMWIzZWQxNzUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsxLjgzMTI4MDIsIDYuMjI1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzNjQuOTkxMzM2MSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzQzNGM3MzU0YjViNDNiMzk0MzQ3N2I0NDU2N2Y1YjUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzE1ZjY1Yjc5MjMyOTQwMDdiOTIyODk3Yzk3YzIwZjMzID0gJChgPGRpdiBpZD0iaHRtbF8xNWY2NWI3OTIzMjk0MDA3YjkyMjg5N2M5N2MyMGYzMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3VyaW5hbWU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzQzNGM3MzU0YjViNDNiMzk0MzQ3N2I0NDU2N2Y1YjUuc2V0Q29udGVudChodG1sXzE1ZjY1Yjc5MjMyOTQwMDdiOTIyODk3Yzk3YzIwZjMzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2YxMzgxNzU5MmM5ZTQ0MzZhYjg4NmUzMDFiM2VkMTc1LmJpbmRQb3B1cChwb3B1cF83NDM0YzczNTRiNWI0M2IzOTQzNDc3YjQ0NTY3ZjViNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xZDU5NTZkODdjYTY0MjE5OTc4ZmNiZTNlNTZmYjdlNSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzU1LjEzMzExOTIsIDY5LjA1OTk2OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDU1MS44NDIxNDgxLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jMDU0NzlhODhlMDI0MDZmOWE0ZWVhYjkwN2I4ZDhlNyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZTMyNjY3ODA5MTE4NDdjYjlhYzNhN2UyYjdkMWU4Y2UgPSAkKGA8ZGl2IGlkPSJodG1sX2UzMjY2NzgwOTExODQ3Y2I5YWMzYTdlMmI3ZDFlOGNlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Td2VkZW48L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzA1NDc5YTg4ZTAyNDA2ZjlhNGVlYWI5MDdiOGQ4ZTcuc2V0Q29udGVudChodG1sX2UzMjY2NzgwOTExODQ3Y2I5YWMzYTdlMmI3ZDFlOGNlKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzFkNTk1NmQ4N2NhNjQyMTk5NzhmY2JlM2U1NmZiN2U1LmJpbmRQb3B1cChwb3B1cF9jMDU0NzlhODhlMDI0MDZmOWE0ZWVhYjkwN2I4ZDhlNykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV85YzNhNWFkYWViOWI0OWY5OTM2MmQ1MmViM2I3OWRhMCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQ1LjgxNzk5NSwgNDcuODA4NDY0OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDYyLjUyMjk5MjQsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M5MjY0YmYzNDVjODQxNjVhYTU3NzBlMDliYWE1ODI0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lNzUyYzU0N2ZkMmI0ODYyOWM3MjA4ODAwMmMyNjI2NCA9ICQoYDxkaXYgaWQ9Imh0bWxfZTc1MmM1NDdmZDJiNDg2MjljNzIwODgwMDJjMjYyNjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN3aXR6ZXJsYW5kPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M5MjY0YmYzNDVjODQxNjVhYTU3NzBlMDliYWE1ODI0LnNldENvbnRlbnQoaHRtbF9lNzUyYzU0N2ZkMmI0ODYyOWM3MjA4ODAwMmMyNjI2NCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV85YzNhNWFkYWViOWI0OWY5OTM2MmQ1MmViM2I3OWRhMC5iaW5kUG9wdXAocG9wdXBfYzkyNjRiZjM0NWM4NDE2NWFhNTc3MGUwOWJhYTU4MjQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZDZlODhiZTZhNjFmNGM0Y2FiMGNkOGVkYWUwMTg2OTIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszMi4zMTEzNTQsIDM3LjMxODQ1ODldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDI3My42OTkwOTQzLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mMTM5NzBiOTYyN2I0ZjBlYjQxYWMyNzVjYmY3NmQ1MSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjE3MTgzZGU0OWEwNDdlMDk0YzYzMWM2MjI4MzA5MGUgPSAkKGA8ZGl2IGlkPSJodG1sX2YxNzE4M2RlNDlhMDQ3ZTA5NGM2MzFjNjIyODMwOTBlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TeXJpYW4gQXJhYiBSZXB1YmxpYzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mMTM5NzBiOTYyN2I0ZjBlYjQxYWMyNzVjYmY3NmQ1MS5zZXRDb250ZW50KGh0bWxfZjE3MTgzZGU0OWEwNDdlMDk0YzYzMWM2MjI4MzA5MGUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZDZlODhiZTZhNjFmNGM0Y2FiMGNkOGVkYWUwMTg2OTIuYmluZFBvcHVwKHBvcHVwX2YxMzk3MGI5NjI3YjRmMGViNDFhYzI3NWNiZjc2ZDUxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzAwOWQ1OGZjODg2ZTRiNmU5OTczMzVhY2U0ZTVlNjc5ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTExLjc2MTI1NCwgLTAuOTg1NDgxMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTUuNDkzNDg5OTk5OTk5OTk4LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wMzgxOTc0MWFhZDg0YWFlYTZhOTI0ZTFmZDVjMzAwMCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZGFjOWI5ZGM0ZmIwNGViZWE4ZTQ4MDdjMWE4ZWRkNmYgPSAkKGA8ZGl2IGlkPSJodG1sX2RhYzliOWRjNGZiMDRlYmVhOGU0ODA3YzFhOGVkZDZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UYW56YW5pYTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wMzgxOTc0MWFhZDg0YWFlYTZhOTI0ZTFmZDVjMzAwMC5zZXRDb250ZW50KGh0bWxfZGFjOWI5ZGM0ZmIwNGViZWE4ZTQ4MDdjMWE4ZWRkNmYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfMDA5ZDU4ZmM4ODZlNGI2ZTk5NzMzNWFjZTRlNWU2NzkuYmluZFBvcHVwKHBvcHVwXzAzODE5NzQxYWFkODRhYWVhNmE5MjRlMWZkNWMzMDAwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzI1MzRkNGZlMjkxZTQ4NDhiODQ0MGY0NmVhMWI4NTI3ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNS42MTI4NTEsIDIwLjQ2NDgzMzddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQ1My40NDkxNzMzOTk5OTk5NSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzdkY2Y0YWY4MzMwNDFkNzkyMDc3ZGJkYTk3YzllNGMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2NiMjY1OWZkNjllZTQ4MGRhNjU4NjBlNmM5MDk4ZjQ1ID0gJChgPGRpdiBpZD0iaHRtbF9jYjI2NTlmZDY5ZWU0ODBkYTY1ODYwZTZjOTA5OGY0NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhhaWxhbmQ8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzdkY2Y0YWY4MzMwNDFkNzkyMDc3ZGJkYTk3YzllNGMuc2V0Q29udGVudChodG1sX2NiMjY1OWZkNjllZTQ4MGRhNjU4NjBlNmM5MDk4ZjQ1KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzI1MzRkNGZlMjkxZTQ4NDhiODQ0MGY0NmVhMWI4NTI3LmJpbmRQb3B1cChwb3B1cF83N2RjZjRhZjgzMzA0MWQ3OTIwNzdkYmRhOTdjOWU0YykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xOTExMWE0NWNhNzE0ZDVlODJlNTM2NmRmYmU4Y2Q0MiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzUuOTI2NTQ3LCAxMS4xMzk1MTAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAzMS45NDQzNjIxOTk5OTk5OTcsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzdlYmNmNTQ3MTIwNjRlYjNhYjM2MGI1ODc5MjZjYjNlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wZDdmNzgzNjU2NTY0MDNkOGEyNzU2YWViNmEyZTJlMiA9ICQoYDxkaXYgaWQ9Imh0bWxfMGQ3Zjc4MzY1NjU2NDAzZDhhMjc1NmFlYjZhMmUyZTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvZ288L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfN2ViY2Y1NDcxMjA2NGViM2FiMzYwYjU4NzkyNmNiM2Uuc2V0Q29udGVudChodG1sXzBkN2Y3ODM2NTY1NjQwM2Q4YTI3NTZhZWI2YTJlMmUyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzE5MTExYTQ1Y2E3MTRkNWU4MmU1MzY2ZGZiZThjZDQyLmJpbmRQb3B1cChwb3B1cF83ZWJjZjU0NzEyMDY0ZWIzYWIzNjBiNTg3OTI2Y2IzZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV83YTgwYzgxMjYyOTA0ZmM2ODg5MjU5YWJiZWM3Mjk0ZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0yNC4xMDM0NDk5LCAtMTUuMzY1NTcyMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogOTguMzU2MTk1OSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDgyMmE1NWY3MTJhNGUzYTlkODExODQ0MjcxMmExNTkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2RiMGYyM2E2OTFhMzQ4N2M4MWUzYjRjZDgzY2RmNDIyID0gJChgPGRpdiBpZD0iaHRtbF9kYjBmMjNhNjkxYTM0ODdjODFlM2I0Y2Q4M2NkZjQyMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9uZ2E8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDgyMmE1NWY3MTJhNGUzYTlkODExODQ0MjcxMmExNTkuc2V0Q29udGVudChodG1sX2RiMGYyM2E2OTFhMzQ4N2M4MWUzYjRjZDgzY2RmNDIyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzdhODBjODEyNjI5MDRmYzY4ODkyNTlhYmJlYzcyOTRlLmJpbmRQb3B1cChwb3B1cF9kODIyYTU1ZjcxMmE0ZTNhOWQ4MTE4NDQyNzEyYTE1OSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV80MWQ3OWFlMjIyNTY0YWM5YmVhYWVjNzZkZWQzYWYwMiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzkuODczMjEwNiwgMTEuNTYyODM3Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMzcxNC4wMDU0MjQsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2RlMGM0MGU5NjMwNTQ2Y2U5MWM4NDFkYzkyMmM1M2E4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mOThlODljNWU3MDc0M2FjOTVkMzI2YTQxM2RiZTc2MyA9ICQoYDxkaXYgaWQ9Imh0bWxfZjk4ZTg5YzVlNzA3NDNhYzk1ZDMyNmE0MTNkYmU3NjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRyaW5pZGFkIGFuZCBUb2JhZ288L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZGUwYzQwZTk2MzA1NDZjZTkxYzg0MWRjOTIyYzUzYTguc2V0Q29udGVudChodG1sX2Y5OGU4OWM1ZTcwNzQzYWM5NWQzMjZhNDEzZGJlNzYzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzQxZDc5YWUyMjI1NjRhYzliZWFhZWM3NmRlZDNhZjAyLmJpbmRQb3B1cChwb3B1cF9kZTBjNDBlOTYzMDU0NmNlOTFjODQxZGM5MjJjNTNhOCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV80MTE4NGQ3ZjI2MGU0NWU1OWNhMzc3ZDhiYWE2MjcwZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzMwLjIzMDIzNiwgMzcuNzYxMjA1Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjQwLjI0NTU2MzkwMDAwMDA0LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hMzU2ZmY5Y2M0ZjA0N2VmOTFlZjM0MmY3NTczMWQxNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjQyMmU4MWYyNGE2NGMwNDlhZWUxZDMzYWIwZjAxMjMgPSAkKGA8ZGl2IGlkPSJodG1sX2Y0MjJlODFmMjRhNjRjMDQ5YWVlMWQzM2FiMGYwMTIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UdW5pc2lhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2EzNTZmZjljYzRmMDQ3ZWY5MWVmMzQyZjc1NzMxZDE0LnNldENvbnRlbnQoaHRtbF9mNDIyZTgxZjI0YTY0YzA0OWFlZTFkMzNhYjBmMDEyMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV80MTE4NGQ3ZjI2MGU0NWU1OWNhMzc3ZDhiYWE2MjcwZS5iaW5kUG9wdXAocG9wdXBfYTM1NmZmOWNjNGYwNDdlZjkxZWYzNDJmNzU3MzFkMTQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZWEwODU2YWVmNzc5NGJkNzk1NTgyMmU5YTIyNmZlNTIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszNS44MDc2ODA0LCA0Mi4yOTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMzMzg4ZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IGZhbHNlLCAiZmlsbENvbG9yIjogIiMzMzg4ZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQzOC4zMTA0NTE2LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzQwOTZjNGE5ODc5NDQ5MzU4MWI0ODNhYzAzZThmYjE3KTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iNzc1NDhmYThhNjM0MmU5YTA1ZTkzNTE1OWQ2ZmMwMSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMTQ0ODlmODg1ZjQ3NGJhYTg0ZDM3ODJhMmEzMDg1ODAgPSAkKGA8ZGl2IGlkPSJodG1sXzE0NDg5Zjg4NWY0NzRiYWE4NGQzNzgyYTJhMzA4NTgwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UdXJrZXk8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjc3NTQ4ZmE4YTYzNDJlOWEwNWU5MzUxNTlkNmZjMDEuc2V0Q29udGVudChodG1sXzE0NDg5Zjg4NWY0NzRiYWE4NGQzNzgyYTJhMzA4NTgwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2VhMDg1NmFlZjc3OTRiZDc5NTU4MjJlOWEyMjZmZTUyLmJpbmRQb3B1cChwb3B1cF9iNzc1NDhmYThhNjM0MmU5YTA1ZTkzNTE1OWQ2ZmMwMSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8wYWFjNTBjMDBkNjE0MDhiYWI2MTZiNGQxYWVkMWUzYyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0xLjQ4MjMxNzksIDQuMjM0MDc2Nl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTEuMDg4NjU3NiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzk5ZmRkMGIyMWQ3NGQ1NGE0YmI4MmZlZGM2ZjZkOGUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FhZGU1ZDc1MGYwNzQyMWQ4OTE0YjdiN2Q4ZGY4ZjM0ID0gJChgPGRpdiBpZD0iaHRtbF9hYWRlNWQ3NTBmMDc0MjFkODkxNGI3YjdkOGRmOGYzNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VWdhbmRhPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzc5OWZkZDBiMjFkNzRkNTRhNGJiODJmZWRjNmY2ZDhlLnNldENvbnRlbnQoaHRtbF9hYWRlNWQ3NTBmMDc0MjFkODkxNGI3YjdkOGRmOGYzNCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8wYWFjNTBjMDBkNjE0MDhiYWI2MTZiNGQxYWVkMWUzYy5iaW5kUG9wdXAocG9wdXBfNzk5ZmRkMGIyMWQ3NGQ1NGE0YmI4MmZlZGM2ZjZkOGUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMGFhZTE1NDRmMzM5NGRlMGE2ZjliYzBmYzdkOWI5N2YgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsyMi42MzE2MjE0LCAyNi4xNTE3MjE5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAyMDQzLjM4Mzc2MiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOWMxNTdlMzY0MmIwNDFkYjkwYTFmMzJjNGVhYmMxZDggPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Y4NjIwZjA3MjE3ZjRhN2RiMjFlYjBjYTIxNTk2OTMzID0gJChgPGRpdiBpZD0iaHRtbF9mODYyMGYwNzIxN2Y0YTdkYjIxZWIwY2EyMTU5NjkzMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdGVkIEFyYWIgRW1pcmF0ZXM8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOWMxNTdlMzY0MmIwNDFkYjkwYTFmMzJjNGVhYmMxZDguc2V0Q29udGVudChodG1sX2Y4NjIwZjA3MjE3ZjRhN2RiMjFlYjBjYTIxNTk2OTMzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzBhYWUxNTQ0ZjMzOTRkZTBhNmY5YmMwZmM3ZDliOTdmLmJpbmRQb3B1cChwb3B1cF85YzE1N2UzNjQyYjA0MWRiOTBhMWYzMmM0ZWFiYzFkOCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8yYWNkNTU1NzZmNGE0MDkwODYyZGQ4NGRmMDA1MDljNyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzQ5LjY3NCwgNjEuMDYxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA3MDguNTczMjA4NiwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMjZmNzZmYzc4YjVkNDNlYzg3NTBjOTk3MzQxYWUwMzQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2UxMDc1OGE3OWM0ZjQ2ZWY5ODk1YTVjNDlmNmMxMDg0ID0gJChgPGRpdiBpZD0iaHRtbF9lMTA3NThhNzljNGY0NmVmOTg5NWE1YzQ5ZjZjMTA4NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdGVkIEtpbmdkb208L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMjZmNzZmYzc4YjVkNDNlYzg3NTBjOTk3MzQxYWUwMzQuc2V0Q29udGVudChodG1sX2UxMDc1OGE3OWM0ZjQ2ZWY5ODk1YTVjNDlmNmMxMDg0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzJhY2Q1NTU3NmY0YTQwOTA4NjJkZDg0ZGYwMDUwOWM3LmJpbmRQb3B1cChwb3B1cF8yNmY3NmZjNzhiNWQ0M2VjODc1MGM5OTczNDFhZTAzNCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9hZTQ0NDNhMmZkYWI0NzM4OWY2YTY2MzVjMWNkNTdlZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0xNC43NjA4MzU4LCA3MS42MDQ4MjE3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNzAyLjAyMTYzNDAwMDAwMDIsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2U1NTY0OWFlN2U1NTQ5YjQ5OWFjNTVlODU3MDQ3NGVjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81ODJhMjA2ZGQwMTE0NTllODA2MDA0NzIzOTRhNTIyOSA9ICQoYDxkaXYgaWQ9Imh0bWxfNTgyYTIwNmRkMDExNDU5ZTgwNjAwNDcyMzk0YTUyMjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXRlZCBTdGF0ZXM8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTU1NjQ5YWU3ZTU1NDliNDk5YWM1NWU4NTcwNDc0ZWMuc2V0Q29udGVudChodG1sXzU4MmEyMDZkZDAxMTQ1OWU4MDYwMDQ3MjM5NGE1MjI5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2FlNDQ0M2EyZmRhYjQ3Mzg5ZjZhNjYzNWMxY2Q1N2VlLmJpbmRQb3B1cChwb3B1cF9lNTU2NDlhZTdlNTU0OWI0OTlhYzU1ZTg1NzA0NzRlYykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV83NzM5MzM4ZDg5MDg0MTA2OTg4ZGJkNDgzN2ZjZDVjZSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0zNS43ODI0NDgxLCAtMzAuMDg1Mzk2Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzMzODhmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogZmFsc2UsICJmaWxsQ29sb3IiOiAiIzMzODhmZiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjI5LjYyMDA2ODUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfNDA5NmM0YTk4Nzk0NDkzNTgxYjQ4M2FjMDNlOGZiMTcpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzJlZmU2MTg4MmIxYzQ4MzY4NmFjYmQ1NjkyZjZmMjA1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jYzUzNzllMWZiNDk0ZTc5YjRkMmRmMjEwZjdjYzE4YSA9ICQoYDxkaXYgaWQ9Imh0bWxfY2M1Mzc5ZTFmYjQ5NGU3OWI0ZDJkZjIxMGY3Y2MxOGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVydWd1YXk8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMmVmZTYxODgyYjFjNDgzNjg2YWNiZDU2OTJmNmYyMDUuc2V0Q29udGVudChodG1sX2NjNTM3OWUxZmI0OTRlNzliNGQyZGYyMTBmN2NjMThhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzc3MzkzMzhkODkwODQxMDY5ODhkYmQ0ODM3ZmNkNWNlLmJpbmRQb3B1cChwb3B1cF8yZWZlNjE4ODJiMWM0ODM2ODZhY2JkNTY5MmY2ZjIwNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8zYjk5NjNhNWVlMTU0YTgyOTk4MWVhZGY0YTNmODA4NCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzcuODkxMTQ4MSwgMjMuMzkzMzk1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjMzM4OGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiBmYWxzZSwgImZpbGxDb2xvciI6ICIjMzM4OGZmIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxOTcuMTQzMzU3MSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF80MDk2YzRhOTg3OTQ0OTM1ODFiNDgzYWMwM2U4ZmIxNyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOWJkYzFmMTJjYzIwNDMxZWEzOTM4Mzc4ZjQyMTZlZjAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzAzYjM4NDdlYjk1MDRkZjZiOTJiZjhkYTJmYThmMTVlID0gJChgPGRpdiBpZD0iaHRtbF8wM2IzODQ3ZWI5NTA0ZGY2YjkyYmY4ZGEyZmE4ZjE1ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmlldG5hbTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85YmRjMWYxMmNjMjA0MzFlYTM5MzgzNzhmNDIxNmVmMC5zZXRDb250ZW50KGh0bWxfMDNiMzg0N2ViOTUwNGRmNmI5MmJmOGRhMmZhOGYxNWUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfM2I5OTYzYTVlZTE1NGE4Mjk5ODFlYWRmNGEzZjgwODQuYmluZFBvcHVwKHBvcHVwXzliZGMxZjEyY2MyMDQzMWVhMzkzODM3OGY0MjE2ZWYwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>

# Analysis per whole country emission

## Which countries have the highest emissions historically ?


```python
df['Whole emission'] = df['CO2 Per Capita (metric tons)'] * df['Population']
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Year</th>
      <th>CO2 Per Capita (metric tons)</th>
      <th>Continent</th>
      <th>Population</th>
      <th>Whole emission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>1960</td>
      <td>0.046068</td>
      <td>Asia</td>
      <td>8996351.0</td>
      <td>414442.773324</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>1961</td>
      <td>0.053615</td>
      <td>Asia</td>
      <td>9166764.0</td>
      <td>491475.529354</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>1962</td>
      <td>0.073781</td>
      <td>Asia</td>
      <td>9345868.0</td>
      <td>689550.645811</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>1963</td>
      <td>0.074251</td>
      <td>Asia</td>
      <td>9533954.0</td>
      <td>707909.126949</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>1964</td>
      <td>0.086317</td>
      <td>Asia</td>
      <td>9731361.0</td>
      <td>839977.440205</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_hist = pd.DataFrame(df.groupby(by='Country Name', as_index=False)['Whole emission'].mean())
df_hist = df_hist.sort_values(by=['Whole emission'], ascending=False)
df_hist.head()
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
      <th>Country Name</th>
      <th>Whole emission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>141</th>
      <td>United States</td>
      <td>4.699988e+09</td>
    </tr>
    <tr>
      <th>28</th>
      <td>China</td>
      <td>2.639401e+09</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Japan</td>
      <td>9.278755e+08</td>
    </tr>
    <tr>
      <th>66</th>
      <td>India</td>
      <td>7.040976e+08</td>
    </tr>
    <tr>
      <th>140</th>
      <td>United Kingdom</td>
      <td>5.702256e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style="whitegrid")
plt.figure(figsize=(16, 40))
sns.barplot(x="Whole emission", 
            y="Country Name", 
            data=df_hist)
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_81_0.png)


## Which countries have the highest emissions lately ?


```python
# for instance in year 2011
df_lately = df[df.Year == 2011]
df_lately = df_lately.sort_values(by=['Whole emission'], ascending=False)
plt.figure(figsize=(16, 40))
sns.barplot(x='Whole emission', y="Country Name", data=df_lately)
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_83_0.png)


## Are the annual emissions decreasing or increasing ?
Let's select few countries to show the evolution


```python
selected_countries = ['France', 'Israel', 'Switzerland', 'Chile', 'China', 
                      'Colombia', 'United Kingdom', 'United States', 'Brazil', 'Australia']
plt.figure(figsize=(12, 8))
sns.lineplot(x="Year", 
             y='Whole emission', 
             hue="Country Name", 
             data=df[df["Country Name"].isin(selected_countries)])
plt.title('Evolution of the CO2 emissions for few countries')
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_85_0.png)



```python
plt.figure(figsize=(12, 8))
plt.title('Evolution of the emissions for each continent')

sns.lineplot(x="Year", 
             y='Whole emission', 
             hue="Continent", 
             data=df)
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_86_0.png)



```python
df_mean = pd.DataFrame(df.groupby(by=['Continent', 'Year'], as_index=False)['Whole emission'].mean())

plt.figure(figsize=(12, 8))
plt.title('Evolution of the MEAN emissions for each continent')

sns.lineplot(x="Year", 
             y='Whole emission', 
             hue="Continent", 
             data=df_mean)

plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_87_0.png)


## Evolution of emission share


```python
df_mean_pivot = pd.pivot_table(df_mean, index='Year', values='Whole emission', columns='Continent')
df_mean_pivot.head()
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
      <th>Continent</th>
      <th>Africa</th>
      <th>Asia</th>
      <th>Europe</th>
      <th>North America</th>
      <th>Oceania</th>
      <th>South America</th>
    </tr>
    <tr>
      <th>Year</th>
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
      <th>1960</th>
      <td>3.536444e+06</td>
      <td>3.928689e+07</td>
      <td>6.609129e+07</td>
      <td>1.219828e+08</td>
      <td>1.010739e+07</td>
      <td>1.660111e+07</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>3.695651e+06</td>
      <td>3.453619e+07</td>
      <td>6.850683e+07</td>
      <td>1.217903e+08</td>
      <td>1.037739e+07</td>
      <td>1.681596e+07</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>3.804803e+06</td>
      <td>3.239489e+07</td>
      <td>7.269145e+07</td>
      <td>1.265432e+08</td>
      <td>1.072653e+07</td>
      <td>1.796754e+07</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>4.057752e+06</td>
      <td>3.418465e+07</td>
      <td>7.773613e+07</td>
      <td>1.316383e+08</td>
      <td>1.145539e+07</td>
      <td>1.827552e+07</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>4.494858e+06</td>
      <td>3.575509e+07</td>
      <td>8.103279e+07</td>
      <td>1.384992e+08</td>
      <td>1.240671e+07</td>
      <td>1.917087e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_mean_perc = df_mean_pivot.divide(df_mean_pivot.sum(axis=1), axis=0)

plt.figure(figsize=(12, 8))

# Make the plot
plt.stackplot(range(1,53),
              df_mean_perc['Africa'], 
              df_mean_perc["Asia"], 
              df_mean_perc["Europe"],
              df_mean_perc["North America"],
              df_mean_perc["Oceania"],
              df_mean_perc["South America"],
              labels=['Africa','Asia','Europe','North America','Oceania','South America'])

# Formatting the plot
plt.legend(loc='upper left')
plt.margins(0,0)
plt.title('Evolution of emissions share over time')
plt.show()
```


![png](/images/2019-06-12-carbon_dioxide/output_90_0.png)


## World map
