---
title: "H&M Personalized Fashion 1/2 - EDA"
date: 2024-02-05
categories:
  - Recommendation System
tags: [Recommendation System]
header:
  image: "/images/2024-02-05-hm_personalized_fashion_eda/pexels-tembela-bohle-1884581.WebP"
excerpt: "Exploratory analysis of articles, customers, and transactions datasets with recommendation engine specific caracteristics, such as a long tail."
mathjax: "true"
---

Banner made from a photo by [Tembela Bohle on pexels](https://www.pexels.com/fr-fr/photo/photographie-en-niveaux-de-gris-de-vetements-assortis-sur-etagere-1884581/)


## Introduction

__H&M Group__ is a family of brands and businesses with 53 online markets and approximately 4,850 stores. The online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, __product recommendations are key__. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.

The goal of this data science challenge is to develop product recommendations __based on data from previous transactions, as well as from customer and product meta data__. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images. _Here we're not going to use the images_.


This project is divided in 2 parts:
  - the first one (this notebook) is an EDA in order to gain insights from the available datasets, and to know how to prepare the dataset for the 2nd step
  - in a second notebook: we'll use the python library _LightFM_ to build different recommendations models.

---

## First insight

Let's start by importing all the libraries we're going to use, and load the three datasets relative to the customers, articles and transactions:


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# plotly as pandas backend
# pd.options.plotting.backend = "plotly"



import os
from scipy import sparse

ENV = "COLAB"  # "LOCAL"  #

if ENV == "COLAB":
    from google.colab import drive
    drive.mount('/content/drive')
    dir_path = "drive/MyDrive/recomm/projet/"
else:
    dir_path = "../../../dataset/"


file_customers = "customers.csv"
file_articles = "articles.csv"
file_transactions = "transactions_train.csv"


df_customers = pd.read_csv(dir_path + file_customers)
df_articles = pd.read_csv(dir_path + file_articles)
df_transactions = pd.read_csv(dir_path + file_transactions)
```

Usually informations on the customers are more used for marketings (clustering / segmentation & KYC) purpose rather than for building the recommendation system. In order to get a big picture, we can start by a description of each datasets' features (type, number & percentage of missing values, number & percentage of unique values and so on):

__Metadata for each `customer_id` in dataset__


```python
def describe_df(df):
    list_item = []
    for col in df.columns:
        list_item.append([
            col,
            df[col].dtype,
            df[col].isna().sum(),
            round(df[col].isna().sum()/len(df[col])*100, 2),
            df[col].nunique(),
            round(df[col].nunique()/len(df[col])*100, 2),
            list(df[col].unique()[:5])
        ])
    return pd.DataFrame(
        columns=['feature', 'type', '# null', '% null', '# unique', '% unique', 'sample'],
        data = list_item
    )


assert df_customers.customer_id.nunique() == df_customers.shape[0]
describe_df(df_customers)
```






  <div id="df-51922206-0239-4654-bc83-ce53fedf13ff">
    <div class="colab-df-container">
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
      <th>feature</th>
      <th>type</th>
      <th># null</th>
      <th>% null</th>
      <th># unique</th>
      <th>% unique</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>customer_id</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>1371980</td>
      <td>100.00</td>
      <td>[00000dbacae5abe5e23885899a1fa44253a17956c6d1c...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FN</td>
      <td>float64</td>
      <td>895050</td>
      <td>65.24</td>
      <td>1</td>
      <td>0.00</td>
      <td>[nan, 1.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Active</td>
      <td>float64</td>
      <td>907576</td>
      <td>66.15</td>
      <td>1</td>
      <td>0.00</td>
      <td>[nan, 1.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>club_member_status</td>
      <td>object</td>
      <td>6062</td>
      <td>0.44</td>
      <td>3</td>
      <td>0.00</td>
      <td>[ACTIVE, nan, PRE-CREATE, LEFT CLUB]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fashion_news_frequency</td>
      <td>object</td>
      <td>16009</td>
      <td>1.17</td>
      <td>4</td>
      <td>0.00</td>
      <td>[NONE, Regularly, nan, Monthly, None]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>age</td>
      <td>float64</td>
      <td>15861</td>
      <td>1.16</td>
      <td>84</td>
      <td>0.01</td>
      <td>[49.0, 25.0, 24.0, 54.0, 52.0]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>postal_code</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>352899</td>
      <td>25.72</td>
      <td>[52043ee2162cf5aa7ee79974281641c6f11a68d276429...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-51922206-0239-4654-bc83-ce53fedf13ff')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-4a0679bb-a5eb-4e16-a718-a09fc0b2e614">
      <button class="colab-df-quickchart" onclick="quickchart('df-4a0679bb-a5eb-4e16-a718-a09fc0b2e614')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>

      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-4a0679bb-a5eb-4e16-a718-a09fc0b2e614 button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-51922206-0239-4654-bc83-ce53fedf13ff button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-51922206-0239-4654-bc83-ce53fedf13ff');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




__Detailed metadata for each article_id available for purchase:__


This database contains information about the assortiment of H&M shops.

Unique indentifier of an article:
- `article_id` (int64) - an unique 9-digit identifier of the article, 105 542 unique values (as the length of the database)

5 product related columns:
- `product_code` (int64) - 6-digit product code (the first 6 digits of article_id, 47 224 unique values
- `prod_name` (object) - name of a product, 45 875 unique values
- `product_type_no` (int64) - product type number, 131 unique values
- `product_type_name` (object) - name of a product type, equivalent of product_type_no
- `product_group_name` (object) - name of a product group, in total 19 groups


2 columns related to the pattern:

- `graphical_appearance_no` (int64) - code of a pattern, 30 unique values
- `graphical_appearance_name` (object) - name of a pattern, 30 unique values

2 columns related to the color:

- `colour_group_code` (int64) - code of a color, 50 unique values
- `colour_group_name` (object) - name of a color, 50 unique values

4 columns related to perceived colour (general tone):

- `perceived_colour_value_id` - perceived color id, 8 unique values
- `perceived_colour_value_name` - perceived color name, 8 unique values
- `perceived_colour_master_id` - perceived master color id, 20 unique values
- `perceived_colour_master_name` - perceived master color name, 20 unique values

2 columns related to the department:

- `department_no` - department number, 299 unique values
- `department_name` - department name, 299 unique values

4 columns related to the index, which is actually a top-level category:

- `index_code` - index code, 10 unique values
- `index_name` - index name, 10 unique values
- `index_group_no` - index group code, 5 unique values
- `index_group_name` - index group code, 5 unique values

2 columns related to the section:

- `section_no` - section number, 56 unique values
- `section_name` - section name, 56 unique values

2 columns related to the garment group:

- `garment_group_n` - section number, 56 unique values
- `garment_group_name` - section name, 56 unique values

1 column with a detailed description of the article:

- `detail_desc` - 43 404 unique values


```python
assert df_articles.article_id.nunique() == df_articles.shape[0]
describe_df(df_articles)
```






  <div id="df-8979b985-c9aa-4590-9071-2481b9c286ba">
    <div class="colab-df-container">
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
      <th>feature</th>
      <th>type</th>
      <th># null</th>
      <th>% null</th>
      <th># unique</th>
      <th>% unique</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>article_id</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>105542</td>
      <td>100.00</td>
      <td>[108775015, 108775044, 108775051, 110065001, 1...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>product_code</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>47224</td>
      <td>44.74</td>
      <td>[108775, 110065, 111565, 111586, 111593]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>prod_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>45875</td>
      <td>43.47</td>
      <td>[Strap top, Strap top (1), OP T-shirt (Idro), ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>product_type_no</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>132</td>
      <td>0.13</td>
      <td>[253, 306, 304, 302, 273]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>product_type_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>131</td>
      <td>0.12</td>
      <td>[Vest top, Bra, Underwear Tights, Socks, Leggi...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>product_group_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>19</td>
      <td>0.02</td>
      <td>[Garment Upper body, Underwear, Socks &amp; Tights...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>graphical_appearance_no</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>30</td>
      <td>0.03</td>
      <td>[1010016, 1010017, 1010001, 1010010, 1010019]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>graphical_appearance_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>30</td>
      <td>0.03</td>
      <td>[Solid, Stripe, All over pattern, Melange, Tra...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>colour_group_code</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>50</td>
      <td>0.05</td>
      <td>[9, 10, 11, 12, 13]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>colour_group_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>50</td>
      <td>0.05</td>
      <td>[Black, White, Off White, Light Beige, Beige]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>perceived_colour_value_id</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>8</td>
      <td>0.01</td>
      <td>[4, 3, 1, 2, 5]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>perceived_colour_value_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>8</td>
      <td>0.01</td>
      <td>[Dark, Light, Dusty Light, Medium Dusty, Bright]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>perceived_colour_master_id</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>20</td>
      <td>0.02</td>
      <td>[5, 9, 11, 12, 2]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>perceived_colour_master_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>20</td>
      <td>0.02</td>
      <td>[Black, White, Beige, Grey, Blue]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>department_no</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>299</td>
      <td>0.28</td>
      <td>[1676, 1339, 3608, 6515, 1334]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>department_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>250</td>
      <td>0.24</td>
      <td>[Jersey Basic, Clean Lingerie, Tights basic, B...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>index_code</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>10</td>
      <td>0.01</td>
      <td>[A, B, G, F, C]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>index_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>10</td>
      <td>0.01</td>
      <td>[Ladieswear, Lingeries/Tights, Baby Sizes 50-9...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>index_group_no</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>5</td>
      <td>0.00</td>
      <td>[1, 4, 3, 26, 2]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>index_group_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>5</td>
      <td>0.00</td>
      <td>[Ladieswear, Baby/Children, Menswear, Sport, D...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>section_no</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>57</td>
      <td>0.05</td>
      <td>[16, 61, 62, 44, 26]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>section_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>56</td>
      <td>0.05</td>
      <td>[Womens Everyday Basics, Womens Lingerie, Wome...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>garment_group_no</td>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>21</td>
      <td>0.02</td>
      <td>[1002, 1017, 1021, 1005, 1019]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>garment_group_name</td>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>21</td>
      <td>0.02</td>
      <td>[Jersey Basic, Under-, Nightwear, Socks and Ti...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>detail_desc</td>
      <td>object</td>
      <td>416</td>
      <td>0.39</td>
      <td>43404</td>
      <td>41.12</td>
      <td>[Jersey top with narrow shoulder straps., Micr...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8979b985-c9aa-4590-9071-2481b9c286ba')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-8cc77db8-685c-4fa6-a689-270923d4db60">
      <button class="colab-df-quickchart" onclick="quickchart('df-8cc77db8-685c-4fa6-a689-270923d4db60')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>

      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-8cc77db8-685c-4fa6-a689-270923d4db60 button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8979b985-c9aa-4590-9071-2481b9c286ba button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8979b985-c9aa-4590-9071-2481b9c286ba');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Our task is to predict the `article_id` each customer will purchase :


```python
df_transactions.t_dat = pd.to_datetime(df_transactions.t_dat, infer_datetime_format=True)
describe_df(df_transactions)
```






  <div id="df-7447e1f3-44b2-4165-b039-62dbb7724633">
    <div class="colab-df-container">
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
      <th>feature</th>
      <th>type</th>
      <th># null</th>
      <th>% null</th>
      <th># unique</th>
      <th>% unique</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>t_dat</td>
      <td>datetime64[ns]</td>
      <td>0</td>
      <td>0.0</td>
      <td>734</td>
      <td>0.00</td>
      <td>[2018-09-20T00:00:00.000000000, 2018-09-21T00:...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>customer_id</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>1362281</td>
      <td>4.29</td>
      <td>[000058a12d5b43e67d225668fa1f8d618c13dc232df0c...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>article_id</td>
      <td>int64</td>
      <td>0</td>
      <td>0.0</td>
      <td>104547</td>
      <td>0.33</td>
      <td>[663713001, 541518023, 505221004, 685687003, 6...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>price</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>9857</td>
      <td>0.03</td>
      <td>[0.0508305084745762, 0.0304915254237288, 0.015...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sales_channel_id</td>
      <td>int64</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>[2, 1]</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7447e1f3-44b2-4165-b039-62dbb7724633')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-bb6e5455-94c7-40fc-a063-4ec2acc60252">
      <button class="colab-df-quickchart" onclick="quickchart('df-bb6e5455-94c7-40fc-a063-4ec2acc60252')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>

      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-bb6e5455-94c7-40fc-a063-4ec2acc60252 button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7447e1f3-44b2-4165-b039-62dbb7724633 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7447e1f3-44b2-4165-b039-62dbb7724633');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




---
## Focus on Customers

The metadata for each `customer_id` in the dataset consists of `club member status`, whether they `subscribe` to fashion news or not, and `age`. Some types are wrong due to missing values:


```python
print(f"there are {(df_customers.age != df_customers.age.round()).sum()} customers whose age is not an integer")
print(f"there are {df_customers.age.isnull().sum()} ages missing")
```

    there are 15861 customers whose age is not an integer
    there are 15861 ages missing


Missing ages are arbitrarily replaced by 0 so that unknow values can be distinguished:


```python
mapping = {"FN": 0, "Active": 1, "club_member_status": "N.C", "fashion_news_frequency": "N.C", "age": 0}

df_customers.fillna(value=mapping, inplace=True)
df_customers.drop(columns="postal_code", inplace=True)

for col in ["FN", "age", "Active"]:
    df_customers[col] = df_customers[col].astype(np.int8)
```

Now let's see the ratios of the number of customers for each feature:


```python
cols = ["FN", "Active", "club_member_status", "fashion_news_frequency"]
fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(12, 6), tight_layout=True)

for i, c in enumerate(cols):
  df_customers[c].value_counts().plot.pie(ax=axes[i], title=c)
```


    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_13_0.png)
    



```python
cols = ["FN", "Active", "club_member_status", "fashion_news_frequency"]
fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(12, 3), tight_layout=True)

for i, c in enumerate(cols):
  df_customers[c].value_counts().plot.bar(ax=axes[i], title=c)
```


    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_14_0.png)
    


By visualizing the distribution of the age, we can clearly see that:
- there are two types of clients: between 20 & 40, and older than 40 yrs old.
- the number of missing values isn't too high


```python
plt.figure(figsize=(6, 2))
sns.histplot(data=df_customers, x='age', bins=50)
```


    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_16_1.png)
    


The shape of the age distribution remains quite the same for each category of customer:


```python
for c in cols:
  plt.figure(figsize=(6, 2))
  sns.histplot(data=df_customers, x='age', bins=50, hue=c, element="poly")
  plt.show()
```


    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_18_0.png)
    



    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_18_1.png)
    



    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_18_2.png)
    



    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_18_3.png)
    


---

## Focus on Articles


With a sunburst chart, we visualize the hierarchical structures of the different clothes' categories:
- the "sport" category is the least represented
- while the "ladieswear" & the "baby/children" seem to be the most important
- unlike the "ladieswear" composed mostly of "Garment Upper Body", the "baby/children" clothes are more diversified:


```python
cols = ["index_group_name", "index_name",  "product_group_name"]
df_temp = pd.DataFrame(df_articles[cols].value_counts()).rename(columns={0: "counts"}).reset_index()
px.sunburst(
    df_temp,
    path=cols,
    values='counts',
)
```

![](/images/2024-02-05-hm_personalized_fashion_eda/05.png)

If we don't group those clothes by `index_group` by rather by `product type`, we can see that the most represented product are: dresses, sweaters, swim wear bodies & trousers:


```python
def plot_bar(df, column):
    long_df = pd.DataFrame(df.groupby(column)['article_id'].count().reset_index().rename({'article_id': 'count'}, axis=1))
    fig = px.bar(long_df, x=column, y="count", color=column, title=f"bar plot for {column} ", width=900, height=550)
    fig.show()

def plot_hist(df, column):
    fig = px.histogram(df, x=column, nbins=10, title=f'{column} distribution ')
    fig.show()


plot_bar(df_articles,'product_type_name')
```

![](/images/2024-02-05-hm_personalized_fashion_eda/06.png)


```python
plot_bar(df_articles,'product_group_name')
```

![](/images/2024-02-05-hm_personalized_fashion_eda/07.png)


```python
plot_bar(df_articles,'graphical_appearance_name')
```

![](/images/2024-02-05-hm_personalized_fashion_eda/08.png)

# Focus an Transactions



Most item catalogs exhibit the long tail effect (popularity bias): very few items are demanded & sold, whereas most of the articles aren't sold that much.


```python
df_transactions['article_id'].value_counts().reset_index().drop(columns=["index"]).plot(figsize=(6, 4))
```

    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_31_1.png)
    

Curiously, the same is also true for customers. This can be explained by the fact that few customers are in fact societies or resellers that buy high volumnes of items, where as the vast majority of the customers have only bought 1 or 2 items:


```python
df_transactions['customer_id'].value_counts()\
  .reset_index().sort_index().plot(figsize=(6, 4))
```
    
![png](/images/2024-02-05-hm_personalized_fashion_eda/output_33_1.png)
    

The history of transactions spans three years:


```python
df_transactions["t_dat"].dt.year.unique()
```




    array([2018, 2019, 2020])



Few spikes can be observed from the total daily sales, and can be explained by the shopping events or highly promoted sales at discounted price such as the "black friday":


```python
df_transactions['month'] = df_transactions["t_dat"].dt.month
df_transactions['year'] = df_transactions["t_dat"].dt.year
df_transactions['dow'] = df_transactions["t_dat"].dt.day_name()

df_temp = df_transactions.groupby('t_dat')['price'].agg(['sum', 'mean']).sort_values(by = 't_dat', ascending=False).reset_index()
px.line(df_temp, x='t_dat', y='sum', title='Total Sales daily', width=900, height=450).show()
```

![](/images/2024-02-05-hm_personalized_fashion_eda/11.png)

The count of monthly sells shows that more items are sold during the summer (remember the swim wear body as most demanded):


```python
df_temp = df_transactions.groupby(['year', 'month']).count()["article_id"].reset_index().rename(columns={"article_id": "count"})
px.line(df_temp, x="month", y="count", color='year', width=900, height=350, markers=True)
```

![](/images/2024-02-05-hm_personalized_fashion_eda/12.png)

2018 is an incomplete year in our dataset, that's why there are fewer sells monthly. But there are also fewer monthly sells in 2020 compared to 2019 for the month


```python
df_temp = df_transactions.groupby(["year", "month"]).agg({"price": "sum"}).reset_index()

px.histogram(
    df_temp,
    x="month",
    y="price",
    title='Monthly sells for each year',
    color='year',
    barmode='group',
    nbins=12,
    width=900,
    height=450
).show()
```

![](/images/2024-02-05-hm_personalized_fashion_eda/13.png)


```python
df_temp = df_transactions.groupby(["year", "dow"]).agg({"price": "sum"}).reset_index()

px.histogram(
    df_temp,
    x="dow",
    y="price",
    title='Daily sells for each year',
    color='year',
    barmode='group',
    nbins=10,
    width=900,
    height=450
).show()
```

![](/images/2024-02-05-hm_personalized_fashion_eda/14.png)

---
## Transactions analysis for different customers or articles categories

Let's keep only few months for the sake of simplicity (and also because it's quite hard to process huge amount of data with limited ressources):


```python
df = df_transactions[(df_transactions.t_dat.dt.year == 2019) & (df_transactions.t_dat.dt.month.isin([5, 6, 7, 9]))]
df.shape
```




    (6501193, 8)




```python
df_transactions.head()
```






  <div id="df-c34c1af1-5256-4a06-89cd-708af7e7678d">
    <div class="colab-df-container">
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
      <th>t_dat</th>
      <th>customer_id</th>
      <th>article_id</th>
      <th>price</th>
      <th>sales_channel_id</th>
      <th>month</th>
      <th>year</th>
      <th>dow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-09-20</td>
      <td>000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...</td>
      <td>663713001</td>
      <td>0.050831</td>
      <td>2</td>
      <td>9</td>
      <td>2018</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-09-20</td>
      <td>000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...</td>
      <td>541518023</td>
      <td>0.030492</td>
      <td>2</td>
      <td>9</td>
      <td>2018</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-09-20</td>
      <td>00007d2de826758b65a93dd24ce629ed66842531df6699...</td>
      <td>505221004</td>
      <td>0.015237</td>
      <td>2</td>
      <td>9</td>
      <td>2018</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-09-20</td>
      <td>00007d2de826758b65a93dd24ce629ed66842531df6699...</td>
      <td>685687003</td>
      <td>0.016932</td>
      <td>2</td>
      <td>9</td>
      <td>2018</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-09-20</td>
      <td>00007d2de826758b65a93dd24ce629ed66842531df6699...</td>
      <td>685687004</td>
      <td>0.016932</td>
      <td>2</td>
      <td>9</td>
      <td>2018</td>
      <td>Thursday</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c34c1af1-5256-4a06-89cd-708af7e7678d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-8b3e1d0f-2657-4e12-a5cf-334bafa888f5">
      <button class="colab-df-quickchart" onclick="quickchart('df-8b3e1d0f-2657-4e12-a5cf-334bafa888f5')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>

      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-8b3e1d0f-2657-4e12-a5cf-334bafa888f5 button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c34c1af1-5256-4a06-89cd-708af7e7678d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c34c1af1-5256-4a06-89cd-708af7e7678d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df = df.merge(df_articles[["article_id", "index_group_name", "index_name", "section_name"]], on='article_id')
df.drop(columns=["article_id"], inplace=True)#, "month", "year"])
del df_articles


df = df.merge(df_customers, on='customer_id')
df.drop(columns=["customer_id"], inplace=True)
del df_customers


# df.drop(columns=["postal_code"], inplace=True)
df['month'] = df.t_dat.dt.month
# df['year'] = df.t_dat.dt.year
df['dow'] = df.t_dat.dt.day_name
print(f"Total Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
df.head()
```

    Total Memory Usage: 2751.01 MB







  <div id="df-77691cab-fac4-466c-af06-809cb3ae9364">
    <div class="colab-df-container">
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
      <th>t_dat</th>
      <th>price</th>
      <th>sales_channel_id</th>
      <th>month</th>
      <th>year</th>
      <th>dow</th>
      <th>index_group_name</th>
      <th>index_name</th>
      <th>section_name</th>
      <th>FN</th>
      <th>Active</th>
      <th>club_member_status</th>
      <th>fashion_news_frequency</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-05-01</td>
      <td>0.050831</td>
      <td>2</td>
      <td>5</td>
      <td>2019</td>
      <td>&lt;bound method PandasDelegate._add_delegate_acc...</td>
      <td>Divided</td>
      <td>Divided</td>
      <td>Ladies Denim</td>
      <td>0</td>
      <td>1</td>
      <td>PRE-CREATE</td>
      <td>NONE</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-05-01</td>
      <td>0.050831</td>
      <td>2</td>
      <td>5</td>
      <td>2019</td>
      <td>&lt;bound method PandasDelegate._add_delegate_acc...</td>
      <td>Ladieswear</td>
      <td>Ladieswear</td>
      <td>Womens Everyday Collection</td>
      <td>0</td>
      <td>1</td>
      <td>PRE-CREATE</td>
      <td>NONE</td>
      <td>55</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-05-01</td>
      <td>0.016932</td>
      <td>2</td>
      <td>5</td>
      <td>2019</td>
      <td>&lt;bound method PandasDelegate._add_delegate_acc...</td>
      <td>Ladieswear</td>
      <td>Lingeries/Tights</td>
      <td>Womens Lingerie</td>
      <td>0</td>
      <td>1</td>
      <td>PRE-CREATE</td>
      <td>NONE</td>
      <td>55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-05-01</td>
      <td>0.033881</td>
      <td>2</td>
      <td>5</td>
      <td>2019</td>
      <td>&lt;bound method PandasDelegate._add_delegate_acc...</td>
      <td>Ladieswear</td>
      <td>Ladieswear</td>
      <td>Womens Everyday Collection</td>
      <td>0</td>
      <td>1</td>
      <td>PRE-CREATE</td>
      <td>NONE</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05-01</td>
      <td>0.016932</td>
      <td>2</td>
      <td>5</td>
      <td>2019</td>
      <td>&lt;bound method PandasDelegate._add_delegate_acc...</td>
      <td>Ladieswear</td>
      <td>Ladieswear</td>
      <td>Womens Everyday Collection</td>
      <td>0</td>
      <td>1</td>
      <td>PRE-CREATE</td>
      <td>NONE</td>
      <td>55</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-77691cab-fac4-466c-af06-809cb3ae9364')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-0a6e0c8d-41ef-4b7a-8f87-3b742ffba6c9">
      <button class="colab-df-quickchart" onclick="quickchart('df-0a6e0c8d-41ef-4b7a-8f87-3b742ffba6c9')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>

      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-0a6e0c8d-41ef-4b7a-8f87-3b742ffba6c9 button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-77691cab-fac4-466c-af06-809cb3ae9364 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-77691cab-fac4-466c-af06-809cb3ae9364');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df['dow'] = df["t_dat"].dt.day_name()


def plot_var_accross_time(var):
  for time_scale in ['month', 'dow']:
    df_temp = df.groupby([time_scale, var]).count()["t_dat"].reset_index().rename(columns={"t_dat": "count"})
    px.line(df_temp, x=time_scale, y="count", color=var, width=900, height=350, markers=True, title=f'Evolution of transactions for different {var} for each {time_scale} over 2019').show()


plot_var_accross_time("index_group_name")
```

![](/images/2024-02-05-hm_personalized_fashion_eda/15.png)


```python
plot_var_accross_time("index_name")
```

![](/images/2024-02-05-hm_personalized_fashion_eda/16.png)


```python
plot_var_accross_time("Active")
```

![](/images/2024-02-05-hm_personalized_fashion_eda/18.png)
