---
title: "Wind energy generation 1/2 - Which countries in E.U present the same profiles ?"
date: 2021-02-25
categories:
  - Data Science
tags: [Kaggle Competitions]
header:
  image: "/images/2021-02-25-wind-energy-clustering/pexels-narcisa-aciko-1292464 - cropped.png"
excerpt: "In this 1st part, we're going to make clusters of european countries with similar wind generation capacities."
mathjax: "true"
---

banner made from an image of [Narcisa Aciko on pexels.com](https://www.pexels.com/fr-fr/photo/photo-de-lot-d-eoliennes-1292464/)

__Description of the data set__

This dataset contains hourly estimates of an area’s energy potential for 1986-2015 as a percentage of a power plant’s maximum output.

The overall scope of EMHIRES is to allow users to assess the impact of meteorological and climate variability on the generation of wind power in Europe and not to mime the actual evolution of wind power production in the latest decades. For this reason, the hourly wind power generation time series are released for meteorological conditions of the years 1986-2015 (30 years) without considering any changes in the wind installed capacity. Thus, the installed capacity considered is fixed as the one installed at the end of 2015. For this reason, data from EMHIRES should not be compared with actual power generation data other than referring to the reference year 2015.

__Content__

The data is available at both the national level and the NUTS 2 level. The NUTS 2 system divides the EU into 276 statistical units.
Please see the manual for the technical details of how these estimates were generated.
This product is intended for policy analysis over a wide area and is not the best for estimating the output from a single system. Please don’t use it commercially.

__Acknowledgements__

This dataset was kindly made available by [the European Commission’s STETIS program](https://setis.ec.europa.eu/about-setis). You can find the original dataset here.

__Goal of this 1st step__

In a similar manner to the solar energy (see [part 1](https://obrunet.github.io/data%20science/Solar_Clustering/), [part 2](https://obrunet.github.io/data%20science/data%20analysis/Solar_EDA/) & [part 3](https://obrunet.github.io/data%20science/Solar_predictions/)) this is the first part of two. Here we’re going to study wind generation on a country level in order to make clusters of countries which present the same profile so that each group can be investigate in more details later.

__First look at the dataset__

There is one column per country, and each line / record is an hourly estimate


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

df_wind_on = pd.read_csv(path + "wind_generation_by_country.csv")
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
      <td>0</td>
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
      <td>1</td>
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



### Dealing with timestamps

We have to add the corresponding time at the hour level for each row in order to use and analyze this dataset


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
      <th>time</th>
      <th>hour</th>
      <th>month</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>262966</td>
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
      <td>2015-12-31 22:00:00</td>
      <td>22</td>
      <td>12</td>
      <td>53</td>
    </tr>
    <tr>
      <td>262967</td>
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
      <td>2015-12-31 23:00:00</td>
      <td>23</td>
      <td>12</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>



Let’s keep the records of one year and tranpose the dataset, because we need to have one line per region.


```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_wind_on = df_wind_on.drop(columns=['time', 'hour', 'month', 'week'])

df_wind_transposed = df_wind_on[-24*365:].T
df_wind_transposed.tail(2)
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
      <td>SE</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02559</td>
      <td>0.028774</td>
      <td>0.024368</td>
      <td>0.029511</td>
      <td>0.026991</td>
      <td>0.025740</td>
      <td>0.015349</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.021616</td>
      <td>0.035272</td>
      <td>0.053339</td>
      <td>0.061699</td>
      <td>0.066683</td>
      <td>0.028761</td>
      <td>0.015128</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025668</td>
      <td>0.076981</td>
      <td>0.123490</td>
      <td>0.127871</td>
      <td>0.094509</td>
      <td>0.037900</td>
      <td>0.017013</td>
      <td>0.000000</td>
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
      <td>0.026893</td>
      <td>0.120466</td>
      <td>0.231455</td>
      <td>0.265153</td>
      <td>0.193312</td>
      <td>0.072419</td>
      <td>0.015935</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025531</td>
      <td>0.075039</td>
      <td>0.123906</td>
      <td>0.138702</td>
      <td>0.087216</td>
      <td>0.037514</td>
      <td>0.014901</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.021719</td>
      <td>0.025448</td>
      <td>0.029375</td>
      <td>0.028795</td>
      <td>0.025825</td>
      <td>0.018200</td>
      <td>0.014727</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>0.025873</td>
      <td>0.042821</td>
      <td>0.072591</td>
      <td>0.080633</td>
      <td>0.058003</td>
      <td>0.029394</td>
      <td>0.016086</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025720</td>
      <td>0.024646</td>
      <td>0.033946</td>
      <td>0.045143</td>
      <td>0.034991</td>
      <td>0.023347</td>
      <td>0.015876</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025784</td>
      <td>0.029620</td>
      <td>0.038149</td>
      <td>0.038235</td>
      <td>0.030227</td>
      <td>0.025546</td>
      <td>0.015927</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025471</td>
      <td>0.035349</td>
      <td>0.054981</td>
      <td>0.060864</td>
      <td>0.051826</td>
      <td>0.034001</td>
      <td>0.016073</td>
      <td>0.000000</td>
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
      <td>0.025909</td>
      <td>0.035216</td>
      <td>0.041221</td>
      <td>0.039333</td>
      <td>0.034922</td>
      <td>0.030504</td>
      <td>0.016031</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025530</td>
      <td>0.039458</td>
      <td>0.075218</td>
      <td>0.083126</td>
      <td>0.078889</td>
      <td>0.051887</td>
      <td>0.016154</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>UK</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01436</td>
      <td>0.018766</td>
      <td>0.035738</td>
      <td>0.042646</td>
      <td>0.048847</td>
      <td>0.043734</td>
      <td>0.035139</td>
      <td>0.017733</td>
      <td>0.008758</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014415</td>
      <td>0.079417</td>
      <td>0.193538</td>
      <td>0.287261</td>
      <td>0.327187</td>
      <td>0.289763</td>
      <td>0.201915</td>
      <td>0.067396</td>
      <td>0.008839</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014523</td>
      <td>0.036608</td>
      <td>0.062609</td>
      <td>0.065165</td>
      <td>0.078764</td>
      <td>0.085781</td>
      <td>0.082844</td>
      <td>0.040742</td>
      <td>0.00941</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014904</td>
      <td>0.046943</td>
      <td>0.120540</td>
      <td>0.188751</td>
      <td>0.212467</td>
      <td>0.179504</td>
      <td>0.117221</td>
      <td>0.048684</td>
      <td>0.010172</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015829</td>
      <td>0.040851</td>
      <td>0.074304</td>
      <td>0.100413</td>
      <td>0.112761</td>
      <td>0.090133</td>
      <td>0.056462</td>
      <td>0.023553</td>
      <td>0.010498</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.016210</td>
      <td>0.043734</td>
      <td>0.065709</td>
      <td>0.097748</td>
      <td>0.149532</td>
      <td>0.177274</td>
      <td>0.150674</td>
      <td>0.070279</td>
      <td>0.011097</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>0.016101</td>
      <td>0.031604</td>
      <td>0.055157</td>
      <td>0.062935</td>
      <td>0.056517</td>
      <td>0.054014</td>
      <td>0.043407</td>
      <td>0.023064</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015938</td>
      <td>0.043679</td>
      <td>0.096823</td>
      <td>0.135553</td>
      <td>0.146867</td>
      <td>0.127883</td>
      <td>0.086434</td>
      <td>0.029047</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015992</td>
      <td>0.031767</td>
      <td>0.061956</td>
      <td>0.093995</td>
      <td>0.110041</td>
      <td>0.088555</td>
      <td>0.050805</td>
      <td>0.030407</td>
      <td>0.008839</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015775</td>
      <td>0.072835</td>
      <td>0.187119</td>
      <td>0.275783</td>
      <td>0.311086</td>
      <td>0.259574</td>
      <td>0.138164</td>
      <td>0.055918</td>
      <td>0.00903</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015557</td>
      <td>0.025239</td>
      <td>0.042863</td>
      <td>0.049173</td>
      <td>0.046725</td>
      <td>0.039328</td>
      <td>0.033127</td>
      <td>0.020779</td>
      <td>0.009193</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015666</td>
      <td>0.082681</td>
      <td>0.191144</td>
      <td>0.242983</td>
      <td>0.214589</td>
      <td>0.177763</td>
      <td>0.109878</td>
      <td>0.04814</td>
      <td>0.009247</td>
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



How to find the optimal K ? :) using the elbow method i've already covered / explained in depth [here](https://obrunet.github.io/data%20science/Solar_Clustering/#evaluating-the-cluster-quality)

__How many clusters would you choose ?__

A common, empirical method, is the elbow method. You plot the mean distance of every point toward its cluster center, as a function of the number of clusters. Sometimes the plot has an arm shape, and the elbow would be the optimal K.


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


plot_elbow_scores(df_wind_transposed, 20)
```


    
![png](/images/2021-02-25-wind-energy-clustering/output_7_0.png)
  



    
![png](/images/2021-02-25-wind-energy-clustering/output_7_1.png)
    


The best nb of k clusters seems to be 8 or 10 even if there isn't any real elbow on the 1st plot...  
So let's re-train the model with the k number of clusters of 6 :


```python
X = df_wind_transposed

km = KMeans(n_clusters=6).fit(X)
X['label'] = km.labels_
print("Cluster nb / Nb of countries in the cluster", X.label.value_counts())
```

    Cluster nb / Nb of countries in the cluster 3    8
    2    8
    0    6
    5    3
    4    2
    1    2
    Name: label, dtype: int64
    

# Conclusion

At the end, we are able to list the countries in the E.U, countries which share the same profiles when it comes to produces wind energy. To be honest, this was not a difficult task : the data are already cleaned and there is no need to use complicated machine learning here. But knowing the countries with the same characteristics will be usefull for the [second step] when we will analyze each profile.

You will find below the list of countries in each cluster :


```python
print("Countries grouped by cluster")
for k in range(6):
    print(f'cluster nb : {k}', " ".join(list(X[X.label == k].index)))
```

    Countries grouped by cluster
    cluster nb : 0 EE FI LT LV PL SE
    cluster nb : 1 ES PT
    cluster nb : 2 AT CH CZ HR HU IT SI SK
    cluster nb : 3 BE DE DK FR IE LU NL UK
    cluster nb : 4 CY NO
    cluster nb : 5 BG EL RO
    
