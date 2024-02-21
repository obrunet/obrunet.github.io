---
title: "Scraping Car Datasheet (1970-1982)"
date: 2019-10-07
categories:
  - Data Analysis
tags: [Web scraping]
header:
  image: "/images/banners/banner_code.WebP"
excerpt: "Basics of scraping using BeautifulSoup"
mathjax: "true"
---

I've followed the Pluralsight course ["Web Scraping: Python Data Playbook"](https://www.pluralsight.com/courses/web-scraping-python-data-playbook) by Ian Ozsvald. It's really interesting. Here is a little project i've made in order to practice.

I've learnt how to scrape using the requests module and BeautifulSoup4 and discovered how to write a trustworthy scraping module backed by a unit test. Finally, i've explored how to turn the columns of data in a graphical story with this [Jupyter Notebook](https://github.com/obrunet/Web_Scraping_Projects/blob/master/2019-10-07-car_datasheet/making_scraped_data_usable.ipynb).

This data comes from a [public source](https://archive.ics.uci.edu/ml/datasets/Auto%2BMPG) with citation "Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository

The code is well commented and self explanatory. It can be used as a base for more complicated projects.


```python
import re
import pickle
import os
import csv
import requests
from bs4 import BeautifulSoup

PAGE="http://localhost:8000/auto_mpg.html"


def process_car_blocks(soup):
    """Extract information from repeated divisions"""
    car_blocks = soup.find_all('div', class_='car_block')
    rows = []
    for cb in car_blocks:
        row = extract_data(cb)
        rows.append(row)
    print(f"We have {len(rows)} rows of scraped car data")
    print(rows[0], '\n', rows[-1])

    with open("scraped_cars.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerows(rows)


def extract_data(cb):
    str_name = cb.find('span', class_='car_name').text
    str_cylinders = cb.find('span', class_='cylinders').text
    cylinders = int(str_cylinders)
    assert cylinders > 0, f"Expecting cylinders to be positive not {cylinders}"
    str_weight = cb.find('span', class_='weight').text
    weight = int(str_weight.replace(',', ''))
    assert weight > 0, f"Expecting weight to be positive not {weight}"
    territory, year = extract_territory_year(cb)
    acceleration = float(cb.find('span', class_='acceleration').text)
    assert acceleration > 0, f"Expecting acceleration to be positive"
    mpg = extract_mpg(cb)
    hp = extract_horsepower(cb)
    displacement = extract_displacement(cb.text)
    row = dict(name=str_name,
               cylinders=cylinders,
               weight=weight,
               year=year,
               territory=territory,
               acceleration=acceleration,
               mpg=mpg,
               hp=hp,
               displacement=displacement)
    return row


def extract_territory_year(cb):
    str_from = cb.find('span', class_='from').text
    year, territory = str_from.strip('()').split(',')
    year = int(year.strip())
    assert year > 0, f"Expecting year to be positive not {year}"
    territory = territory.strip()
    assert len(territory) > 1, f"Expecting territory to be a \
        useful string not {territory}"
    return territory, year


def extract_horsepower(cb):
    hp_str = cb.find('span', class_='horsepower').text
    try:
        hp = float(hp_str)
        assert hp > 30, f"Expecting reasonable hp, not {hp}"
    except ValueError:
        hp = "NULL"
    return hp


def extract_mpg(cb):
    mpg_str = cb.find('span', class_='mpg').text
    try:
        mpg = float(mpg_str.split(' ')[0])
        assert mpg > 7, f"Expecting reasonable mpg, not {mpg}"
    except ValueError:
        mpg = "NULL"
    return mpg


def extract_displacement(text):
    displacement_str = re.findall(r'.* (\d+.\d+) cubic inches', text)[0]
    displacement = float(displacement_str)
    assert displacement > 60, f"Expecting a reasonable \
displacement, not {displacement}"
    return displacement


if __name__ == "__main__":
    filename = 'scraped_page_result.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            print(f"Loading cached {filename}")
            result = pickle.load(f)
    else:
        print(f"Fetching {PAGE} from the internet")
        result = requests.get(PAGE)
        with open(filename, 'wb') as f:
            print(f"Writing cached {filename}")
            pickle.dump(result, f)
    assert result.status_code == 200, f"Got status code {result.status_code} \
which isn't a success"
    source = result.text
    soup = BeautifulSoup(source, 'html.parser')
    process_car_blocks(soup)
```