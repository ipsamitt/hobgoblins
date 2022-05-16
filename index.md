
# Weather Prediction Algorithms for Pedestrian Safety
## Introduction

THIS IS FOR THE HOBGOBLINS!!!
## Data Collection
Scraping the data: We will be obtaining the data from https://www.kaggle.com/datasets/grubenm/austin-weather

The weather data, is a subset of the overall, Austin Bike Shares Trip mobile app database https://www.kaggle.com/datasets/jboysen/austin-bike

We wanted to do something to improve the overall Austin Bike Shares Trip mobile app, so we went with weather prediction in order to ensure the safety of riders. Having accurate weather prediciton daily allows riders to safely prepare for whatever inclimate weather conditions could occur. 

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm
austin_weather = pd.read_csv("austin_weather.csv")
austin_weather.columns
``` 
## Data Management
For the "Events" Column which describes the condition of the weather for the day, all blanks will be turned into "sun". In the precipitation category, 'T' represents 'Trace', an amount measuring less than 0.01 inch. We will replace all 'T' with 0.01. Converted the string dates into panda's datetimes objects for better ease of sorting later on. This dataset displayed missing data as '-' and has a total of 80 missing data points across all columns.
As this data is negligible in consideration to the dataset of over 1300 values, we have decided to remove all rows with data points that have missing values. In order to remove discrepancies in calculations, we will also convert everything to float values from string values in order to make numerical analyses if necessary.

``` 
#Converting string dates into datetime.date objects
for i, currRow in austin_weather.iterrows():
    austin_weather.at[i, 'Date'] = datetime.strptime(austin_weather.at[i, 'Date'], "%Y-%m-%d").date()

austin_weather['Events'].replace(' ', 'Sun', inplace=True)
austin_weather['PrecipitationSumInches'].replace('T', 0.01, inplace=True)

#austin_weather[TempHighF]
austin_weather = austin_weather.replace('-', np.NaN, regex=True)
austin_weather.isnull().sum(axis = 0)
austin_weather = austin_weather.dropna()
```
In order to remove discrepancies in calculations, we will also convert everything to float values from string values in order to make numerical analyses if necessary. We will also create separate dataframes based on the weather event of each day.
```
austin_weather['TempHighF'] = austin_weather['TempHighF'].astype(float)
austin_weather['TempAvgF'] = austin_weather['TempAvgF'].astype(float)
austin_weather['TempLowF'] = austin_weather['TempLowF'].astype(float)
austin_weather['DewPointHighF'] = austin_weather['DewPointHighF'].astype(float)
austin_weather['DewPointAvgF'] = austin_weather['DewPointAvgF'].astype(float)
austin_weather['DewPointLowF'] = austin_weather['DewPointLowF'].astype(float)
austin_weather['HumidityHighPercent'] = austin_weather['HumidityHighPercent'].astype(float)
austin_weather['HumidityAvgPercent'] = austin_weather['HumidityAvgPercent'].astype(float)
austin_weather['HumidityLowPercent'] = austin_weather['HumidityLowPercent'].astype(float)
austin_weather['SeaLevelPressureHighInches'] = austin_weather['SeaLevelPressureHighInches'].astype(float)
austin_weather['SeaLevelPressureAvgInches'] = austin_weather['SeaLevelPressureAvgInches'].astype(float)
austin_weather['SeaLevelPressureLowInches'] = austin_weather['SeaLevelPressureLowInches'].astype(float)
austin_weather['VisibilityHighMiles'] = austin_weather['VisibilityHighMiles'].astype(float)
austin_weather['VisibilityAvgMiles'] = austin_weather['VisibilityAvgMiles'].astype(float)
austin_weather['VisibilityLowMiles'] = austin_weather['VisibilityLowMiles'].astype(float)
austin_weather['WindHighMPH'] = austin_weather['WindHighMPH'].astype(float)
austin_weather['WindAvgMPH'] = austin_weather['WindAvgMPH'].astype(float)
austin_weather['WindGustMPH'] = austin_weather['WindGustMPH'].astype(float)
austin_weather['PrecipitationSumInches'] = austin_weather['PrecipitationSumInches'].astype(float)

#create new dataframe based on Events column, that contain at least "Rain" event, and new dataframe that contains "Sun" event
austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()
austin_sun = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Sun')].copy()
austin_snow = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Snow')].copy()
austin_fog = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Fog')].copy()
austin_thunderstorm = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Thunderstorm')].copy()
```

## Exploratory Data Analysis
We try to create graphs to look at any important cycles that relate to time and the potential different variables that affect weather events.
```
# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Dew Point over Time")
plt.xlabel("Date")
plt.ylabel("Dew Meters")
plt.plot(austin_weather['Date'], austin_weather['DewPointAvgF'])
plt.show()

# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Temperature over Time")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.plot(austin_weather['Date'], austin_weather['TempAvgF'])
plt.show()

# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Humidity over Time")
plt.xlabel("Date")
plt.ylabel("Humidity")
plt.plot(austin_weather['Date'], austin_weather['HumidityAvgPercent'])
plt.show()

# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Precipitation over Time")
plt.xlabel("Date")
plt.ylabel("Precipitation")
plt.plot(austin_weather['Date'], austin_weather['PrecipitationSumInches'])
plt.show()

# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Sea Level Pressure over Time")
plt.xlabel("Date")
plt.ylabel("Sea Level Pressure")
plt.plot(austin_weather['Date'], austin_weather['SeaLevelPressureAvgInches'])
plt.show()

# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Visibility over Time")
plt.xlabel("Date")
plt.ylabel("Visibility")
plt.plot(austin_weather['Date'], austin_weather['VisibilityAvgMiles'])
plt.show()

# Create x, y axis and title of graph
plt.figure(figsize = (14, 12))
plt.title("Wind MPH over Time")
plt.xlabel("Date")
plt.ylabel("Wind MPH")
plt.plot(austin_weather['Date'], austin_weather['WindAvgMPH'])
plt.show()
```
