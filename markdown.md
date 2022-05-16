
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
We make bar charts to the breakdown of weather events over each range of potential variables that affect the event outcomes.
```
#for precipitation
austin_weather_rain1 = austin_weather[austin_weather['PrecipitationSumInches'] <= 1.0]
rain_1 = austin_weather_rain1['Events'].str.contains('Rain').sum()
sun_1 = austin_weather_rain1['Events'].str.contains('Sun').sum()
snow_1 = austin_weather_rain1['Events'].str.contains('Snow').sum()
fog_1 = austin_weather_rain1['Events'].str.contains('Fog').sum()
thunderstorm_1 = austin_weather_rain1['Events'].str.contains('Thunderstorm').sum()

austin_weather_rain2 = austin_weather[austin_weather['PrecipitationSumInches'] <= 2.0]
austin_weather_rain_2 = austin_weather_rain2[austin_weather_rain2['PrecipitationSumInches'] > 1.0]

rain_2 = austin_weather_rain_2['Events'].str.contains('Rain').sum()
sun_2 = austin_weather_rain_2['Events'].str.contains('Sun').sum()
snow_2 = austin_weather_rain_2['Events'].str.contains('Snow').sum()
fog_2 = austin_weather_rain_2['Events'].str.contains('Fog').sum()
thunderstorm_2 = austin_weather_rain_2['Events'].str.contains('Thunderstorm').sum()



austin_weather_rain3 = austin_weather[austin_weather['PrecipitationSumInches'] <= 3.0]
austin_weather_rain_3 = austin_weather_rain3[austin_weather_rain3['PrecipitationSumInches'] > 2.0]

rain_3 = austin_weather_rain_3['Events'].str.contains('Rain').sum()
sun_3 = austin_weather_rain_3['Events'].str.contains('Sun').sum()
snow_3 = austin_weather_rain_3['Events'].str.contains('Snow').sum()
fog_3 = austin_weather_rain_3['Events'].str.contains('Fog').sum()
thunderstorm_3 = austin_weather_rain_3['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain4 = austin_weather[austin_weather['PrecipitationSumInches'] <= 4.0]
austin_weather_rain_4 = austin_weather_rain4[austin_weather_rain4['PrecipitationSumInches'] > 3.0]

rain_4 = austin_weather_rain_4['Events'].str.contains('Rain').sum()
sun_4 = austin_weather_rain_4['Events'].str.contains('Sun').sum()
snow_4 = austin_weather_rain_4['Events'].str.contains('Snow').sum()
fog_4 = austin_weather_rain_4['Events'].str.contains('Fog').sum()
thunderstorm_4 = austin_weather_rain_4['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain5 = austin_weather[austin_weather['PrecipitationSumInches'] <= 10.0]
austin_weather_rain_5 = austin_weather_rain5[austin_weather_rain5['PrecipitationSumInches'] > 4.0]

rain_5 = austin_weather_rain_5['Events'].str.contains('Rain').sum()
sun_5 = austin_weather_rain_5['Events'].str.contains('Sun').sum()
snow_5 = austin_weather_rain_5['Events'].str.contains('Snow').sum()
fog_5 = austin_weather_rain_5['Events'].str.contains('Fog').sum()
thunderstorm_5 = austin_weather_rain_5['Events'].str.contains('Thunderstorm').sum()

labels = ['Prec. <= 1 in', '1 in < Prec. <= 2 in', '2 in < Prec. <= 3 in', '3 in < Prec. <= 4 in', '4 in < Prec. <= 10 in']
rain_arr = [rain_1, rain_2, rain_3, rain_4, rain_5]
sun_arr = [sun_1, sun_2, sun_3, sun_4, sun_5]
snow_arr = [snow_1, snow_2, snow_3, snow_4, snow_5]
fog_arr = [fog_1, fog_2, fog_3, fog_4, fog_5]
thunderstorm_arr = [thunderstorm_1, thunderstorm_2, thunderstorm_3, thunderstorm_4, thunderstorm_5]


x = np.arange(len(labels))  # the label locations
width = 0.16  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - (width*2), rain_arr, width, label='Rain')
rects2 = ax.bar(x - width, sun_arr, width, label='Sun')
rects3 = ax.bar(x , snow_arr, width, label='Snow')
rects4 = ax.bar(x + width, fog_arr, width, label='Fog')
rects5 = ax.bar(x + (width*2), thunderstorm_arr, width, label='Thunderstorm')

fig.set_size_inches(14, 12)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Occurences')
ax.set_title('Precipitation Levels of Different Events')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=4)
ax.bar_label(rects2, padding=4)
ax.bar_label(rects3, padding=4)
ax.bar_label(rects4, padding=4)
ax.bar_label(rects5, padding=4)


fig.tight_layout()

plt.show()

#for dew point average
austin_weather_rain1 = austin_weather[austin_weather['DewPointAvgF'] <= 20.0]
rain_1 = austin_weather_rain1['Events'].str.contains('Rain').sum()
sun_1 = austin_weather_rain1['Events'].str.contains('Sun').sum()
snow_1 = austin_weather_rain1['Events'].str.contains('Snow').sum()
fog_1 = austin_weather_rain1['Events'].str.contains('Fog').sum()
thunderstorm_1 = austin_weather_rain1['Events'].str.contains('Thunderstorm').sum()

austin_weather_rain2 = austin_weather[austin_weather['DewPointAvgF'] <= 40.0]
austin_weather_rain_2 = austin_weather_rain2[austin_weather_rain2['DewPointAvgF'] > 20.0]

rain_2 = austin_weather_rain_2['Events'].str.contains('Rain').sum()
sun_2 = austin_weather_rain_2['Events'].str.contains('Sun').sum()
snow_2 = austin_weather_rain_2['Events'].str.contains('Snow').sum()
fog_2 = austin_weather_rain_2['Events'].str.contains('Fog').sum()
thunderstorm_2 = austin_weather_rain_2['Events'].str.contains('Thunderstorm').sum()



austin_weather_rain3 = austin_weather[austin_weather['DewPointAvgF'] <= 60.0]
austin_weather_rain_3 = austin_weather_rain3[austin_weather_rain3['DewPointAvgF'] > 40.0]

rain_3 = austin_weather_rain_3['Events'].str.contains('Rain').sum()
sun_3 = austin_weather_rain_3['Events'].str.contains('Sun').sum()
snow_3 = austin_weather_rain_3['Events'].str.contains('Snow').sum()
fog_3 = austin_weather_rain_3['Events'].str.contains('Fog').sum()
thunderstorm_3 = austin_weather_rain_3['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain4 = austin_weather[austin_weather['DewPointAvgF'] <= 80.0]
austin_weather_rain_4 = austin_weather_rain4[austin_weather_rain4['DewPointAvgF'] > 60.0]

rain_4 = austin_weather_rain_4['Events'].str.contains('Rain').sum()
sun_4 = austin_weather_rain_4['Events'].str.contains('Sun').sum()
snow_4 = austin_weather_rain_4['Events'].str.contains('Snow').sum()
fog_4 = austin_weather_rain_4['Events'].str.contains('Fog').sum()
thunderstorm_4 = austin_weather_rain_4['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain5 = austin_weather[austin_weather['DewPointAvgF'] <= 100.0]
austin_weather_rain_5 = austin_weather_rain5[austin_weather_rain5['DewPointAvgF'] > 80.0]

rain_5 = austin_weather_rain_5['Events'].str.contains('Rain').sum()
sun_5 = austin_weather_rain_5['Events'].str.contains('Sun').sum()
snow_5 = austin_weather_rain_5['Events'].str.contains('Snow').sum()
fog_5 = austin_weather_rain_5['Events'].str.contains('Fog').sum()
thunderstorm_5 = austin_weather_rain_5['Events'].str.contains('Thunderstorm').sum()

labels = ['DewP <= 20 ', '20 < DewP <= 40 ', '40 < DewP <= 60 ', '60 < DewP <= 80 ', '80 < DewP <= 100 ']
rain_arr = [rain_1, rain_2, rain_3, rain_4, rain_5]
sun_arr = [sun_1, sun_2, sun_3, sun_4, sun_5]
snow_arr = [snow_1, snow_2, snow_3, snow_4, snow_5]
fog_arr = [fog_1, fog_2, fog_3, fog_4, fog_5]
thunderstorm_arr = [thunderstorm_1, thunderstorm_2, thunderstorm_3, thunderstorm_4, thunderstorm_5]


x = np.arange(len(labels))  # the label locations
width = 0.16  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - (width*2), rain_arr, width, label='Rain')
rects2 = ax.bar(x - width, sun_arr, width, label='Sun')
rects3 = ax.bar(x , snow_arr, width, label='Snow')
rects4 = ax.bar(x + width, fog_arr, width, label='Fog')
rects5 = ax.bar(x + (width*2), thunderstorm_arr, width, label='Thunderstorm')

fig.set_size_inches(14, 12)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Occurences')
ax.set_title('Dew Point of Different Events')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=4)
ax.bar_label(rects2, padding=4)
ax.bar_label(rects3, padding=4)
ax.bar_label(rects4, padding=4)
ax.bar_label(rects5, padding=4)



fig.tight_layout()

plt.show()

#for humidity
austin_weather_rain1 = austin_weather[austin_weather['HumidityAvgPercent'] <= 20.0]
rain_1 = austin_weather_rain1['Events'].str.contains('Rain').sum()
sun_1 = austin_weather_rain1['Events'].str.contains('Sun').sum()
snow_1 = austin_weather_rain1['Events'].str.contains('Snow').sum()
fog_1 = austin_weather_rain1['Events'].str.contains('Fog').sum()
thunderstorm_1 = austin_weather_rain1['Events'].str.contains('Thunderstorm').sum()

austin_weather_rain2 = austin_weather[austin_weather['HumidityAvgPercent'] <= 40.0]
austin_weather_rain_2 = austin_weather_rain2[austin_weather_rain2['HumidityAvgPercent'] > 20.0]

rain_2 = austin_weather_rain_2['Events'].str.contains('Rain').sum()
sun_2 = austin_weather_rain_2['Events'].str.contains('Sun').sum()
snow_2 = austin_weather_rain_2['Events'].str.contains('Snow').sum()
fog_2 = austin_weather_rain_2['Events'].str.contains('Fog').sum()
thunderstorm_2 = austin_weather_rain_2['Events'].str.contains('Thunderstorm').sum()



austin_weather_rain3 = austin_weather[austin_weather['HumidityAvgPercent'] <= 60.0]
austin_weather_rain_3 = austin_weather_rain3[austin_weather_rain3['HumidityAvgPercent'] > 40.0]

rain_3 = austin_weather_rain_3['Events'].str.contains('Rain').sum()
sun_3 = austin_weather_rain_3['Events'].str.contains('Sun').sum()
snow_3 = austin_weather_rain_3['Events'].str.contains('Snow').sum()
fog_3 = austin_weather_rain_3['Events'].str.contains('Fog').sum()
thunderstorm_3 = austin_weather_rain_3['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain4 = austin_weather[austin_weather['HumidityAvgPercent'] <= 80.0]
austin_weather_rain_4 = austin_weather_rain4[austin_weather_rain4['HumidityAvgPercent'] > 60.0]

rain_4 = austin_weather_rain_4['Events'].str.contains('Rain').sum()
sun_4 = austin_weather_rain_4['Events'].str.contains('Sun').sum()
snow_4 = austin_weather_rain_4['Events'].str.contains('Snow').sum()
fog_4 = austin_weather_rain_4['Events'].str.contains('Fog').sum()
thunderstorm_4 = austin_weather_rain_4['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain5 = austin_weather[austin_weather['HumidityAvgPercent'] <= 100.0]
austin_weather_rain_5 = austin_weather_rain5[austin_weather_rain5['HumidityAvgPercent'] > 80.0]

rain_5 = austin_weather_rain_5['Events'].str.contains('Rain').sum()
sun_5 = austin_weather_rain_5['Events'].str.contains('Sun').sum()
snow_5 = austin_weather_rain_5['Events'].str.contains('Snow').sum()
fog_5 = austin_weather_rain_5['Events'].str.contains('Fog').sum()
thunderstorm_5 = austin_weather_rain_5['Events'].str.contains('Thunderstorm').sum()

labels = ['Humidity <= 20 ', '20 < Humidity <= 40 ', '40 < Humidity <= 60 ', '60 < Humidity <= 80 ', '80 < Humidity <= 100 ']
rain_arr = [rain_1, rain_2, rain_3, rain_4, rain_5]
sun_arr = [sun_1, sun_2, sun_3, sun_4, sun_5]
snow_arr = [snow_1, snow_2, snow_3, snow_4, snow_5]
fog_arr = [fog_1, fog_2, fog_3, fog_4, fog_5]
thunderstorm_arr = [thunderstorm_1, thunderstorm_2, thunderstorm_3, thunderstorm_4, thunderstorm_5]


x = np.arange(len(labels))  # the label locations
width = 0.16  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - (width*2), rain_arr, width, label='Rain')
rects2 = ax.bar(x - width, sun_arr, width, label='Sun')
rects3 = ax.bar(x , snow_arr, width, label='Snow')
rects4 = ax.bar(x + width, fog_arr, width, label='Fog')
rects5 = ax.bar(x + (width*2), thunderstorm_arr, width, label='Thunderstorm')

fig.set_size_inches(14, 12)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Occurences')
ax.set_title('Groups of Dew Points, by event occurences')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=4)
ax.bar_label(rects2, padding=4)
ax.bar_label(rects3, padding=4)
ax.bar_label(rects4, padding=4)
ax.bar_label(rects5, padding=4)

fig.tight_layout()

plt.show()
```
Here we are trying to find the best relationship, between 2 measurements and each event that occurs.
# only plotted 2 for ease of analysis for now
# blue is rain, orange is sun 
```
austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()

# plot for sea level and humidity
fig, ax = plt.subplots()
ax.scatter(austin_rain['SeaLevelPressureAvgInches'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['SeaLevelPressureAvgInches'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.set(xlim=(26, 34), xticks=np.arange(26, 34),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('Sea Level Pressure Avg Inches', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Sea Level vs Humidity')
fig.set_size_inches(10, 8)

plt.show()
```
Here we can see that SeaLevelPressure is a negligible factor in between all 5 different events.
While we can rule out Sea level Pressure as a contributing factor, rain seems to occur more often while there is higher humidity, so we can look further into that.
```
# plot for humidity and dew point
fig, ax = plt.subplots()
ax.scatter(austin_rain['DewPointAvgF'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['DewPointAvgF'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)

ax.set(xlim=(0, 100), xticks=np.arange(0, 100),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('DewPointAvgF', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Dew Point vs Humidity')
fig.set_size_inches(10, 8)

plt.show()
```
### MAKE A COMMENT ON HUMIDITY VS DEW POINT
```
# plot for temp and humidity
fig, ax = plt.subplots()
ax.scatter(austin_rain['TempAvgF'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['TempAvgF'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.set(xlim=(0, 100), xticks=np.arange(0, 100),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('TempAvgF', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Temperature vs Humidity')
fig.set_size_inches(12, 8)

plt.show()
```
### MAKE A COMMENT ON TEMP VS HUMIDITY
```
# plot for visibility and humidity
fig, ax = plt.subplots()
ax.scatter(austin_rain['VisibilityAvgMiles'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['VisibilityAvgMiles'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.set(xlim=(0, 12), xticks=np.arange(0, 12),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('VisibilityAvgMiles', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Visibility vs Humidity')
fig.set_size_inches(12, 8)

plt.show()
```
Here we can see that both humidity and visiabliliy are contributing factors towards the event.
With high humidity and low visibiilty we can see an increase in the event rain, and with low humidity and 
high visibility, we can see a increase in the event sun.

## Hypothesis Testing and Classification Methods
We want to determine a good model that will determine what the weather event is based on variable information. Through our exploration of the past data, we have recognized that humidity and visibility may be good variables that can predict what the weather event will be. Since we will be using to data to determine a type of event, we need to use a classification method. For ease, we have chosen to try to determine a classification technique using linear SVM. However, we need to determine which variable will be more useful in giving an accurate classification. In order to determine this, we will be performing a paired t-test. We will compare the difference 
