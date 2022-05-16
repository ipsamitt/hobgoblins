```python
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
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

```

Scraping the data:
    We will be obtaining the data from https://www.kaggle.com/datasets/grubenm/austin-weather

The weather data, is a subset of the overall, Austin Bike Shares Trip mobile app database
    https://www.kaggle.com/datasets/jboysen/austin-bike 
   
We wanted to do something to improve the overall Austin Bike Shares Trip mobile app, so we went with weather prediction
in order to ensure the safety of riders.
Having accurate weather prediciton daily allows riders to safely prepare for whatever inclimate weather conditions could occur.


```python
austin_weather = pd.read_csv("austin_weather.csv")
austin_weather
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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>SeaLevelPressureAvgInches</th>
      <th>SeaLevelPressureLowInches</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-12-21</td>
      <td>74</td>
      <td>60</td>
      <td>45</td>
      <td>67</td>
      <td>49</td>
      <td>43</td>
      <td>93</td>
      <td>75</td>
      <td>57</td>
      <td>...</td>
      <td>29.68</td>
      <td>29.59</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>20</td>
      <td>4</td>
      <td>31</td>
      <td>0.46</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-12-22</td>
      <td>56</td>
      <td>48</td>
      <td>39</td>
      <td>43</td>
      <td>36</td>
      <td>28</td>
      <td>93</td>
      <td>68</td>
      <td>43</td>
      <td>...</td>
      <td>30.13</td>
      <td>29.87</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
      <td>6</td>
      <td>25</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-12-23</td>
      <td>58</td>
      <td>45</td>
      <td>32</td>
      <td>31</td>
      <td>27</td>
      <td>23</td>
      <td>76</td>
      <td>52</td>
      <td>27</td>
      <td>...</td>
      <td>30.49</td>
      <td>30.41</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-24</td>
      <td>61</td>
      <td>46</td>
      <td>31</td>
      <td>36</td>
      <td>28</td>
      <td>21</td>
      <td>89</td>
      <td>56</td>
      <td>22</td>
      <td>...</td>
      <td>30.45</td>
      <td>30.3</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>12</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-12-25</td>
      <td>58</td>
      <td>50</td>
      <td>41</td>
      <td>44</td>
      <td>40</td>
      <td>36</td>
      <td>86</td>
      <td>71</td>
      <td>56</td>
      <td>...</td>
      <td>30.33</td>
      <td>30.27</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>10</td>
      <td>2</td>
      <td>16</td>
      <td>T</td>
      <td></td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>2017-07-27</td>
      <td>103</td>
      <td>89</td>
      <td>75</td>
      <td>71</td>
      <td>67</td>
      <td>61</td>
      <td>82</td>
      <td>54</td>
      <td>25</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.88</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>5</td>
      <td>21</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1315</th>
      <td>2017-07-28</td>
      <td>105</td>
      <td>91</td>
      <td>76</td>
      <td>71</td>
      <td>64</td>
      <td>55</td>
      <td>87</td>
      <td>54</td>
      <td>20</td>
      <td>...</td>
      <td>29.9</td>
      <td>29.81</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>14</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2017-07-29</td>
      <td>107</td>
      <td>92</td>
      <td>77</td>
      <td>72</td>
      <td>64</td>
      <td>55</td>
      <td>82</td>
      <td>51</td>
      <td>19</td>
      <td>...</td>
      <td>29.86</td>
      <td>29.79</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>4</td>
      <td>17</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1317</th>
      <td>2017-07-30</td>
      <td>106</td>
      <td>93</td>
      <td>79</td>
      <td>70</td>
      <td>68</td>
      <td>63</td>
      <td>69</td>
      <td>48</td>
      <td>27</td>
      <td>...</td>
      <td>29.91</td>
      <td>29.87</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>13</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1318</th>
      <td>2017-07-31</td>
      <td>99</td>
      <td>88</td>
      <td>77</td>
      <td>66</td>
      <td>61</td>
      <td>54</td>
      <td>64</td>
      <td>43</td>
      <td>22</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.91</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>1319 rows × 21 columns</p>
</div>



For the "Events" Column which describes the condition of the weather for the day, all blanks will be turned into "sun".
In the precipitation category, 'T' represents 'Trace', an amount measuring less than 0.01 inch. We will replace all 'T' with 0.01.
Converted the string dates into panda's datetimes objects for better ease of sorting later on.


```python
#Converting string dates into datetime.date objects
for i, currRow in austin_weather.iterrows():
    austin_weather.at[i, 'Date'] = datetime.strptime(austin_weather.at[i, 'Date'], "%Y-%m-%d").date()

austin_weather['Events'].replace(' ', 'Sun', inplace=True)
austin_weather['PrecipitationSumInches'].replace('T', 0.01, inplace=True)
austin_weather
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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>SeaLevelPressureAvgInches</th>
      <th>SeaLevelPressureLowInches</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-12-21</td>
      <td>74</td>
      <td>60</td>
      <td>45</td>
      <td>67</td>
      <td>49</td>
      <td>43</td>
      <td>93</td>
      <td>75</td>
      <td>57</td>
      <td>...</td>
      <td>29.68</td>
      <td>29.59</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>20</td>
      <td>4</td>
      <td>31</td>
      <td>0.46</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-12-22</td>
      <td>56</td>
      <td>48</td>
      <td>39</td>
      <td>43</td>
      <td>36</td>
      <td>28</td>
      <td>93</td>
      <td>68</td>
      <td>43</td>
      <td>...</td>
      <td>30.13</td>
      <td>29.87</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
      <td>6</td>
      <td>25</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-12-23</td>
      <td>58</td>
      <td>45</td>
      <td>32</td>
      <td>31</td>
      <td>27</td>
      <td>23</td>
      <td>76</td>
      <td>52</td>
      <td>27</td>
      <td>...</td>
      <td>30.49</td>
      <td>30.41</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-24</td>
      <td>61</td>
      <td>46</td>
      <td>31</td>
      <td>36</td>
      <td>28</td>
      <td>21</td>
      <td>89</td>
      <td>56</td>
      <td>22</td>
      <td>...</td>
      <td>30.45</td>
      <td>30.3</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>12</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-12-25</td>
      <td>58</td>
      <td>50</td>
      <td>41</td>
      <td>44</td>
      <td>40</td>
      <td>36</td>
      <td>86</td>
      <td>71</td>
      <td>56</td>
      <td>...</td>
      <td>30.33</td>
      <td>30.27</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>10</td>
      <td>2</td>
      <td>16</td>
      <td>0.01</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>2017-07-27</td>
      <td>103</td>
      <td>89</td>
      <td>75</td>
      <td>71</td>
      <td>67</td>
      <td>61</td>
      <td>82</td>
      <td>54</td>
      <td>25</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.88</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>5</td>
      <td>21</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>2017-07-28</td>
      <td>105</td>
      <td>91</td>
      <td>76</td>
      <td>71</td>
      <td>64</td>
      <td>55</td>
      <td>87</td>
      <td>54</td>
      <td>20</td>
      <td>...</td>
      <td>29.9</td>
      <td>29.81</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>14</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2017-07-29</td>
      <td>107</td>
      <td>92</td>
      <td>77</td>
      <td>72</td>
      <td>64</td>
      <td>55</td>
      <td>82</td>
      <td>51</td>
      <td>19</td>
      <td>...</td>
      <td>29.86</td>
      <td>29.79</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>4</td>
      <td>17</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>2017-07-30</td>
      <td>106</td>
      <td>93</td>
      <td>79</td>
      <td>70</td>
      <td>68</td>
      <td>63</td>
      <td>69</td>
      <td>48</td>
      <td>27</td>
      <td>...</td>
      <td>29.91</td>
      <td>29.87</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>13</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1318</th>
      <td>2017-07-31</td>
      <td>99</td>
      <td>88</td>
      <td>77</td>
      <td>66</td>
      <td>61</td>
      <td>54</td>
      <td>64</td>
      <td>43</td>
      <td>22</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.91</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
  </tbody>
</table>
<p>1319 rows × 21 columns</p>
</div>



This dataset displayed missing data as '-' and has a total of 80 missing data points across all columns.
As this data is negliable in consideration to the 1030 dataset, we have decided to remove it.


```python
#austin_weather[TempHighF]
austin_weather = austin_weather.replace('-', np.NaN, regex=True)
austin_weather.isnull().sum(axis = 0)

```




    Date                           0
    TempHighF                      0
    TempAvgF                       0
    TempLowF                       0
    DewPointHighF                  7
    DewPointAvgF                   7
    DewPointLowF                   7
    HumidityHighPercent            2
    HumidityAvgPercent             2
    HumidityLowPercent             2
    SeaLevelPressureHighInches     3
    SeaLevelPressureAvgInches      3
    SeaLevelPressureLowInches      3
    VisibilityHighMiles           12
    VisibilityAvgMiles            12
    VisibilityLowMiles            12
    WindHighMPH                    2
    WindAvgMPH                     2
    WindGustMPH                    4
    PrecipitationSumInches         0
    Events                         0
    dtype: int64




```python
austin_weather = austin_weather.dropna()
austin_weather
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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>SeaLevelPressureAvgInches</th>
      <th>SeaLevelPressureLowInches</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-12-21</td>
      <td>74</td>
      <td>60</td>
      <td>45</td>
      <td>67</td>
      <td>49</td>
      <td>43</td>
      <td>93</td>
      <td>75</td>
      <td>57</td>
      <td>...</td>
      <td>29.68</td>
      <td>29.59</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>20</td>
      <td>4</td>
      <td>31</td>
      <td>0.46</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-12-22</td>
      <td>56</td>
      <td>48</td>
      <td>39</td>
      <td>43</td>
      <td>36</td>
      <td>28</td>
      <td>93</td>
      <td>68</td>
      <td>43</td>
      <td>...</td>
      <td>30.13</td>
      <td>29.87</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
      <td>6</td>
      <td>25</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-12-23</td>
      <td>58</td>
      <td>45</td>
      <td>32</td>
      <td>31</td>
      <td>27</td>
      <td>23</td>
      <td>76</td>
      <td>52</td>
      <td>27</td>
      <td>...</td>
      <td>30.49</td>
      <td>30.41</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-24</td>
      <td>61</td>
      <td>46</td>
      <td>31</td>
      <td>36</td>
      <td>28</td>
      <td>21</td>
      <td>89</td>
      <td>56</td>
      <td>22</td>
      <td>...</td>
      <td>30.45</td>
      <td>30.3</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>12</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-12-25</td>
      <td>58</td>
      <td>50</td>
      <td>41</td>
      <td>44</td>
      <td>40</td>
      <td>36</td>
      <td>86</td>
      <td>71</td>
      <td>56</td>
      <td>...</td>
      <td>30.33</td>
      <td>30.27</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>10</td>
      <td>2</td>
      <td>16</td>
      <td>0.01</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>2017-07-27</td>
      <td>103</td>
      <td>89</td>
      <td>75</td>
      <td>71</td>
      <td>67</td>
      <td>61</td>
      <td>82</td>
      <td>54</td>
      <td>25</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.88</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>5</td>
      <td>21</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>2017-07-28</td>
      <td>105</td>
      <td>91</td>
      <td>76</td>
      <td>71</td>
      <td>64</td>
      <td>55</td>
      <td>87</td>
      <td>54</td>
      <td>20</td>
      <td>...</td>
      <td>29.9</td>
      <td>29.81</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>14</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2017-07-29</td>
      <td>107</td>
      <td>92</td>
      <td>77</td>
      <td>72</td>
      <td>64</td>
      <td>55</td>
      <td>82</td>
      <td>51</td>
      <td>19</td>
      <td>...</td>
      <td>29.86</td>
      <td>29.79</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>4</td>
      <td>17</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>2017-07-30</td>
      <td>106</td>
      <td>93</td>
      <td>79</td>
      <td>70</td>
      <td>68</td>
      <td>63</td>
      <td>69</td>
      <td>48</td>
      <td>27</td>
      <td>...</td>
      <td>29.91</td>
      <td>29.87</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>13</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1318</th>
      <td>2017-07-31</td>
      <td>99</td>
      <td>88</td>
      <td>77</td>
      <td>66</td>
      <td>61</td>
      <td>54</td>
      <td>64</td>
      <td>43</td>
      <td>22</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.91</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>Sun</td>
    </tr>
  </tbody>
</table>
<p>1305 rows × 21 columns</p>
</div>



In order to remove discrepancies in calculations, we will convert everything to float values


```python
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
austin_weather
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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>SeaLevelPressureAvgInches</th>
      <th>SeaLevelPressureLowInches</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-12-21</td>
      <td>74.0</td>
      <td>60.0</td>
      <td>45.0</td>
      <td>67.0</td>
      <td>49.0</td>
      <td>43.0</td>
      <td>93.0</td>
      <td>75.0</td>
      <td>57.0</td>
      <td>...</td>
      <td>29.68</td>
      <td>29.59</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>4.0</td>
      <td>31.0</td>
      <td>0.46</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-12-22</td>
      <td>56.0</td>
      <td>48.0</td>
      <td>39.0</td>
      <td>43.0</td>
      <td>36.0</td>
      <td>28.0</td>
      <td>93.0</td>
      <td>68.0</td>
      <td>43.0</td>
      <td>...</td>
      <td>30.13</td>
      <td>29.87</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-12-23</td>
      <td>58.0</td>
      <td>45.0</td>
      <td>32.0</td>
      <td>31.0</td>
      <td>27.0</td>
      <td>23.0</td>
      <td>76.0</td>
      <td>52.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>30.49</td>
      <td>30.41</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-24</td>
      <td>61.0</td>
      <td>46.0</td>
      <td>31.0</td>
      <td>36.0</td>
      <td>28.0</td>
      <td>21.0</td>
      <td>89.0</td>
      <td>56.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>30.45</td>
      <td>30.30</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-12-25</td>
      <td>58.0</td>
      <td>50.0</td>
      <td>41.0</td>
      <td>44.0</td>
      <td>40.0</td>
      <td>36.0</td>
      <td>86.0</td>
      <td>71.0</td>
      <td>56.0</td>
      <td>...</td>
      <td>30.33</td>
      <td>30.27</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>0.01</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>2017-07-27</td>
      <td>103.0</td>
      <td>89.0</td>
      <td>75.0</td>
      <td>71.0</td>
      <td>67.0</td>
      <td>61.0</td>
      <td>82.0</td>
      <td>54.0</td>
      <td>25.0</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.88</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>5.0</td>
      <td>21.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>2017-07-28</td>
      <td>105.0</td>
      <td>91.0</td>
      <td>76.0</td>
      <td>71.0</td>
      <td>64.0</td>
      <td>55.0</td>
      <td>87.0</td>
      <td>54.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>29.90</td>
      <td>29.81</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2017-07-29</td>
      <td>107.0</td>
      <td>92.0</td>
      <td>77.0</td>
      <td>72.0</td>
      <td>64.0</td>
      <td>55.0</td>
      <td>82.0</td>
      <td>51.0</td>
      <td>19.0</td>
      <td>...</td>
      <td>29.86</td>
      <td>29.79</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>2017-07-30</td>
      <td>106.0</td>
      <td>93.0</td>
      <td>79.0</td>
      <td>70.0</td>
      <td>68.0</td>
      <td>63.0</td>
      <td>69.0</td>
      <td>48.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>29.91</td>
      <td>29.87</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1318</th>
      <td>2017-07-31</td>
      <td>99.0</td>
      <td>88.0</td>
      <td>77.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>54.0</td>
      <td>64.0</td>
      <td>43.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.91</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
  </tbody>
</table>
<p>1305 rows × 21 columns</p>
</div>




```python
#create new dataframe based on Events column, that contain at least "Rain" event, and new dataframe that contains "Sun" event
austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()
austin_sun = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Sun')].copy()
austin_snow = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Snow')].copy()
austin_fog = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Fog')].copy()
austin_thunderstorm = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Thunderstorm')].copy()

austin_rain
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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>SeaLevelPressureAvgInches</th>
      <th>SeaLevelPressureLowInches</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-12-21</td>
      <td>74.0</td>
      <td>60.0</td>
      <td>45.0</td>
      <td>67.0</td>
      <td>49.0</td>
      <td>43.0</td>
      <td>93.0</td>
      <td>75.0</td>
      <td>57.0</td>
      <td>...</td>
      <td>29.68</td>
      <td>29.59</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>4.0</td>
      <td>31.0</td>
      <td>0.46</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2014-01-08</td>
      <td>53.0</td>
      <td>47.0</td>
      <td>40.0</td>
      <td>51.0</td>
      <td>45.0</td>
      <td>30.0</td>
      <td>93.0</td>
      <td>75.0</td>
      <td>57.0</td>
      <td>...</td>
      <td>30.20</td>
      <td>30.12</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>0.16</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2014-01-10</td>
      <td>72.0</td>
      <td>65.0</td>
      <td>57.0</td>
      <td>64.0</td>
      <td>61.0</td>
      <td>54.0</td>
      <td>93.0</td>
      <td>81.0</td>
      <td>68.0</td>
      <td>...</td>
      <td>29.87</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>21.0</td>
      <td>0.10</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2014-01-12</td>
      <td>67.0</td>
      <td>57.0</td>
      <td>46.0</td>
      <td>58.0</td>
      <td>47.0</td>
      <td>33.0</td>
      <td>84.0</td>
      <td>68.0</td>
      <td>52.0</td>
      <td>...</td>
      <td>30.01</td>
      <td>29.90</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>0.01</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2014-01-23</td>
      <td>56.0</td>
      <td>42.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>29.0</td>
      <td>20.0</td>
      <td>78.0</td>
      <td>64.0</td>
      <td>50.0</td>
      <td>...</td>
      <td>30.47</td>
      <td>30.06</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>9.0</td>
      <td>31.0</td>
      <td>0.06</td>
      <td>Rain , Snow</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>2017-07-07</td>
      <td>99.0</td>
      <td>87.0</td>
      <td>75.0</td>
      <td>74.0</td>
      <td>71.0</td>
      <td>65.0</td>
      <td>88.0</td>
      <td>61.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>30.05</td>
      <td>29.98</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>27.0</td>
      <td>0.02</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>2017-07-15</td>
      <td>103.0</td>
      <td>87.0</td>
      <td>71.0</td>
      <td>73.0</td>
      <td>70.0</td>
      <td>63.0</td>
      <td>100.0</td>
      <td>65.0</td>
      <td>29.0</td>
      <td>...</td>
      <td>30.01</td>
      <td>29.94</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>23.0</td>
      <td>3.0</td>
      <td>35.0</td>
      <td>0.16</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>2017-07-17</td>
      <td>98.0</td>
      <td>88.0</td>
      <td>77.0</td>
      <td>75.0</td>
      <td>71.0</td>
      <td>66.0</td>
      <td>88.0</td>
      <td>63.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>29.90</td>
      <td>29.84</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>23.0</td>
      <td>0.01</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1310</th>
      <td>2017-07-23</td>
      <td>103.0</td>
      <td>90.0</td>
      <td>77.0</td>
      <td>74.0</td>
      <td>71.0</td>
      <td>66.0</td>
      <td>85.0</td>
      <td>58.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>29.88</td>
      <td>29.82</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>0.04</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1311</th>
      <td>2017-07-24</td>
      <td>102.0</td>
      <td>89.0</td>
      <td>76.0</td>
      <td>75.0</td>
      <td>71.0</td>
      <td>56.0</td>
      <td>91.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>...</td>
      <td>29.95</td>
      <td>29.89</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>0.01</td>
      <td>Rain , Thunderstorm</td>
    </tr>
  </tbody>
</table>
<p>375 rows × 21 columns</p>
</div>



TRying to exmplore whataevr the hecko this is

Ideas)

Part 1)

1. Graph the date vs temp/humid/precipitation, to see patterns over course of year.
    We want to split date up by (TASK DONE BY IPSA)
    
2. Bar graph split on x-axis base on a certain condition (precipitaiton, humidity, temperature), and in each section (for example 0-4 inches 4-8 inches, have sub-categories for each of the Events (Rain, Snow, Fog, Thunderstorm etc)


Part 2)
    Use graphs from part 1, to get a deeper understanding of what conditions most effect the type of Event (Snow, Rain, Thunder, etc)
    Once deeper insight has been gained then we will know which condition to focus our training data on to predicit what the Weather or 'Event' will be (Rain, Snow, Thunder)
    (Use training data that is completely, random) 400 pts, remaining data points will be testing points
    
Hypothesis test)
    null hypothesis - assuming everything to be true
    alternate hypothesis - what you are testing - certain condition (humidity, temp, etc) has an effect on the type of 
    note - ipsa - hypothesis test, is there an average difference in temperature based on sunny or rainy?
    


```python
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


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



    
![png](output_12_5.png)
    



    
![png](output_12_6.png)
    



```python
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
```


    
![png](output_13_0.png)
    



```python
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
```


    
![png](output_14_0.png)
    



```python
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


    
![png](output_15_0.png)
    



```python
austin_weather_rain2 = austin_weather[austin_weather['HumidityAvgPercent'] <= 70.0]
austin_weather_rain_2 = austin_weather_rain2[austin_weather_rain2['HumidityAvgPercent'] > 60.0]

rain_2 = austin_weather_rain_2['Events'].str.contains('Rain').sum()
sun_2 = austin_weather_rain_2['Events'].str.contains('Sun').sum()
snow_2 = austin_weather_rain_2['Events'].str.contains('Snow').sum()
fog_2 = austin_weather_rain_2['Events'].str.contains('Fog').sum()
thunderstorm_2 = austin_weather_rain_2['Events'].str.contains('Thunderstorm').sum()



austin_weather_rain3 = austin_weather[austin_weather['HumidityAvgPercent'] <= 80.0]
austin_weather_rain_3 = austin_weather_rain3[austin_weather_rain3['HumidityAvgPercent'] > 70.0]

rain_3 = austin_weather_rain_3['Events'].str.contains('Rain').sum()
sun_3 = austin_weather_rain_3['Events'].str.contains('Sun').sum()
snow_3 = austin_weather_rain_3['Events'].str.contains('Snow').sum()
fog_3 = austin_weather_rain_3['Events'].str.contains('Fog').sum()
thunderstorm_3 = austin_weather_rain_3['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain4 = austin_weather[austin_weather['HumidityAvgPercent'] <= 90.0]
austin_weather_rain_4 = austin_weather_rain4[austin_weather_rain4['HumidityAvgPercent'] > 80.0]

rain_4 = austin_weather_rain_4['Events'].str.contains('Rain').sum()
sun_4 = austin_weather_rain_4['Events'].str.contains('Sun').sum()
snow_4 = austin_weather_rain_4['Events'].str.contains('Snow').sum()
fog_4 = austin_weather_rain_4['Events'].str.contains('Fog').sum()
thunderstorm_4 = austin_weather_rain_4['Events'].str.contains('Thunderstorm').sum()


austin_weather_rain5 = austin_weather[austin_weather['HumidityAvgPercent'] <= 100.0]
austin_weather_rain_5 = austin_weather_rain5[austin_weather_rain5['HumidityAvgPercent'] > 90.0]

rain_5 = austin_weather_rain_5['Events'].str.contains('Rain').sum()
sun_5 = austin_weather_rain_5['Events'].str.contains('Sun').sum()
snow_5 = austin_weather_rain_5['Events'].str.contains('Snow').sum()
fog_5 = austin_weather_rain_5['Events'].str.contains('Fog').sum()
thunderstorm_5 = austin_weather_rain_5['Events'].str.contains('Thunderstorm').sum()

labels = ['60 < Humidity <= 70', '70 < Humidity <= 80', '80 < Humidity <= 90', '90 < Humidity <= 100']
rain_arr = [rain_2, rain_3, rain_4, rain_5]
sun_arr = [sun_2, sun_3, sun_4, sun_5]
snow_arr = [snow_2, snow_3, snow_4, snow_5]
fog_arr = [fog_2, fog_3, fog_4, fog_5]
thunderstorm_arr = [thunderstorm_2, thunderstorm_3, thunderstorm_4, thunderstorm_5]


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


    
![png](output_16_0.png)
    



```python
austin_weather.columns
```




    Index(['Date', 'TempHighF', 'TempAvgF', 'TempLowF', 'DewPointHighF',
           'DewPointAvgF', 'DewPointLowF', 'HumidityHighPercent',
           'HumidityAvgPercent', 'HumidityLowPercent',
           'SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches',
           'SeaLevelPressureLowInches', 'VisibilityHighMiles',
           'VisibilityAvgMiles', 'VisibilityLowMiles', 'WindHighMPH', 'WindAvgMPH',
           'WindGustMPH', 'PrecipitationSumInches', 'Events'],
          dtype='object')



Here we are trying to find the best relationship, between 2 measurements and event that occurs.


```python
#only plotted 2 for ease of analysis for now
#blue is rain, orange is sun 

austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()

# plot
fig, ax = plt.subplots()
#HumidityAvgPercent SeaLevelPressureAvgInches
ax.scatter(austin_rain['SeaLevelPressureAvgInches'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['SeaLevelPressureAvgInches'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_snow['SeaLevelPressureAvgInches'], austin_snow['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_fog['SeaLevelPressureAvgInches'], austin_fog['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_thunderstorm['SeaLevelPressureAvgInches'], austin_thunderstorm['HumidityAvgPercent'], s=None, vmin=0, vmax=100)

 #ax.scatter(x, y, c=color, s=scale, label=color,
 #              alpha=0.3, edgecolors='none')


ax.set(xlim=(26, 34), xticks=np.arange(26, 34),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('Sea Level Pressure Avg Inches', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Sea Level vs Humidity')
fig.set_size_inches(10, 8)

plt.show()
```


    
![png](output_19_0.png)
    


Here we can see that SeaLevelPressure is a negliable factor in between all 5 different events.
While we can rule out Sea level Pressure as a contributing factor, rain seems to occur more often while 
there is higher humidity, so we can look further into that.


```python
#only plotted 2 for ease of analysis for now
#blue is rain, orange is sun 

austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()

# plot
fig, ax = plt.subplots()
#HumidityAvgPercent SeaLevelPressureAvgInches
ax.scatter(austin_rain['DewPointAvgF'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['DewPointAvgF'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_snow['DewPointAvgF'], austin_snow['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_fog['DewPointAvgF'], austin_fog['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_thunderstorm['DewPointAvgF'], austin_thunderstorm['HumidityAvgPercent'], s=None, vmin=0, vmax=100)

 #ax.scatter(x, y, c=color, s=scale, label=color,
 #              alpha=0.3, edgecolors='none')


ax.set(xlim=(0, 100), xticks=np.arange(0, 100),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('DewPointAvgF', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Sea Level vs Humidity')
fig.set_size_inches(10, 8)

plt.show()
```


    
![png](output_21_0.png)
    


Here we can see that DewPoint is a negliable factor in between all 5 different events.
While we can rule out Sea level Pressure as a contributing factor, rain seems to occur more often while 
there is higher humidity, so we can look further into that.


```python
#only plotted 2 for ease of analysis for now
#blue is rain, orange is sun 

austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()

# plot
fig, ax = plt.subplots()
#HumidityAvgPercent SeaLevelPressureAvgInches
ax.scatter(austin_rain['TempAvgF'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['TempAvgF'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_snow['TempAvgF'], austin_snow['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_fog['TempAvgF'], austin_fog['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_thunderstorm['TempAvgF'], austin_thunderstorm['HumidityAvgPercent'], s=None, vmin=0, vmax=100)

 #ax.scatter(x, y, c=color, s=scale, label=color,
 #              alpha=0.3, edgecolors='none')


ax.set(xlim=(0, 100), xticks=np.arange(0, 100),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('TempAvgF', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Sea Level vs Humidity')
fig.set_size_inches(10, 8)

plt.show()
```


    
![png](output_23_0.png)
    


Here we can see that Temperature is a negliable factor in between all 5 different events.
While we can rule out Sea level Pressure as a contributing factor, rain seems to occur more often while 
there is higher humidity, so we can look further into that.


```python
#only plotted 2 for ease of analysis for now
#blue is rain, orange is sun 

austin_rain = austin_weather.loc[austin_weather['Events'].astype(str).str.contains('Rain')].copy()

# plot
fig, ax = plt.subplots()
#HumidityAvgPercent SeaLevelPressureAvgInches
ax.scatter(austin_rain['VisibilityAvgMiles'], austin_rain['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
ax.scatter(austin_sun['VisibilityAvgMiles'], austin_sun['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_snow['TempAvgF'], austin_snow['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_fog['TempAvgF'], austin_fog['HumidityAvgPercent'], s=None, vmin=0, vmax=100)
#ax.scatter(austin_thunderstorm['TempAvgF'], austin_thunderstorm['HumidityAvgPercent'], s=None, vmin=0, vmax=100)

 #ax.scatter(x, y, c=color, s=scale, label=color,
 #              alpha=0.3, edgecolors='none')


ax.set(xlim=(0, 12), xticks=np.arange(0, 12),
       ylim=(20, 100), yticks=np.arange(20, 100))
ax.set_xlabel('VisibilityAvgMiles', fontsize=15)
ax.set_ylabel('Humidity Avg Percent', fontsize=15)
ax.set_title('Sea Level vs Humidity')
fig.set_size_inches(10, 8)

plt.show()
```


    
![png](output_25_0.png)
    


Here we can see that both humidity and visiabliliy are contributing factors towards the event.
Wuth high humidity and low visibiilty we can see an increase in the event rain, and with low humitdity and 
high visibiltiy, we can see a increase in the event sun.


```python
austin_weather.columns
```




    Index(['Date', 'TempHighF', 'TempAvgF', 'TempLowF', 'DewPointHighF',
           'DewPointAvgF', 'DewPointLowF', 'HumidityHighPercent',
           'HumidityAvgPercent', 'HumidityLowPercent',
           'SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches',
           'SeaLevelPressureLowInches', 'VisibilityHighMiles',
           'VisibilityAvgMiles', 'VisibilityLowMiles', 'WindHighMPH', 'WindAvgMPH',
           'WindGustMPH', 'PrecipitationSumInches', 'Events'],
          dtype='object')




```python
train, test = train_test_split(austin_weather, test_size = 0.2, random_state = 42, shuffle=True)
train
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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>SeaLevelPressureAvgInches</th>
      <th>SeaLevelPressureLowInches</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>994</th>
      <td>2016-09-10</td>
      <td>93.0</td>
      <td>82.0</td>
      <td>71.0</td>
      <td>75.0</td>
      <td>71.0</td>
      <td>64.0</td>
      <td>94.0</td>
      <td>68.0</td>
      <td>41.0</td>
      <td>...</td>
      <td>30.06</td>
      <td>29.94</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>18.0</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>0.05</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>981</th>
      <td>2016-08-28</td>
      <td>92.0</td>
      <td>83.0</td>
      <td>73.0</td>
      <td>75.0</td>
      <td>72.0</td>
      <td>70.0</td>
      <td>91.0</td>
      <td>72.0</td>
      <td>52.0</td>
      <td>...</td>
      <td>29.98</td>
      <td>29.90</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>22.0</td>
      <td>0.16</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2013-12-31</td>
      <td>55.0</td>
      <td>46.0</td>
      <td>36.0</td>
      <td>31.0</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>76.0</td>
      <td>54.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>30.39</td>
      <td>30.27</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>147</th>
      <td>2014-05-17</td>
      <td>86.0</td>
      <td>73.0</td>
      <td>60.0</td>
      <td>59.0</td>
      <td>55.0</td>
      <td>47.0</td>
      <td>84.0</td>
      <td>58.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>29.97</td>
      <td>29.90</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>9.0</td>
      <td>26.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>350</th>
      <td>2014-12-06</td>
      <td>69.0</td>
      <td>62.0</td>
      <td>55.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>48.0</td>
      <td>90.0</td>
      <td>72.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>30.32</td>
      <td>30.19</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1109</th>
      <td>2017-01-03</td>
      <td>75.0</td>
      <td>60.0</td>
      <td>44.0</td>
      <td>44.0</td>
      <td>39.0</td>
      <td>31.0</td>
      <td>80.0</td>
      <td>52.0</td>
      <td>24.0</td>
      <td>...</td>
      <td>30.04</td>
      <td>29.95</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1144</th>
      <td>2017-02-07</td>
      <td>86.0</td>
      <td>71.0</td>
      <td>56.0</td>
      <td>66.0</td>
      <td>54.0</td>
      <td>36.0</td>
      <td>100.0</td>
      <td>59.0</td>
      <td>17.0</td>
      <td>...</td>
      <td>29.80</td>
      <td>29.73</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>13.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>0.00</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>2017-07-21</td>
      <td>104.0</td>
      <td>91.0</td>
      <td>77.0</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>61.0</td>
      <td>85.0</td>
      <td>56.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>29.95</td>
      <td>29.86</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>0.01</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>871</th>
      <td>2016-05-10</td>
      <td>91.0</td>
      <td>79.0</td>
      <td>66.0</td>
      <td>74.0</td>
      <td>70.0</td>
      <td>64.0</td>
      <td>93.0</td>
      <td>75.0</td>
      <td>56.0</td>
      <td>...</td>
      <td>29.87</td>
      <td>29.77</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>29.0</td>
      <td>0.27</td>
      <td>Rain , Thunderstorm</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>2017-02-03</td>
      <td>53.0</td>
      <td>49.0</td>
      <td>45.0</td>
      <td>37.0</td>
      <td>34.0</td>
      <td>31.0</td>
      <td>61.0</td>
      <td>57.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>30.37</td>
      <td>30.29</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>21.0</td>
      <td>0.01</td>
      <td>Sun</td>
    </tr>
  </tbody>
</table>
<p>1044 rows × 21 columns</p>
</div>




```python
#austin_weather['Numerical_Events'] = np.where(austin_weather['Events'].astype(str).str.contains('Rain'), '1', '0')
holder = []
for i, j in train.iterrows():
    if 'Sun' in train.at[i, 'Events']:
        holder.append(1)
    elif 'Rain' in train.at[i, 'Events']:
        holder.append(2)
    elif 'Snow' in train.at[i, 'Events']:
        holder.append(3)
    elif 'Fog' in train.at[i, 'Events']:
        holder.append(4)
    elif 'Thunderstorm' in train.at[i, 'Events']:
        holder.append(5)

train['Numerical_Events'] = holder
train

holder3 = []
for i, j in test.iterrows():
    if 'Sun' in test.at[i, 'Events']:
        holder3.append(1)
    elif 'Rain' in test.at[i, 'Events']:
        holder3.append(2)
    elif 'Snow' in test.at[i, 'Events']:
        holder3.append(3)
    elif 'Fog' in test.at[i, 'Events']:
        holder3.append(4)
    elif 'Thunderstorm' in test.at[i, 'Events']:
        holder3.append(5)

test['Numerical_Events'] = holder3
test

```

    C:\Users\ipsam\AppData\Local\Temp/ipykernel_8872/1283157352.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train['Numerical_Events'] = holder
    C:\Users\ipsam\AppData\Local\Temp/ipykernel_8872/1283157352.py:31: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test['Numerical_Events'] = holder3
    




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
      <th>Date</th>
      <th>TempHighF</th>
      <th>TempAvgF</th>
      <th>TempLowF</th>
      <th>DewPointHighF</th>
      <th>DewPointAvgF</th>
      <th>DewPointLowF</th>
      <th>HumidityHighPercent</th>
      <th>HumidityAvgPercent</th>
      <th>HumidityLowPercent</th>
      <th>...</th>
      <th>VisibilityHighMiles</th>
      <th>VisibilityAvgMiles</th>
      <th>VisibilityLowMiles</th>
      <th>WindHighMPH</th>
      <th>WindAvgMPH</th>
      <th>WindGustMPH</th>
      <th>PrecipitationSumInches</th>
      <th>Events</th>
      <th>Numerical_Events</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1186</th>
      <td>2017-03-21</td>
      <td>85.0</td>
      <td>74.0</td>
      <td>63.0</td>
      <td>63.0</td>
      <td>61.0</td>
      <td>59.0</td>
      <td>93.0</td>
      <td>70.0</td>
      <td>47.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>8.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>518</th>
      <td>2015-05-23</td>
      <td>87.0</td>
      <td>77.0</td>
      <td>66.0</td>
      <td>74.0</td>
      <td>71.0</td>
      <td>66.0</td>
      <td>100.0</td>
      <td>83.0</td>
      <td>65.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>32.0</td>
      <td>1.41</td>
      <td>Fog , Rain , Thunderstorm</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>854</th>
      <td>2016-04-23</td>
      <td>83.0</td>
      <td>69.0</td>
      <td>55.0</td>
      <td>59.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>87.0</td>
      <td>61.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>989</th>
      <td>2016-09-05</td>
      <td>94.0</td>
      <td>84.0</td>
      <td>73.0</td>
      <td>77.0</td>
      <td>74.0</td>
      <td>72.0</td>
      <td>94.0</td>
      <td>77.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>5.0</td>
      <td>19.0</td>
      <td>0.01</td>
      <td>Rain , Thunderstorm</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>2017-03-17</td>
      <td>83.0</td>
      <td>74.0</td>
      <td>65.0</td>
      <td>64.0</td>
      <td>62.0</td>
      <td>61.0</td>
      <td>87.0</td>
      <td>69.0</td>
      <td>51.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>24.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2014-05-10</td>
      <td>91.0</td>
      <td>76.0</td>
      <td>60.0</td>
      <td>66.0</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>97.0</td>
      <td>71.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>961</th>
      <td>2016-08-08</td>
      <td>100.0</td>
      <td>89.0</td>
      <td>78.0</td>
      <td>75.0</td>
      <td>72.0</td>
      <td>68.0</td>
      <td>91.0</td>
      <td>65.0</td>
      <td>39.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>419</th>
      <td>2015-02-13</td>
      <td>70.0</td>
      <td>55.0</td>
      <td>39.0</td>
      <td>44.0</td>
      <td>38.0</td>
      <td>32.0</td>
      <td>79.0</td>
      <td>58.0</td>
      <td>37.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2014-07-12</td>
      <td>98.0</td>
      <td>85.0</td>
      <td>71.0</td>
      <td>71.0</td>
      <td>66.0</td>
      <td>56.0</td>
      <td>93.0</td>
      <td>59.0</td>
      <td>25.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.00</td>
      <td>Sun</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>720</th>
      <td>2015-12-11</td>
      <td>79.0</td>
      <td>66.0</td>
      <td>52.0</td>
      <td>65.0</td>
      <td>61.0</td>
      <td>51.0</td>
      <td>100.0</td>
      <td>79.0</td>
      <td>58.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>Fog</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>261 rows × 23 columns</p>
</div>




```python
holder2 = []
for i, j in train.iterrows():
    curr = []
    curr.append(train.at[i, 'HumidityAvgPercent'])
    #curr.append(train.at[i, 'VisibilityAvgMiles'])
    holder2.append(curr)

X = np.array(holder2)
Y = train['Numerical_Events']

clf1 = svm.SVC(kernel='linear', C=1.0)
clf1.fit(X, Y)
#clf1.predict([[79.0, 2.4]])
holder4 = []
for i, j in test.iterrows():
    curr = []
    curr.append(test.at[i, 'HumidityAvgPercent'])
    #curr.append(test.at[i, 'VisibilityAvgMiles'])
    holder4.append(curr)
    
test['result'] = clf1.predict(holder4)
count = 0

for i, j in test.iterrows():
    if test.at[i, 'Numerical_Events'] == test.at[i, 'result']:
        count += 1
        count 
correct = count/261
correct

```

    C:\Users\ipsam\AppData\Local\Temp/ipykernel_8872/116868951.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test['result'] = clf1.predict(holder4)
    




    0.8199233716475096




```python
holder2 = []
for i, j in train.iterrows():
    curr = []
    #curr.append(train.at[i, 'HumidityAvgPercent'])
    curr.append(train.at[i, 'VisibilityAvgMiles'])
    holder2.append(curr)

X = np.array(holder2)
Y = train['Numerical_Events']

clf1 = svm.SVC(kernel='linear', C=1.0)
clf1.fit(X, Y)
#clf1.predict([[79.0, 2.4]])
holder4 = []
for i, j in test.iterrows():
    curr = []
    #curr.append(test.at[i, 'HumidityAvgPercent'])
    curr.append(test.at[i, 'VisibilityAvgMiles'])
    holder4.append(curr)
    
test['result'] = clf1.predict(holder4)
count = []

for i, j in test.iterrows():
    if test.at[i, 'Numerical_Events'] == test.at[i, 'result']:
        count.append(1)
    else:
        count.append(0)

correct = count.count(1)/261

correct
```

    C:\Users\ipsam\AppData\Local\Temp/ipykernel_8872/664821976.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test['result'] = clf1.predict(holder4)
    




    0.8007662835249042




```python
holder5 = []
for i, j in train.iterrows():
    curr = []
    curr.append(train.at[i, 'HumidityAvgPercent'])
    #curr.append(train.at[i, 'VisibilityAvgMiles'])
    holder5.append(curr)

X = np.array(holder5)
Y = train['Numerical_Events']

clf2 = svm.SVC(kernel='linear', C=1.0)
clf2.fit(X, Y)
#clf1.predict([[79.0, 2.4]])
holder6 = []
for i, j in test.iterrows():
    curr = []
    curr.append(test.at[i, 'HumidityAvgPercent'])
    #curr.append(test.at[i, 'VisibilityAvgMiles'])
    holder6.append(curr)
    
test['result'] = clf2.predict(holder4)
count = []

for i, j in test.iterrows():
    if test.at[i, 'Numerical_Events'] == test.at[i, 'result']:
        count.append(1)
    else:
        count.append(0)

correct = count.count(1)/261

correct
```

    C:\Users\ipsam\AppData\Local\Temp/ipykernel_8872/3043253486.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test['result'] = clf2.predict(holder4)
    




    0.685823754789272




```python
clf1score = []
clf2score = []


clf1score.append(clf1.score(X, Y))
clf2score.append(clf2.score(X, Y))

print(clf1score)
print(clf2score)

#Compute the difference between the results
diff = [y - x for y, x in zip(clf1score, clf2score)]
#Comopute the mean of differences
d_bar = np.mean(diff)
#compute the variance of differences
sigma2 = np.var(diff)
#compute the number of data points used for training 
n1 = 1044
#compute the number of data points used for testing 
n2 = 261
#compute the total number of data points
n = 1035
#compute the modified variance
sigma2_mod = sigma2 * (1/n + n2/n1)
#compute the t_static
t_statistic =  d_bar / np.sqrt(sigma2_mod)
from scipy.stats import t
#Compute p-value and plot the results 
Pvalue = 1 - t.cdf(t_statistic, n-1)
Pvalue

```

    [0.6829501915708812]
    [0.7854406130268199]
    

    C:\Users\ipsam\AppData\Local\Temp/ipykernel_8872/910554783.py:26: RuntimeWarning: divide by zero encountered in double_scalars
      t_statistic =  d_bar / np.sqrt(sigma2_mod)
    




    1.0


