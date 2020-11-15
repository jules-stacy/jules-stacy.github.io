```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import altair as alt
import seaborn as sns
```


```python
df = pd.read_csv("./Data/touchpoints.csv", skipinitialspace=True)
df['id'] = df.index
df
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
      <th>TouchpointsPhone</th>
      <th>TouchpointsChat</th>
      <th>TouchpointsEmail</th>
      <th>TouchpointsTotal</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-1</td>
      <td>0.43</td>
      <td>0.13</td>
      <td>0.55</td>
      <td>1.11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-2</td>
      <td>0.35</td>
      <td>0.11</td>
      <td>0.38</td>
      <td>0.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-3</td>
      <td>0.34</td>
      <td>0.11</td>
      <td>0.38</td>
      <td>0.83</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-4</td>
      <td>0.24</td>
      <td>0.18</td>
      <td>0.30</td>
      <td>0.72</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-5</td>
      <td>0.23</td>
      <td>0.17</td>
      <td>0.28</td>
      <td>0.68</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-6</td>
      <td>0.29</td>
      <td>0.17</td>
      <td>0.29</td>
      <td>0.75</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-7</td>
      <td>0.28</td>
      <td>0.14</td>
      <td>0.30</td>
      <td>0.72</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-8</td>
      <td>0.30</td>
      <td>0.14</td>
      <td>0.35</td>
      <td>0.79</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-9</td>
      <td>0.30</td>
      <td>0.11</td>
      <td>0.32</td>
      <td>0.73</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-10</td>
      <td>0.29</td>
      <td>0.12</td>
      <td>0.31</td>
      <td>0.72</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-11</td>
      <td>0.29</td>
      <td>0.11</td>
      <td>0.41</td>
      <td>0.81</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018-12</td>
      <td>0.38</td>
      <td>0.15</td>
      <td>0.54</td>
      <td>1.07</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019-01</td>
      <td>0.45</td>
      <td>0.16</td>
      <td>0.58</td>
      <td>1.19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019-02</td>
      <td>0.32</td>
      <td>0.13</td>
      <td>0.38</td>
      <td>0.83</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019-03</td>
      <td>0.31</td>
      <td>0.13</td>
      <td>0.38</td>
      <td>0.82</td>
      <td>14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019-04</td>
      <td>0.25</td>
      <td>0.15</td>
      <td>0.34</td>
      <td>0.74</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019-05</td>
      <td>0.26</td>
      <td>0.20</td>
      <td>0.34</td>
      <td>0.80</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019-06</td>
      <td>0.26</td>
      <td>0.20</td>
      <td>0.34</td>
      <td>0.80</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019-07</td>
      <td>0.25</td>
      <td>0.15</td>
      <td>0.34</td>
      <td>0.74</td>
      <td>18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019-08</td>
      <td>0.28</td>
      <td>0.16</td>
      <td>0.33</td>
      <td>0.77</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2019-09</td>
      <td>0.24</td>
      <td>0.17</td>
      <td>0.32</td>
      <td>0.73</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2019-10</td>
      <td>0.24</td>
      <td>0.17</td>
      <td>0.34</td>
      <td>0.75</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2019-11</td>
      <td>0.25</td>
      <td>0.18</td>
      <td>0.36</td>
      <td>0.79</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2019-12</td>
      <td>0.33</td>
      <td>0.18</td>
      <td>0.53</td>
      <td>1.04</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2020-01</td>
      <td>0.34</td>
      <td>0.26</td>
      <td>0.50</td>
      <td>1.10</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
l = pd.wide_to_long(df, stubnames='Touchpoints', i=['id', 'Date'], j='Source', suffix='\D+')
```


```python
l
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
      <th></th>
      <th></th>
      <th>Touchpoints</th>
    </tr>
    <tr>
      <th>id</th>
      <th>Date</th>
      <th>Source</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">0</th>
      <th rowspan="4" valign="top">2018-1</th>
      <th>Phone</th>
      <td>0.43</td>
    </tr>
    <tr>
      <th>Chat</th>
      <td>0.13</td>
    </tr>
    <tr>
      <th>Email</th>
      <td>0.55</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>1.11</td>
    </tr>
    <tr>
      <th>1</th>
      <th>2018-2</th>
      <th>Phone</th>
      <td>0.35</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>23</th>
      <th>2019-12</th>
      <th>Total</th>
      <td>1.04</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">24</th>
      <th rowspan="4" valign="top">2020-01</th>
      <th>Phone</th>
      <td>0.34</td>
    </tr>
    <tr>
      <th>Chat</th>
      <td>0.26</td>
    </tr>
    <tr>
      <th>Email</th>
      <td>0.50</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>1.10</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>




```python
#adjust preliminary graph settings: theme and plot size
sns.set_theme(style="whitegrid", rc={"xtick.bottom" : True, "ytick.left" : True})
plt.figure(figsize=(10,10))

#create the base plot and set the labels
plot =  sns.lineplot(data=l, x='Date', y='Touchpoints', palette="tab10", hue='Source', linewidth=1.5)
plot.set(xlabel="Date", ylabel='Touchpoints Per Customer')


#adjust major and minor tick marks on the x axis
plot.xaxis.set_major_locator(MultipleLocator(3))
plot.tick_params(axis='x', which='minor', direction='out')
plot.xaxis.set_minor_locator(MultipleLocator(1))

#put the legend above the plot
plot.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4,
            borderaxespad=0, frameon=False, fontsize=14)

#put some annotations on the graph
plot.annotate('Total touchpoints \nhave increased \nslightly to ~500K \n(+3.8% y/y) from \n2018 to 2019', 
            xy=(3, 1),  xycoords='data', fontsize=14,
            xytext=(1.05, 0.9), textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', color='grey'
            )
plot.annotate('Total touchpoints \nper customer \nexhibits a \nseasonal pattern \nwith consistent \nhighs and lows', 
            xy=(3, 1),  xycoords='data', fontsize=14,
            xytext=(1.05, 0.7), textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', color='grey'
            )

# Now re do the total curve, but biger with distinct color
plt.plot(df['Date'], df['TouchpointsTotal'], marker='', color='red', linewidth=10, alpha=0.7)

#then add some lines to the plot so we can further annotate
plt.plot([0.3, 11.7], [1.3, 1.3], 'k-', lw=2, color='grey')
plot.annotate('478,123 touchpoints', 
            xy=(3, 1),  xycoords='data', fontsize=14,
            xytext=(.15, .985), textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', color='grey'
            )
plt.plot([12.3, 23.7], [1.3, 1.3], 'k-', lw=2, color='grey')
plot.annotate('496,234 touchpoints', 
            xy=(3, 1),  xycoords='data', fontsize=14,
            xytext=(.65, .985), textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', color='grey'
            )

```




    Text(0.65, 0.985, '496,234 touchpoints')




    
![png](output_4_1.png)
    

