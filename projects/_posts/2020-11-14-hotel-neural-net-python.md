---
title: "Neural Network Regression on GPU"
categories:
  - projects
tags:
  - projects
  - hotel
  - python
  - neural network 
  - regression
  - sklearn
  - pipeline
  - grid search
  - tensorflow
  - keras
---

A neural network pipeline was developed to predict the number of nights a guest would stay at the hotel. Coded in Python.  

This project makes use of TensorFlow-GPU to build a neural network. Hyperparameters are then optimized for the network using GridSearchCV. Finally, the trained neural network is used to regress on the number of nights a given guest is expected to stay. Many thanks to Jeff Heaton from the Washington University in St. Louis. If neural networks interest you and you want to learn more, [check out his Youtube page.](https://www.youtube.com/channel/UCR1-GEpyOPzT2AO4D_eifdw)  

### Imports

```python
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import copy
import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score
from sklearn.metrics import mean_squared_error as msescore
from sklearn.metrics import roc_auc_score as rocauc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer, Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

#from scikitplot.metrics import plot_confusion_matrix # this package may not be included in base Anaconda
import seaborn as sns

#filter warnings: https://www.kite.com/python/answers/how-to-suppress-warnings-in-python
import warnings
warnings.filterwarnings("ignore")
```


```python
# Neural Network Imports
# from tensorflow import keras
# from tensorflow.keras import layers
import tensorflow as tf
tf.random.set_seed(12345)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras import models, layers, backend as K
from sklearn.metrics import plot_roc_curve
from tensorflow.python.client import device_lib

tf.keras.backend.clear_session()
tf.debugging.set_log_device_placement(False)
```

Above are the imports for this project. Many of the imports are not used in this notebook, but are included here because they were used in the submission version of the project which included additional work.  


```python
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU Device: ", tf.config.list_physical_devices('XLA_GPU'))
```

    Tensor Flow Version: 2.3.1
    Keras Version: 2.4.0
    
    Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
    Pandas 1.1.3
    Scikit-Learn 0.23.2
    GPU Device:  [PhysicalDevice(name='/physical_device:XLA_GPU:0', device_type='XLA_GPU')]
    

Above is a printout of the TensorFlow version and a confirmation that the package detects my video card. I am using an MSI GF65 Thin with an NVIDIA GeForce GTX 1660Ti. It took me more time than expected to get this to work on my machine, and if you are planning to create your own neural network and run it on your GPU, my advice is to pay attention to the version numbers of the CUDA Toolkit and cuDNN SDK that are required in order to run TensorFlow-GPU on your NVIDIA graphics card.  


```python
#variable initialization
path=r"../Data/hotel_bookings.csv"
y_column = []
```

### Helper Functions

Below I have defined some helper functions which are called later in the notebook. This reduces clutter and improves consistency by referencing all the necessary code which is kept in one place.  

```python
# Function to prep regression dataset

# #encode as function for later cell
def prep_data_regress(path):
    global df
    global df_enc
    global enc_var_list
    global y_column
    global int_var_list
    
    #read csv
    dat_in = pd.read_csv(path)
        
    #replace missing values in certain columns
    #source: https://datatofish.com/replace-nan-values-with-zeros/
    dat_in['children'] = dat_in['children'].fillna(0)
    dat_in['country'] = dat_in['country'].fillna("unknown")
    dat_in['agent'] = dat_in['agent'].fillna(0)
    dat_in['company'] = dat_in['company'].fillna(0)
    # Drop outlier
    dat_in.drop(dat_in[ dat_in['adr'] == 5400 ].index , inplace=True)
    #convert reservation_status_date to day, month, year
    #source: https://stackoverflow.com/questions/25789445/pandas-make-new-column-from-string-slice-of-another-column
    dat_in['reservation_status_year'] = dat_in.reservation_status_date.str[:4]
    dat_in['reservation_status_month'] = dat_in.reservation_status_date.str[5:7]
    dat_in['reservation_status_day'] = dat_in.reservation_status_date.str[8:10]
    dat_in.drop(['reservation_status_date'], axis=1)
    #convert categoricals to proper data type
    dat_in = dat_in.astype({"agent":'category', "company":'category', "is_canceled":'category', "hotel":'category', "is_repeated_guest":'category', "reserved_room_type":'category', "assigned_room_type":'category', "deposit_type":'category', "customer_type":'category', "country":'category',  "arrival_date_month":'category', "meal":'category',  "market_segment":'category', 'reservation_status_year':'category', "distribution_channel":'category', 'reservation_status_month':'category', 'reservation_status_day':'category'})
    #create total stay length df
    dat_in ['total_stay_length'] = dat_in ['stays_in_weekend_nights']+ dat_in ['stays_in_week_nights']
    #set the y column to its own dataframe and remove dependent variables
    y_column = dat_in['total_stay_length']
    del dat_in['is_canceled']
    del dat_in['reservation_status_date']
    del dat_in['reservation_status']
    del dat_in['total_stay_length']
    del dat_in['stays_in_weekend_nights']
    del dat_in['stays_in_week_nights']    
    df = dat_in
    #split dataframes into categorical and integer
    #source: https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type
    df_cat = dat_in.select_dtypes(include=['category'])
    df_int = dat_in.select_dtypes(exclude=['category'])
    #list factors by dtype
    cat_var_list = df_cat.columns.tolist()
    int_var_list = df_int.columns.tolist()
    #dummy encode: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
    cat_enc = pd.get_dummies(df_cat, drop_first=True)
    hot_var_list = cat_enc.columns.tolist()
    #merge encoded and integer dataframes
    df_enc = cat_enc.merge(df_int, left_index=True, right_index=True)
    enc_var_list = df_enc.columns.tolist()
    print("objects created: df_cat, df_int, cat_var_list, int_var_list, cat_enc, hot_var_list, df_enc, enc_var_list")
    
```

Above is a helper function which cleans the dataset properly for regression. This makes it easy to clean the dataset as needed, for instance when the dataset has been previously modified and a freshly cleaned dataset is needed.  


```python
# Function to split the data

def data_split(df_enc, strat=y_column):
    global X_train
    global X_test
    global y_train
    global y_test
    
    # set the x column to its own dataframes/series
    x_columns = df_enc

    # train-test split with stratification on the class we are trying to predict
    X_train, X_test, y_train, y_test = train_test_split( 
                                        x_columns,          # x column values
                                        y_column,           # column to predict
                                        test_size=0.2,      # 80/20 split
                                        random_state=12345, # random state for repeatability
                                        stratify=strat) # stratification to preserve class imbalance 
    print('objects created: X_train, X_test, y_train, y_test')

#Function to perform SMOTE on data    
#SMOTE implementation
#http://glemaitre.github.io/imbalanced-learn/generated/imblearn.under_sampling.RandomUnderSampler.html
#perform random under-sampling and SMOTE on training dataset    
def smote_data():
    global X_train
    global y_train
    global X_sm
    global y_sm

    sm=SMOTE(random_state=12345)
    X_sm, y_sm = sm.fit_sample(X_train, y_train)
    print('objects created: X_sm, y_sm (remember, these are training sets)')
```

Above is a helper function to split the dataset into train and test datasets. Data was split 80/20 into train/test datasets and a random state was specified for consistency within the project. The option for data stratification was preserved through the strat argument within data_split.  


Additionally there is a function to perform random over-sampling on the dataset, which is not used in this notebook.  

```python
def rmse(y_test, y_test_pred):
    rsqmetric=100*r2_score(y_test, y_test_pred)
    rmsemetric=np.sqrt(msescore(y_test, y_test_pred))
    print(f'R2: {rsqmetric}%')
    print(f'RMSE: {rmsemetric}%')
    return rsqmetric, rmsemetric
```

This function is designed to return the root mean square error and R squared metrics. These metrics provide measurements of performance for the neural network.  

### Data Preparation

```python
#prep and split the regression data
prep_data_regress(path)
data_split(df_enc, strat=None)
```

    objects created: df_cat, df_int, cat_var_list, int_var_list, cat_enc, hot_var_list, df_enc, enc_var_list
    objects created: X_train, X_test, y_train, y_test
    


Defining the helper functions above automates the data preparation step, making it very compact. Scaling the data as a preprocessing step is not present here but is present within the pipeline below.  

## Modeling
#### Building the Model

```python
# # #https://keras.io/guides/sequential_model/
# https://www.marktechpost.com/2019/06/17/regression-with-keras-deep-learning-with-keras-part-3/
# adabound optimizer https://reposhub.com/CyberZHG-keras-adabound-python-deep-learning.html
def model(nodes1=512, nodes2=256, nodes3=32, optimizer='Adam'):
    model = models.Sequential()
    model.add(layers.Dense(nodes1, activation='relu', input_shape=[X_train.shape[1]]))
    model.add(layers.Dense(nodes2, activation='relu'))
    model.add(layers.Dense(nodes3, activation='relu'))

    # output layer
    model.add(layers.Dense(1))

    # compile model, specify optimizer and loss fn
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model
```

Above, the skeleton of the neural network is created. The model cannot be used for prediction without first fitting to the training dataset.  

A sequential model is defined and 4 layers are added. There is technically a fifth layer, the "input" layer, which reads the data into the model. This is specified by the input_shape parameter, which tells node1 to expect data in the same shape as the data being fed into the model.  

The hidden layers of the model, node1 node2 and node3, are all rectified linear unit (reLu)-activated layers. This activation outputs the maximum of either the transformed input value or zero. This allows the node to behave linearly while being non-linear.  

The last layer added to the model is the output layer, composed of a single node. This node outputs the predicted number of nights each guest would stay at a hotel. Finally, the model is compiled.  

#### Building the Pipeline

```python
#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

column_transformer = ColumnTransformer([
    ("scaler", StandardScaler(with_mean=False), int_var_list) # adjusts data to the same scale
], remainder="passthrough")

keras_pipeline = Pipeline([
    ('datafeed', column_transformer),              # grabs finalized datasets
    ('selector', SelectKBest(f_classif, k='all')), # variable selection procedure
    ('kr', KerasRegressor(build_fn=model, epochs=150, batch_size=32, verbose=0))           # deep learn regression using Keras
])

```

The pipeline is defined above. The pipeline scales the integer data using Scikit-Learn's StandardScaler function. It then passes the data through a variable selection procedure (this is set to k=all and is required to be so by the neural network), and then calls the model using the KerasRegressor method.  

Within KerasRegressor, the epochs parameter is specified to be 150; this is the number of times the neural network uses the entire training set to update the weights within the model. Batch size is the number of observations used by the model before updating the model weights. Verbose is set to zero to suppress outputs.  

### Hyperparameter Tuning Using Grid Search

```python
# https://www.kaggle.com/med92amine/keras-hyperparameter-tuning
# https://medium.com/@am.benatmane/keras-hyperparameter-tuning-using-sklearn-pipelines-grid-search-with-cross-validation-ccfc74b0ce9f
# define the grid search parameters
param_grid = {
#     'kr__nodes1': [512, 256],
#     'kr__nodes2': [256, 128],
#     'kr__nodes3': [ 16, 32, 64],    
   'kr__batch_size':[16, 32, 64],
   'kr__optimizer':['rmsprop', 'Adam', 'sgd'],
}
```

Now that the pipeline is defined, the optimal values for the hyperparameters can be found by performing a grid search. This is an iterative method of tuning hyperparameters. The hyperparameters specified are nodes1, nodes2, and nodes3. In prior grid searches, the optimizer and batch size were tested, and it was found that the Adam optimizer and a batch size of 32 allowed the model to perform best. These values were run separately from each other because of time constraints; if all the parameters were to be tested together, the estimated time for completion would be approximately 3.5 days on this hardware.  

```python
# perform grid search with multi-metric evaluation
scoring = {'RMSE':'neg_root_mean_squared_error', 'R2':'r2', 'Variance':'explained_variance'}
grid = GridSearchCV(keras_pipeline, param_grid, verbose = 3, scoring=scoring, refit='RMSE') 
#  
# fitting the model for grid search 
grid.fit(X_train, y_train)
    
print(grid.best_estimator_)
```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=16 ...................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV]  kr__nodes1=512, kr__nodes2=256, kr__nodes3=16, R2=0.351, RMSE=-2.117, Variance=0.372, total= 9.0min
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=16 ...................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  9.0min remaining:    0.0s
    

    [CV]  kr__nodes1=512, kr__nodes2=256, kr__nodes3=16, R2=0.125, RMSE=-2.384, Variance=0.126, total= 9.0min
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=16 ...................
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed: 17.9min remaining:    0.0s
    

    [CV]  kr__nodes1=512, kr__nodes2=256, kr__nodes3=16, R2=0.395, RMSE=-2.016, Variance=0.401, total= 8.8min
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=16 ...................
    [CV]  kr__nodes1=512, kr__nodes2=256, kr__nodes3=16, R2=-0.000, RMSE=-2.554, Variance=0.000, total= 9.4min
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=16 ...................
    [CV]  kr__nodes1=512, kr__nodes2=256, kr__nodes3=16, R2=0.347, RMSE=-2.058, Variance=0.350, total= 9.7min
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=32 ...................
    [CV]  kr__nodes1=512, kr__nodes2=256, kr__nodes3=32, R2=0.286, RMSE=-2.219, Variance=0.334, total= 9.0min
    [CV] kr__nodes1=512, kr__nodes2=256, kr__nodes3=32 ...................

# ...    

    [Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed: 528.8min finished
    

    Pipeline(steps=[('datafeed',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('scaler',
                                                      StandardScaler(with_mean=False),
                                                      ['lead_time',
                                                       'arrival_date_year',
                                                       'arrival_date_week_number',
                                                       'arrival_date_day_of_month',
                                                       'adults', 'children',
                                                       'babies',
                                                       'previous_cancellations',
                                                       'previous_bookings_not_canceled',
                                                       'booking_changes',
                                                       'days_in_waiting_list',
                                                       'adr',
                                                       'required_car_parking_spaces',
                                                       'total_of_special_requests'])])),
                    ('selector', SelectKBest(k='all')),
                    ('kr',
                     <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001BB1126D160>)])
    

Above is a small portion of the grid search output. Because verbose is set to 3, a significant amount of information is displayed about each cross-validation step. The optimal model is defined through multi-metric scoring, using the r-squared, root mean square error, and explained variance methods within Scikit-Learn.  


```python
cv_results = pd.DataFrame(grid.cv_results_)
cv_results.iloc[grid.best_index_]
```




    mean_fit_time                                                      519.94
    std_fit_time                                                      12.8087
    mean_score_time                                                  0.738918
    std_score_time                                                  0.0981172
    param_kr__nodes1                                                      512
    param_kr__nodes2                                                      256
    param_kr__nodes3                                                       64
    params                  {'kr__nodes1': 512, 'kr__nodes2': 256, 'kr__no...
    split0_test_RMSE                                                 -2.05955
    split1_test_RMSE                                                 -2.04441
    split2_test_RMSE                                                 -2.04446
    split3_test_RMSE                                                 -2.11146
    split4_test_RMSE                                                 -2.04865
    mean_test_RMSE                                                   -2.06171
    std_test_RMSE                                                   0.0254857
    rank_test_RMSE                                                          1
    split0_test_R2                                                   0.385619
    split1_test_R2                                                   0.356677
    split2_test_R2                                                   0.377971
    split3_test_R2                                                   0.316118
    split4_test_R2                                                   0.352148
    mean_test_R2                                                     0.357707
    std_test_R2                                                     0.0242952
    rank_test_R2                                                            1
    split0_test_Variance                                             0.386977
    split1_test_Variance                                             0.365946
    split2_test_Variance                                             0.382962
    split3_test_Variance                                             0.322105
    split4_test_Variance                                              0.35448
    mean_test_Variance                                               0.362494
    std_test_Variance                                               0.0233485
    rank_test_Variance                                                      3
    Name: 2, dtype: object


Finally, the results of the grid search are displayed above. The optimal number of nodes for each layer are shown to be 512, 256, and 64 for layers 1, 2, and 3 respectively. The scoring metrics for each fold as well as the mean scoring metrics for all five folds are also shown.  

### Fitting and Scoring the Model

```python
keras_pipeline.fit(X_train, y_train)
```




    Pipeline(steps=[('datafeed',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('scaler',
                                                      StandardScaler(with_mean=False),
                                                      ['lead_time',
                                                       'arrival_date_year',
                                                       'arrival_date_week_number',
                                                       'arrival_date_day_of_month',
                                                       'adults', 'children',
                                                       'babies',
                                                       'previous_cancellations',
                                                       'previous_bookings_not_canceled',
                                                       'booking_changes',
                                                       'days_in_waiting_list',
                                                       'adr',
                                                       'required_car_parking_spaces',
                                                       'total_of_special_requests'])])),
                    ('selector', SelectKBest(k='all')),
                    ('kr',
                     <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001BA9ADE09A0>)])




```python
y_test_pred = keras_pipeline.predict(X_test)
```

Above the model is fitted, and then the model predicts total_stay_length values for the test set. Prior to fitting, hyperparameter values from the grid search are updated and the model is re-compiled.  

```python
nn_rsq, nn_rmse = rmse(y_test, y_test_pred)
```

    R2: 37.16639575352713%
    RMSE: 1.9747863240567953%
    
Finally, the rmse helper function defined at the beginning of the notebook is called in order to score the fitted model. This function uses the predicted and actual total_stay_length values for the test dataset, which saves time by not calling the modeling pipeline.  

The predicted values result in an r-squared value of 37%. Higher r-squared values represent smaller differences between observed and predicted values. The root mean square error of 1.97 indicates that the model is able to predict the total stay length of a guest to within two days.  

This model is not highly accurate but does have its uses. Given that one- and two-week stay lengths were common and that the maximum stay length was in excess of 60 days, the fact that the model is able to predict total stay length to within two days is decent. In this case we can probably conclude that the model is not over-fitted. The model can be improved by providing more training data, and potentially by adding or taking away hidden layers. Additional fine-tuning can also be performed by changing the model to a dense model or by adding dropout to the model.  

Because reservation length is a variable that was passed into the model, we can conclude that that variable is not always useful for predicting how long a guest would stay at the hotel. One of the factors that could influence this inaccuracy is that cancellations (or stays of length zero) were not removed from the dataset.