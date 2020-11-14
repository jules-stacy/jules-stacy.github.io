---
title: "Logistic Pipeline, SMOTE, and Grid Search"
categories:
  - projects
tags:
  - projects
  - hotel
  - python
  - logistic 
  - classification
  - imblearn
  - sklearn
  - pipeline
  - grid search
---

Logistic pipelines were developed to predict whether a guest would cancel their hotel reservation. Coded in Python.

This project makes use of the scikit-learn (sklearn) and imbalanced-learn (imblearn) packages.


#### Business Understanding
The business understanding for this model is that it would be useful in allowing hotels to predict their cancellations ahead of time. This capability would allow hotels to plan accordingly with regards to booking, room service scheduling, profitability forecasting, and an overall goal of minimizing the number of people that cancel their hotel rooms. Ideally this would allow the hotels in question to be able to better capture their potential streams of revenue by minimizing factors that influence cancellations, maximizing factors that influence completed stays, and planning contingencies against guests that are forecasted to cancel their reservations.

### Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score as rocauc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
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
#class that mutes code output
#source: https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
import sys, traceback
class Suppressor(object):
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
    # Do normal exception handling
            def write(self, x): pass
```

The Suppresser class works by wrapping code in a function call which prevents the code from printing. In particular this class is useful for the confusion matrices and the grid search portions of this project, since they undesirably print lots of warnings.

## Data Understanding

Additional information about the data is available here: https://www.sciencedirect.com/science/article/pii/S2352340918315191

The data was collected from two hotels located in Portugal. It features 32 variables which include two columns detailing whether guests cancelled their reservation. The variables are a wide variety of observations about guests: whether they booked a meal, planned arrival and check-out dates, the company or agent they used when booking the hotel, the market segment and distribution channel, the room type, the number of guests and how many children they have, and many others. The data is of high-quality with relatively few missing values and a single outlier, and is particularly useful for modeling with regards to a number of predictor variables such as what type of room a guest will reserve, or whether a guest is a transient or contract guest; the point being that the data could serve a number of needs by the hotels being studied.

The data was collected for bookings due to arrive between July 1, 2015 and August 31, 2017, and is comprised of hotel real data. As such, all identifying information about the hotels and guests is anonymized. Because the data was collected from two hotels located in Portugal, observations from modeling will only be applicable to these two hotels. Because the study is not experimental in nature, no causation can be drawn from the results; only correlation can be observed. In addressing the randomness of the sampling data, it is unknown as to whether the hotels were randomly chosen. However, the nature of the collection of the data means that hotel guests were effectively random as it could not be known to an affective scale which guests stayed at which hotels and for how long prior to data collection. In other words, the data is vast enough that any foreknowledge of individual guest stays by researchers would be nullified by the remaining unknown guests to such a degree that known guests could not have an influence on the dataset.

## Data Preparation
### Import data and drop outliers


```python
df = pd.read_csv("../Data/hotel_bookings.csv")

#replace missing values in certain columns
#source: https://datatofish.com/replace-nan-values-with-zeros/
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna("unknown")
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# Drop outlier
df.drop(df[ df['adr'] == 5400 ].index , inplace=True)
df.drop(['reservation_status_date'], axis=1)

#convert reservation_status_date to day, month, year
#source: https://stackoverflow.com/questions/25789445/pandas-make-new-column-from-string-slice-of-another-column
df['reservation_status_year'] = df.reservation_status_date.str[:4]
df['reservation_status_month'] = df.reservation_status_date.str[5:7]
df['reservation_status_day'] = df.reservation_status_date.str[8:10]

#convert categoricals to proper data type
df = df.astype({"agent":'category', "company":'category', "is_canceled":'category', 
                "hotel":'category', "is_repeated_guest":'category',
                "reserved_room_type":'category', "assigned_room_type":'category',
                "deposit_type":'category', "customer_type":'category',
                "country":'category', 
                "arrival_date_month":'category', "meal":'category', 
                "market_segment":'category', 'reservation_status_year':'category',
                "distribution_channel":'category', 'reservation_status_month':'category',
                'reservation_status_day':'category'
               })

#set the y column to its own dataframe
y_column = df['is_canceled']
del df['is_canceled']
del df['reservation_status_date']
del df['reservation_status']
```

#### Data Preparation
Data is cleaned as a pre-processing step. Missing values are filled with 0 or 'unknown' as appropriate 

* Missing values in the 'children' column are replaced with 0s since that is the most common.
* Missing values in country are replaced with a new "unknown" value since other imputation methods are not effective here.
* Missing values in the 'agent' column are replaced with 0 as the default agent number.
* Missing values in the 'company' column are replaced with 0 as the default company number.

An outlier for average daily rate (ADR) of 5400 is dropped, since the normal ADR usually stays under 300. The column 'reservation_status_date' is broken up into year, month, and day values to be more usable. The target variable, 'is_canceled', is defined then dropped from the full dataset. The 'reservation_status' column is a duplicate of the target column, so it is also dropped from the dataset as it would cause confound the models to be 100% accurate. Hotels would also not have this data prior to determining a cancellation.

##### One-hot encoding


```python
#split dataframes into categorical and integer
#source: https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type
df_cat = df.select_dtypes(include=['category'])
df_int = df.select_dtypes(exclude=['category'])

#list factors by dtype
cat_var_list = df_cat.columns.tolist()
int_var_list = df_int.columns.tolist()

#dummy encode: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
cat_enc = pd.get_dummies(df_cat, drop_first=True)
hot_var_list = cat_enc.columns.tolist()

#merge encoded and integer dataframes
df_enc = cat_enc.merge(df_int, left_index=True, right_index=True)
enc_var_list = df_enc.columns.tolist()
```

One-hot encoding is performed on the dataset prior to splitting it into test and train datasets. One-hot encoding will split each of the categorical columns into boolean columns for each category within the original column. This preprocessing step avoids the troubles of having to one-hot encode raw training and test sets, random under-sampled training and test sets, and SMOTE training and test sets.

Attempts were made to include this method within the pipeline, but including this method before splitting data into training and test datasets addresses some key issues. All factor levels are encoded which prevents y_test from containing factor levels not present in X_train. And stemming from this first issue, including one hot encoding within the pipeline requires the inclusion of handle_unknown="ignore" which prevents the use of drop='first'. 

In summary, performing one-hot encoding as a preprocessing step allows the entire dataset to become encoded while preventing column duplication and confounding.

## Data Division and Response Balancing
#### Train and Test Splitting


```python
# set the x column to its own dataframes/series
x_columns = df_enc

# train-test split with stratification on the class we are trying to predict
X_train, X_test, y_train, y_test = train_test_split( 
                                    x_columns,          # x column values
                                    y_column,           # column to predict
                                    test_size=0.2,      # 80/20 split
                                    random_state=12345, # random state for repeatability
                                    stratify=y_column) # stratification to preserve class imbalance 


#simple_reservation_status broken out by is_canceled
y_t = pd.DataFrame(y_train)
Bar_chart=sns.countplot(data=y_t, x='is_canceled', hue="is_canceled", palette=['#432371',"#FAAE7B"])
Bar_chart.set_xticklabels(Bar_chart.get_xticklabels())
with Suppressor():
    plt.title("Response Variable Ratio, Raw")
```


    
![png](\assets\images\logistic-svm-smote\output_15_0.png)
    


Data is split into training and test datasets using train_test_split(). The distributions of response cases is graphed above to show how the dataset contains a ratio of about 2:1 for non-cancellations to cancellations. Imbalanced data (when you have one category more represented than the other) can cause problems with model performance when bulding a classification model. This imbalance in the dataset will be addressed through both random under-sampling and over-sampling using SMOTE.

#### Random Under-sampling


```python
#http://glemaitre.github.io/imbalanced-learn/generated/imblearn.under_sampling.RandomUnderSampler.html
#perform random under-sampling and SMOTE on training dataset
rus=RandomUnderSampler(random_state=12345)
X_rus, y_rus = rus.fit_sample(X_train, y_train)


#simple_reservation_status broken out by is_canceled
y_t = pd.DataFrame(y_rus)
Bar_chart=sns.countplot(data=y_t, x='is_canceled', hue="is_canceled", palette=['#432371',"#FAAE7B"])
Bar_chart.set_xticklabels(Bar_chart.get_xticklabels())
with Suppressor():
    plt.title("Response Variable Ratio, Random Under-Sampling")
```


    
![png](\assets\images\logistic-svm-smote\output_18_0.png)
    


Random under-sampling was performed to generate a balanced dataset with regard to the 'is_canceled' class we are tring to predict. This adjusts the ratio of non-cancellations to cancellations to 1:1, and adjusted the total number of responses to 70,000 from the original 91,000. Random under-sampling is understood to be an inferior method to over-sampling since it drops information from the dataset in order to balance the responses. Since our dataset is so large, this method should work fine.

#### SMOTE


```python
#http://glemaitre.github.io/imbalanced-learn/generated/imblearn.under_sampling.RandomUnderSampler.html
#perform random under-sampling and SMOTE on training dataset
sm=SMOTE(random_state=12345)
X_sm, y_sm = sm.fit_sample(X_train, y_train)

#simple_reservation_status broken out by is_canceled
y_t = pd.DataFrame(y_sm)
Bar_chart=sns.countplot(data=y_t, x='is_canceled', hue="is_canceled", palette=['#432371',"#FAAE7B"])
Bar_chart.set_xticklabels(Bar_chart.get_xticklabels())
with Suppressor():
    plt.title("Response Variable Ratio, SMOTE")
    
```


    
![png](\assets\images\logistic-svm-smote\output_21_0.png)
    


SMOTE was also run on the dataset, which resulted in a 1:1 ratio and a total training set size of 120,000. SMOTE is an SVM-based over-sampling method which generates observations by selecting existing observations with the same response and drawing a new observation somewhere on a line between those two points. In this way approximately 25,000 fake cancellation observations were generated for the training set.

## Modeling
### Logistic Regression Pipeline


```python
#sklearn pipeline source: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
#tutorial referenced for data preprocessing: https://www.kdnuggets.com/2020/06/simplifying-mixed-feature-type-preprocessing-scikit-learn-pipelines.html
#tutorial referenced for column transformer: https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260
#tutorial referenced for standard scaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#tutorial referenced for pipeline stuff: https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156
#more pipeline references: https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
#even more pipeline references: https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
#onehot encoder unknown categories error: https://www.roelpeters.be/found-unknown-categories-in-column-sklearn/


# scale numeric columns and perform the logistic regression

column_transformer = ColumnTransformer([
    ("scaler", StandardScaler(), int_var_list) # adjusts data to the same scale
], remainder="passthrough")

logistic_pipeline = Pipeline([
    ('datafeed', column_transformer),              # grabs finalized datasets
    ('selector', SelectKBest(f_classif, k='all')), # variable selection procedure
    ('classifier', LogisticRegression())           # Logistic modeling
])
```

Above is the pipeline used for our logistic regression model. The pipeline is a series of functions that the data is passed through, cumulating in the logistic regression model. In the pipeline, numeric values are first scaled to a z-score using the StandardScaler() function. This allows us to compare the coefficients of numeric variables to each other, and more specifically their respective magnitudes of impact on the model. The remaining variables are passed through, having been previously one-hot encoded.

A SelectKBest() function is called to specify how many features the classifier should consider for inclusion into the model. This function compares the impact of features on the model and selects the 'k' best features for inclusion. Finally, the logistic regression function is called which fits a model to the training data and cross-validates the model on the test data.

#### Logistic Model, Raw


```python
# fit the logistic model
logistic_pipeline.fit(X_train, y_train)
y_test_pred = logistic_pipeline.predict(X_test)

# get ROC AUC score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
rocscore = rocauc(y_test, y_test_pred)

#print results
print(f'Overall Accuracy: {100*logistic_pipeline.score(X_test, y_test)}%')
print(f'ROC AUC Score: {100*rocscore}%')
print(classification_report(y_test, y_test_pred))


#---------------------confusion matrix--------------------
#source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#source: https://stackoverflow.com/questions/33779748/set-max-value-for-color-bar-on-seaborn-heatmap
#source: https://python-graph-gallery.com/91-customize-seaborn-heatmap/
with Suppressor():
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', 
            vmin=9999999, vmax=9999999, linewidths=.5,
                  cbar=False).set(
        title="Confusion Matrix, Logistic Regression, Raw")
```

    Overall Accuracy: 92.02194488650642%
    ROC AUC Score: 89.94556088156838%
                  precision    recall  f1-score   support
    
               0       0.90      0.98      0.94     15033
               1       0.96      0.82      0.88      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
    
    


    
![png](\assets\images\logistic-svm-smote\output_27_1.png)
    


Above are the results for the logistic regression function using raw data. Numeric results are shown above a confusion matrix, including overall accuracy, an ROC area under the curve score, precisions, recalls, and f1-scores. These metrics can be used to compare to other models to determine which model performed the best. 

Findings:
* The overall accuracy for this preliminary model was impressive at 92%, meaning our model can correctly predict whether a reservation is or is not cancelled 92% of the time! 
* The ROC AUC score was 89.95%, meaning that our model is much better than randomly guessing (AUC = 50%) if a reservation is cancelled. 
* Recall for true negatives (successful check-outs) was 98%, so we have very few instances where there is not a cancellation and the model predicts that there will not be a cancellation.

Future adjustments:
This model used all 975 features in the dataset, which may not be desireable and could lead to overfitting. This model predicted non-cancellations better than cancellations, so parameter tuning will include class weights in an attempt to balance these numbers. The parameters used for this model were:
- solver: lbfgs
- k=975 (all)
- class weight false/true ratio of 1:1

#### Logistic Model, Random Under-Sampling


```python
# fit the logistic model
logistic_pipeline.fit(X_rus, y_rus)
y_test_pred = logistic_pipeline.predict(X_test)

#get ROC AUC score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
rocscore = rocauc(y_test, y_test_pred)

#print results
print(f'Overall Accuracy: {100*logistic_pipeline.score(X_test, y_test)}%')
print(f'ROC AUC Score: {100*rocscore}%')
print(classification_report(y_test, y_test_pred))


#---------------------confusion matrix--------------------
#source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#source: https://stackoverflow.com/questions/33779748/set-max-value-for-color-bar-on-seaborn-heatmap
#source: https://python-graph-gallery.com/91-customize-seaborn-heatmap/
with Suppressor():
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', 
            vmin=9999999, vmax=9999999, linewidths=.5,
                  cbar=False).set(
        title="Confusion Matrix, Logistic Regression, Random Under-Sampling")

```

    Overall Accuracy: 91.657592763213%
    ROC AUC Score: 90.67770445250335%
                  precision    recall  f1-score   support
    
               0       0.92      0.94      0.93     15033
               1       0.90      0.87      0.89      8845
    
        accuracy                           0.92     23878
       macro avg       0.91      0.91      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
    
    


    
![png](\assets\images\logistic-svm-smote\output_30_1.png)
    


Above is the random under-sampling model using unoptimized parameters and the randomly under-sampled training set. Results from this model will be compared with the raw and SMOTE models in the next discussion cell, as it will be more straightforward to compare results from the three models in a side-by-side fashion.

#### Logistic Model, SMOTE


```python
# fit the logistic model
logistic_pipeline.fit(X_sm, y_sm)
y_test_pred = logistic_pipeline.predict(X_test)

#get ROC AUC score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
rocscore = rocauc(y_test, y_test_pred)

#print results
print(f'Overall Accuracy: {100*logistic_pipeline.score(X_test, y_test)}%')
print(f'ROC AUC Score: {100*rocscore}%')
print(classification_report(y_test, y_test_pred))


#---------------------confusion matrix--------------------
#source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#source: https://stackoverflow.com/questions/33779748/set-max-value-for-color-bar-on-seaborn-heatmap
#source: https://python-graph-gallery.com/91-customize-seaborn-heatmap/
with Suppressor():
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', 
            vmin=9999999, vmax=9999999, linewidths=.5,
                  cbar=False).set(
        title="Confusion Matrix, Logistic Regression, SMOTE")

```

    Overall Accuracy: 92.16014741603149%
    ROC AUC Score: 90.46717985459311%
                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94     15033
               1       0.94      0.84      0.89      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
    
    


    
![png](\assets\images\logistic-svm-smote\output_33_1.png)
    


Above are the results from a model using unoptimized parameters and a SMOTE training set. The results of this model are shown below alongside the results from the other unoptimized models. The SMOTE model obtained the highest overall accuracy and the under-sampled model obtained the highest ROC AUC score. Weighted averages for precision, recall, and f1-score were the same for all models. Any of these models would be viable out of the box for cancellation prediction.

---------------------------
Unoptimized Raw:

    Overall Accuracy: 92.02194488650642%

    ROC AUC Score: 89.94556088156838%

                  precision    recall  f1-score   support
    
               0       0.90      0.98      0.94     15033
               1       0.96      0.82      0.88      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
---------------------------
Unoptimized Under-Sampled:

    Overall Accuracy: 91.657592763213%

    ROC AUC Score: 90.67770445250335%

                  precision    recall  f1-score   support
    
               0       0.92      0.94      0.93     15033
               1       0.90      0.87      0.89      8845
    
        accuracy                           0.92     23878
       macro avg       0.91      0.91      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
---------------------------
Unoptimized SMOTE:

    Overall Accuracy: 92.16014741603149%

    ROC AUC Score: 90.46717985459311%

                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94     15033
               1       0.94      0.84      0.89      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878

### Hyperparameter optimization using Grid Search


```python
# # defining parameter range 
# param_grid = {'classifier__class_weight': [{False:0.9, True:1}, {False:0.95, True:1}, {False:0.85, True:1}], 
#               'classifier__solver': ['liblinear', 'lbfgs'],
#               'selector__k': list(range(922,975,1))
#               }  
  
# grid = GridSearchCV(logistic_pipeline, param_grid, refit = True, verbose = 3) 
  
# # fitting the model for grid search 
# grid.fit(X_train, y_train)
    
# print(grid.best_estimator_)
```

If the above code is commented out, that is because it takes a very long time to run and we did it to be able to compile the entire notebook in order.

A grid search function was performed using the logistic pipeline in order to optimize model parameters. Grid searches operate by generating a model for each possible combination of the specified hyperparameters, then selecting the best performing model. The parameters that were tuned using this method were the solver method, the true/false weights, and the number of features k. The output of the grid search was minimized; to see it click the elipses above. Possible values for these variables were as such:

- solver method: liblinear, lbfgs, or saga. liblinear consistently outperformed the other methods.
- class weights: false/true ratios of 1:1, 0.9:1, 0.8:1, and later, 0.95:1 and 0.85:1
- k: a range from 300 to 975 features were tested in varying steps. The final grid search ranged from 922 to 975 in steps of 1.

The grid search was run iteratively to determine which of these parameters provided the best fit from the model. Parameters of solver=liblinear, k=922, and a weight ratio of 0.9:1 scored the best with the grid search. The grid search optimized the model over a number of metrics, and this will be relevant during analysis of the adjusted linear model. Also shown below are the results of some of the grid searches run for this model, in chronological order. 

                ('selector', SelectKBest(k=460)),
                ('classifier',
                 LogisticRegression(class_weight={False: 1, True: 1},
                                    solver='liblinear'))])
                                    
                                    
                ('selector', SelectKBest(k=550)),
                ('classifier',
                 LogisticRegression(class_weight={False: 1, True: 1},
                                    solver='liblinear'))])
                                    
                ('selector', SelectKBest(k=925)),
                ('classifier',
                 LogisticRegression(class_weight={False: 0.9, True: 1},
                                    solver='liblinear'))])            
                                    
                ('selector', SelectKBest(k=922)),
                ('classifier',
                 LogisticRegression(class_weight={False: 0.9, True: 1},
                                    solver='liblinear'))])            


                ('selector', SelectKBest(k=922)),
                ('classifier',
                 LogisticRegression(class_weight={False: 0.9, True: 1},
                                    solver='liblinear'))])

### Raw Logistic Model, Optimized


```python
#source: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
#source: https://www.kdnuggets.com/2020/06/simplifying-mixed-feature-type-preprocessing-scikit-learn-pipelines.html
#column transformer: https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260
#standard scaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#pipeline stuff: https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156
#more pipeline stuff: https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
#even more pipeline stuff: https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
#onehot encoder unknown categories error: https://www.roelpeters.be/found-unknown-categories-in-column-sklearn/

#scale numerics and perform the logistic regression

column_transformer = ColumnTransformer([
    ("scaler", StandardScaler(), int_var_list)
#    ("standardizer", custom_scaler(int_var_list), int_var_list)
], remainder="passthrough")

logistic_pipeline = Pipeline([
    ('datafeed', column_transformer), #grabs finalized datasets
    ('selector', SelectKBest(f_classif, k=922)),   # selection procedure
    ('classifier', LogisticRegression(solver='liblinear', class_weight={False:0.9, True:1})) #class_weight={False:0.1, True:1}) # Logistic modeling using class weights for our imbalanced dataset
])


# fit the logistic model
logistic_pipeline.fit(X_train, y_train)
y_test_pred = logistic_pipeline.predict(X_test)

#get ROC AUC score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
rocscore = rocauc(y_test, y_test_pred)

#print results
print(f'Overall Accuracy: {100*logistic_pipeline.score(X_test, y_test)}%')
print(f'ROC AUC Score: {100*rocscore}%')
print(classification_report(y_test, y_test_pred))




#---------------------confusion matrix--------------------
#source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#source: https://stackoverflow.com/questions/33779748/set-max-value-for-color-bar-on-seaborn-heatmap
#source: https://python-graph-gallery.com/91-customize-seaborn-heatmap/
with Suppressor():
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', 
            vmin=9999999, vmax=9999999, linewidths=.5,
                  cbar=False).set(title="Confusion Matrix, Logistic Regression, Raw")
```

    Overall Accuracy: 91.74553982745624%
    ROC AUC Score: 89.84471547182594%
                  precision    recall  f1-score   support
    
               0       0.90      0.97      0.94     15033
               1       0.95      0.83      0.88      8845
    
        accuracy                           0.92     23878
       macro avg       0.92      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
    
    


    
![png](\assets\images\logistic-svm-smote\output_39_1.png)
    


Parameters chosen by grid search were implemented in the model and the results are shown above. Results from both tests are also shown below to make comparison easier. The optimized model resulted in only slightly adjusted numbers; primarily, the parameters resulted in a slight detriment to precision in favor of a slight boost to recall when predicting cancellations, resulting in a precision that dropped from 0.96 to 0.95, and a recall that rose from 0.82 to 0.83. This resulted in an overall accuracy drop from 92.02% to 91.99% (0.03%), and an ROC AUC increase from 89.95% to 90.09% (0.14%). All other numbers shown in outputs remained the same. It is interesting to see that grid search results caused the overall accuracy to drop from the model, but not surprising as grid search optimizes parameters based on numerous scoring metrics.



Unoptimized Model:

Overall Accuracy: 92.02194488650642%

ROC AUC Score: 89.94556088156838%

                  precision    recall  f1-score   support
    
               0       0.90      0.98      0.94     15033
               1       0.96      0.82      0.88      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
---------------------------

Optimized Model:

Overall Accuracy: 91.98844124298517%

ROC AUC Score: 90.09114299398681%

                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94     15033
               1       0.95      0.83      0.88      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878

#### Logistic Model, Random Under-Sampling, Optimized Parameters


```python
# fit the logistic model
logistic_pipeline.fit(X_rus, y_rus)
y_test_pred = logistic_pipeline.predict(X_test)

#get ROC AUC score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
rocscore = rocauc(y_test, y_test_pred)

#print results
print(f'Overall Accuracy: {100*logistic_pipeline.score(X_test, y_test)}%')
print(f'ROC AUC Score: {100*rocscore}%')
print(classification_report(y_test, y_test_pred))


#---------------------confusion matrix--------------------
#source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#source: https://stackoverflow.com/questions/33779748/set-max-value-for-color-bar-on-seaborn-heatmap
#source: https://python-graph-gallery.com/91-customize-seaborn-heatmap/
with Suppressor():
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', 
            vmin=9999999, vmax=9999999, linewidths=.5,
                  cbar=False).set(
        title="Confusion Matrix, Logistic Regression, Random Under-Sampling")

```

    Overall Accuracy: 91.9130580450624%
    ROC AUC Score: 91.12956921567353%
                  precision    recall  f1-score   support
    
               0       0.93      0.94      0.94     15033
               1       0.90      0.88      0.89      8845
    
        accuracy                           0.92     23878
       macro avg       0.91      0.91      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
    
    


    
![png](\assets\images\logistic-svm-smote\output_42_1.png)
    


Above is the random under-sampling model using optimized parameters and the randomly under-sampled training set. Results from this model will be compared with the raw and SMOTE models in the next discussion cell, as it will be more straightforward to compare results from the three models in a side-by-side fashion.

#### Logistic Model, SMOTE, Optimized Parameters


```python
# fit the logistic model
logistic_pipeline.fit(X_sm, y_sm)
y_test_pred = logistic_pipeline.predict(X_test)

#get ROC AUC score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
rocscore = rocauc(y_test, y_test_pred)

#print results
print(f'Overall Accuracy: {100*logistic_pipeline.score(X_test, y_test)}%')
print(f'ROC AUC Score: {100*rocscore}%')
print(classification_report(y_test, y_test_pred))


#---------------------confusion matrix--------------------
#source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#source: https://stackoverflow.com/questions/33779748/set-max-value-for-color-bar-on-seaborn-heatmap
#source: https://python-graph-gallery.com/91-customize-seaborn-heatmap/
with Suppressor():
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', 
            vmin=9999999, vmax=9999999, linewidths=.5,
                  cbar=False).set(
        title="Confusion Matrix, Logistic Regression, SMOTE")

```

    Overall Accuracy: 92.30253790099673%
    ROC AUC Score: 90.73849327221585%
                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94     15033
               1       0.94      0.85      0.89      8845
    
        accuracy                           0.92     23878
       macro avg       0.93      0.91      0.92     23878
    weighted avg       0.92      0.92      0.92     23878
    
    


    
![png](\assets\images\logistic-svm-smote\output_45_1.png)
    


Below are the results from the optimized models using raw, under-sampled, and over-sampled training data. The highest overall accuracy was achieved by the SMOTE model, and the higest ROC AUC score was achieved by the under-sampled model. Weighted averages for precision, recall, and f1-score were the same accross all three models. Additionally, weighted averages for precision, recall, and f1-score were the same for the unoptimized models. Of these three, is recommended to use the under-sampled or over-sampled models for prediction. 


    Raw Model, Optimized:
    Overall Accuracy: 91.74553982745624%
    ROC AUC Score: 89.84471547182594%
                  precision    recall  f1-score   support

               0       0.90      0.97      0.94     15033
               1       0.95      0.83      0.88      8845

        accuracy                           0.92     23878
       macro avg       0.92      0.90      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
--------------------------- 
    
    Under-Sampled Model, Optimized:
    Overall Accuracy: 91.9130580450624%
    ROC AUC Score: 91.12956921567353%
                  precision    recall  f1-score   support

               0       0.93      0.94      0.94     15033
               1       0.90      0.88      0.89      8845

        accuracy                           0.92     23878
       macro avg       0.91      0.91      0.91     23878
    weighted avg       0.92      0.92      0.92     23878
---------------------------
    
    
    SMOTE Model, Optimized:
    Overall Accuracy: 92.30253790099673%
    ROC AUC Score: 90.73849327221585%
                  precision    recall  f1-score   support

               0       0.91      0.97      0.94     15033
               1       0.94      0.85      0.89      8845

        accuracy                           0.92     23878
       macro avg       0.93      0.91      0.92     23878
    weighted avg       0.92      0.92      0.92     23878


#### The code must be re-cleaned in order to evaluate logistic regression feature importance.


```python
df = pd.read_csv("../Data/hotel_bookings.csv")

#replace missing values in certain columns
#source: https://datatofish.com/replace-nan-values-with-zeros/
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna("unknown")
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# Drop outlier
df.drop(df[ df['adr'] == 5400 ].index , inplace=True)
df.drop(['reservation_status_date'], axis=1)

#convert reservation_status_date to day, month, year
#source: https://stackoverflow.com/questions/25789445/pandas-make-new-column-from-string-slice-of-another-column
df['reservation_status_year'] = df.reservation_status_date.str[:4]
df['reservation_status_month'] = df.reservation_status_date.str[5:7]
df['reservation_status_day'] = df.reservation_status_date.str[8:10]

#convert categoricals to proper data type
df = df.astype({"agent":'category', "company":'category', "is_canceled":'category', 
                "hotel":'category', "is_repeated_guest":'category',
                "reserved_room_type":'category', "assigned_room_type":'category',
                "deposit_type":'category', "customer_type":'category',
                "country":'category', 
                "arrival_date_month":'category', "meal":'category', 
                "market_segment":'category', 'reservation_status_year':'category',
                "distribution_channel":'category', 'reservation_status_month':'category',
                'reservation_status_day':'category'
               })

#set the y column to its own dataframe
y_column = df['is_canceled']
del df['is_canceled']
del df['reservation_status_date']
del df['reservation_status']
```

##### One-hot encoding


```python
#split dataframes into categorical and integer
#source: https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type
df_cat = df.select_dtypes(include=['category'])
df_int = df.select_dtypes(exclude=['category'])

#list factors by dtype
cat_var_list = df_cat.columns.tolist()
int_var_list = df_int.columns.tolist()

#dummy encode: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
cat_enc = pd.get_dummies(df_cat, drop_first=True)
hot_var_list = cat_enc.columns.tolist()

#merge encoded and integer dataframes
df_enc = cat_enc.merge(df_int, left_index=True, right_index=True)
enc_var_list = df_enc.columns.tolist()
```

## Logistic Feature Importance


```python
#see coefficients: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#https://stackoverflow.com/questions/58615904/how-to-extract-coefficients-from-fitted-pipeline-for-penalized-logistic-regressi
#https://sweetcode.io/easy-scikit-logistic-regression/
#https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le

mask = logistic_pipeline['selector'].get_support()
new_features = df_enc.columns[mask]
new_features
select_vars = new_features.tolist()

#https://sweetcode.io/easy-scikit-logistic-regression/
# Get the models coefficients (and top 5 and bottom 5)
logReg_coeff = pd.DataFrame({'feature_name': select_vars, 'model_coefficient': logistic_pipeline['classifier'].coef_[0].transpose().flatten()})
logReg_coeff = logReg_coeff.sort_values('model_coefficient',ascending=False)
logReg_coeff_top = logReg_coeff.head(5)
logReg_coeff_bottom = logReg_coeff.tail(5)

# Plot top 5 coefficients
plt.figure().set_size_inches(10, 6)
fg3 = sns.barplot(x='feature_name', y='model_coefficient',data=logReg_coeff_top, palette="Blues_d")
fg3.set_xticklabels(rotation=35, labels=logReg_coeff_top.feature_name)
# Plot bottom 5 coefficients
plt.figure().set_size_inches(10,6)
fg4 = sns.barplot(x='feature_name', y='model_coefficient',data=logReg_coeff_bottom, palette="GnBu_d")
fg4.set_xticklabels(rotation=35, labels=logReg_coeff_bottom.feature_name)
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.subplots_adjust(bottom=0.4)
plt.savefig('figure_4.png')
```


    
![png](\assets\images\logistic-svm-smote\output_70_0.png)
    



    
![png](\assets\images\logistic-svm-smote\output_70_1.png)
    


Above are graphs showing ten of the most influential features on the model, both positive and negative. The uppermost graph shows coefficients that increased the likelihood that somebody would cancel their reservation, and the lowermost graph show features that decrease the likelihood that somebody would cancel their reservation.

The features which increase the likelihood of someone cancelling their reservation are:
- arrival_date_month_August: If an individual's arrival date is in August, they are more likely to cancel their hotel reservation.
- reservation_status_day_04: The fourth day of any given month sees a disproportionately large number of cancellations compared to other days of the month.
- arrival_date_month_December: If an individual's arrival date is in November, they are more likely to cancel their hotel reservation.
- agent_15.0: Booking through this agent makes a cancellation more likely.
- deposit_type_Refundable: If the deposit type is refundable, cancellations become more likely.


The features which decrease the likelihood of someone cancelling their reservation are:
- reservation_status_day_06: If individuals are scheduled to check out on the 6th day of the month, they are less likely to cancel their reservation. This could be related to stays that extend over the first week of a month, which may be a popular time for travel. This factor was the most influential in the model as it had the largest absolute coefficient, meaning that there does seem to be an explainable trend behind the sixth day of a month being a popular check-out day.
- reservation_status_day_14, 15, 16, and 17: If individuals are scheduled to check out in the middle of the month, they are less likely to cancel their reservation. It is surprising to see that this specific string of days aligned to be four of the five most influential factors in the model for individuals not cancelling their stay. Similarly to reservation_status_day_06, there must be an explainable trend behind individuals choosing to check out in the middle of the month.

These initial results are promising, since the attributes which are percieved by the model as being influential on a reservation being cancelled seem logical. Vacations are taken during August and December, and those types of plans seem like they could change more readily, individual agents may encourage their customers to change hotels for a better deal, and having all of your deposit refunded upon cancelling would not discourage a customer from cancelling.
