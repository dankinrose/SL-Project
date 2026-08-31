[README (1).md](https://github.com/user-attachments/files/31638303/README.1.md)
# Supervised Learning Projects

This repository contains two supervised machine learning projects
implemented in Jupyter Notebooks:

1.  **Customer Churn Classification** --- predicts whether a customer
    will churn.
2.  **GDP Regression** --- predicts country-level GDP from COVID-19 and
    economic indicators.

------------------------------------------------------------------------

## Project Structure

``` text
.
├── SLClassificationPro.ipynb
├── SLRegProject.ipynb
├── final_rf_model.joblib
├── churn_feature_columns.joblib
├── final_model_linear.joblib
└── scaler_full.joblib
```

> The `.joblib` files are the exported models/preprocessing objects
> produced by the notebooks.

------------------------------------------------------------------------

# 1. Customer Churn Classification

## Objective

The goal of this project is to build a classification model that
predicts the `Churn` label for customers using demographic, usage,
support, payment, subscription, and contract-related features.

The dataset contains **64,374 rows and 12 columns** before
preprocessing.

### Features

-   `Age`
-   `Gender`
-   `Tenure`
-   `Usage Frequency`
-   `Support Calls`
-   `Payment Delay`
-   `Subscription Type`
-   `Contract Length`
-   `Total Spend`
-   `Last Interaction`

### Target

-   `Churn` --- binary classification target (`0` / `1`)

`CustomerID` is removed because it is an identifier and is not
considered predictive.

## Data Preparation

The notebook performs the following steps:

1.  Load the customer churn dataset.
2.  Remove `CustomerID`.
3.  Check for missing values --- no missing values were found.
4.  Check for duplicated rows --- no duplicated rows were found.
5.  Convert categorical variables to numerical variables using
    `pd.get_dummies(..., drop_first=True)`.
6.  Analyze feature correlations.
7.  Split the data into training and test sets using a **70/30 split**
    with `random_state=101`.
8.  Standardize the training and test features using `StandardScaler`.

## Exploratory Analysis

The correlation analysis identified:

-   `Payment Delay` as the strongest positive correlation with `Churn`
    (0.56).
-   `Support Calls` as another positive relationship (0.30).
-   `Tenure` with a positive correlation of 0.20.
-   `Usage Frequency` (-0.12) and `Total Spend` (-0.08) showed weak
    negative correlations with churn.

These relationships were explored using correlation plots, pairplots,
and other visualizations.

## Models

Four classification algorithms were trained and evaluated:

-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Support Vector Machine (SVM)
-   Random Forest

Hyperparameter optimization was performed with `GridSearchCV` using
F1-score and 3-fold cross-validation.

### Selected Parameters

  -----------------------------------------------------------------------
  Model                               Selected parameters
  ----------------------------------- -----------------------------------
  Logistic Regression                 `C=0.001`, `penalty=l1`,
                                      `solver=liblinear`

  KNN                                 `n_neighbors=10`

  SVM                                 `C=10`, `kernel=rbf`

  Random Forest                       `n_estimators=500`,
                                      `max_depth=None`,
                                      `min_samples_split=2`,
                                      `min_samples_leaf=1`,
                                      `bootstrap=True`, `oob_score=True`
  -----------------------------------------------------------------------

For SVM, the hyperparameter search was performed on a stratified
subsample of **10,000 training rows** to reduce computational cost. The
selected parameters were then used to retrain the SVM on the full
training set.

## Results

  Model                    Accuracy      Recall    F1-Score
  --------------------- ----------- ----------- -----------
  Logistic Regression         0.826       0.861       0.823
  KNN                         0.912       0.921       0.908
  SVM                         0.953       0.959       0.951
  **Random Forest**       **0.998**   **0.997**   **0.998**

The Random Forest achieved the best results on the test set. Its
reported OOB score was **0.9982**.

## Final Model

Random Forest was selected as the final churn prediction model.

The final model was retrained on the entire dataset and exported as:

``` text
final_rf_model.joblib
```

The feature-column order was also saved:

``` text
churn_feature_columns.joblib
```

Saving the feature columns helps ensure that future prediction data uses
the same feature structure and ordering as the training data.

------------------------------------------------------------------------

# 2. GDP Regression

## Objective

The goal of this project is to predict country-level **GDP** using
COVID-19 statistics and economic indicators.

The source data contains COVID-19 and economic information including:

-   Confirmed cases
-   Deaths
-   Recovered cases
-   GDP
-   Unemployment
-   CPI

## Data Preparation

The notebook performs the following preprocessing:

1.  Load the COVID/GDP dataset.
2.  Remove `Unnamed: 0`, which is a technical CSV index.
3.  Remove `Province/State`, because it contains mostly zero values and
    does not contribute to the prediction.
4.  Remove the two rows containing missing values, leaving **338 rows**.
5.  Investigate zero values, especially in `Recovered`.
6.  Address the large number of zero `Recovered` values in 2022 by
    estimating values using the 2021 country-level
    `Recovered / Confirmed` ratio.
7.  For three countries without usable 2021 recovery information
    (`Belgium`, `Serbia`, and `Sweden`), use the median-based fallback
    described in the notebook.
8.  Aggregate the data by country:
    -   `SUM` for `Confirmed`, `Deaths`, and `Recovered`.
    -   `MEAN` for `GDP`, `Unemployment`, and `CPI`.
9.  Scale GDP down by `1e9` before model training.
10. Split the data into training and test sets using a **70/30 split**
    with `random_state=42`.
11. Standardize the input features using `StandardScaler`.

## Features and Target

### Features

``` text
Confirmed
Deaths
Recovered
Unemployment
CPI
```

### Target

``` text
GDP
```

## Exploratory Analysis

The correlation analysis showed strong relationships among the
COVID-related variables:

-   Confirmed ↔ Deaths: **0.90**
-   Confirmed ↔ Recovered: **0.93**
-   Deaths ↔ Recovered: **0.84**

These high correlations indicate substantial multicollinearity among the
COVID-related variables.

The relationships with GDP were weaker:

-   Confirmed ↔ GDP: **0.27**
-   Deaths ↔ GDP: **0.24**
-   Recovered ↔ GDP: **0.21**

The notebook therefore concludes that the COVID variables have only mild
individual correlations with GDP.

## Regression Models

Four regression approaches were evaluated:

-   Linear Regression
-   RidgeCV
-   LassoCV
-   Polynomial Regression

For Polynomial Regression, degrees **1 through 8** were tested. The test
RMSE increased substantially for degrees above 1, so degree 1 was
selected.

### Selected Parameters

  Model                   Selected parameter
  ----------------------- ------------------------
  Linear Regression       Best CV RMSE: 340.7692
  RidgeCV                 `alpha ≈ 159.9859`
  LassoCV                 `alpha ≈ 71.6729`
  Polynomial Regression   `degree=1`

## Results

  Model                             MAE               MSE          RMSE
  ----------------------- ------------- ----------------- -------------
  **Linear Regression**     **362.179**   **422,340.918**   **649.878**
  Ridge                         423.220       522,573.116       722.892
  Lasso                         395.675       472,225.305       687.187
  Polynomial (degree 1)         362.179       422,340.918       649.878

Lower values are better for MAE, MSE, and RMSE.

Linear Regression achieved the lowest errors. Polynomial Regression with
degree 1 produced identical results, which is expected because degree 1
corresponds to a linear model.

## Linear Regression Coefficients

The standardized-model coefficients reported in the notebook are:

  Feature          Coefficient
  -------------- -------------
  Confirmed         729.687993
  Deaths            -40.491444
  Recovered        -372.858410
  Unemployment     -149.613059
  CPI               -28.997966

## Final Model

Linear Regression was selected as the final GDP prediction model.

The model was retrained on the entire processed dataset after fitting a
new `StandardScaler`.

The exported files are:

``` text
final_model_linear.joblib
scaler_full.joblib
```

These files can be used together to apply the same scaling process and
load the trained regression model.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Jupyter Notebook
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   SciPy
-   Scikit-learn
-   Joblib

## Main Scikit-learn Components

### Classification

-   `StandardScaler`
-   `train_test_split`
-   `GridSearchCV`
-   `LogisticRegression`
-   `KNeighborsClassifier`
-   `SVC`
-   `RandomForestClassifier`
-   Classification metrics and confusion matrices

### Regression

-   `StandardScaler`
-   `PolynomialFeatures`
-   `train_test_split`
-   `KFold`
-   `cross_val_score`
-   `LinearRegression`
-   `RidgeCV`
-   `LassoCV`
-   MAE / MSE / RMSE evaluation

------------------------------------------------------------------------

# How to Run

1.  Install the required Python packages:

``` bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib jupyter
```

2.  Place the datasets in an accessible location.

3.  Open the relevant notebook:

``` bash
jupyter notebook
```

4.  Run the cells in order.

> **Important:** The notebooks currently contain local Windows paths to
> the datasets. Before running them on another computer, update the
> dataset paths to match your local environment.

------------------------------------------------------------------------

# Outputs

The classification project produces:

``` text
final_rf_model.joblib
churn_feature_columns.joblib
```

The regression project produces:

``` text
final_model_linear.joblib
scaler_full.joblib
```

------------------------------------------------------------------------

# Summary

  -----------------------------------------------------------------------
  Project           Task              Final Model       Main Result
  ----------------- ----------------- ----------------- -----------------
  Customer Churn    Classification    **Random Forest** Accuracy:
                                                        **0.998**, F1:
                                                        **0.998**

  GDP Prediction    Regression        **Linear          RMSE:
                                      Regression**      **649.878**, MAE:
                                                        **362.179**
  -----------------------------------------------------------------------

Both projects demonstrate a complete supervised-learning workflow: data
preparation, exploratory analysis, feature preprocessing, model
training, hyperparameter selection, evaluation, model selection, and
model export.
