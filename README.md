# Machine Learning Clustering and Classification Project Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Clustering Analysis](#clustering-analysis)
   - [Dataset Overview](#dataset-overview)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Development](#model-development)
   - [Cluster Analysis](#cluster-analysis)
3. [Classification Analysis](#classification-analysis)
   - [Model Building](#model-building)
   - [Model Evaluation](#model-evaluation)
   - [Model Tuning](#model-tuning)
   - [Results Analysis](#results-analysis)
4. [Conclusion](#conclusion)

## Introduction

This documentation outlines the process of unsupervised clustering followed by supervised classification on banking transaction data. The project demonstrates how to discover natural groupings in unlabeled data through clustering, then using those cluster assignments as labels for a classification model.

## Clustering Analysis

### Dataset Overview

For this analysis, we used a bank transactions dataset with the following characteristics:
- **Source**: Banking transaction records
- **Size**: We sampled 1.5% from the original dataset (which contained thousands of records)
- **Features**: The dataset includes both numerical features (account balance, transaction amount, transaction time) and categorical features (customer gender, customer location)

```python
# Loading the dataset
df = pd.read_csv('bank_transactions.csv')

# Using 1.5% of dataset for analysis
df_sampled = df.sample(frac=0.015, random_state=32)

print(f'Total data: {len(df)} records')
print(f'Sample data used: {len(df_sampled)} records')
```

### Data Preprocessing

The preprocessing phase involved several crucial steps to prepare the data for clustering:

1. **Feature Selection**: We selected key features for clustering analysis
   ```python
   selected_features = ['CustAccountBalance', 'TransactionAmount (INR)', 
                       'TransactionTime', 'CustGender', 'CustLocation']
   df_selected = df_sampled[selected_features].copy()
   ```

2. **Handling Missing Values**:
   ```python
   # For numerical features
   imputer_num = SimpleImputer(strategy='mean')
   df_selected[numerical_features] = imputer_num.fit_transform(df_selected[numerical_features])
   
   # For categorical features
   imputer_cat = SimpleImputer(strategy='most_frequent')
   df_selected[categorical_features] = imputer_cat.fit_transform(df_selected[categorical_features])
   ```

3. **Feature Standardization**: To ensure equal contribution of all features
   ```python
   scaler = StandardScaler()
   df_selected[numerical_features] = scaler.fit_transform(df_selected[numerical_features])
   ```

4. **Categorical Encoding**: Converting categorical data to numerical format
   ```python
   label_encoders = {}
   for col in categorical_features:
       le = LabelEncoder()
       df_selected[col] = le.fit_transform(df_selected[col])
       label_encoders[col] = le
   ```

### Model Development

We employed the K-Means clustering algorithm and determined the optimal number of clusters using both the Elbow Method and Silhouette Score:

```python
range_n_clusters = list(range(2, 11))
silhouette_avg = []
inertia = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=22, n_init=10)
    cluster_labels = kmeans.fit_predict(df_selected)
    inertia.append(kmeans.inertia_)
    silhouette_avg.append(silhouette_score(df_selected, cluster_labels))
```

We applied feature selection to improve model performance:

```python
# Feature Selection
X = df_selected.drop(columns=['CustGender'])
y = df_selected['CustGender']
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()]
df_selected = df_selected[selected_columns]
```

### Cluster Analysis

After performing clustering, we analyzed the characteristics of each cluster:

**Cluster 0:**  
- **Avg CustAccountBalance:** 135.86 billion (Min: 119.90K, Max: 13.41T)  
- **Avg TransactionTime:** 7.95 billion  
- **Avg TransactionAmount:** 14.59 million (Min: 1.68K, Max: 4.05B)  
- **Dominant Demographics:** NEW DELHI, Male  
- **Analysis:** Customers with large account balances and high transaction volumes, indicating strong financial stability and active transactional behavior.

**Cluster 1:**  
- **Avg CustAccountBalance:** 90.62 billion (Min: 119.90K, Max: 8.49T)  
- **Avg TransactionTime:** 8.03 billion  
- **Avg TransactionAmount:** 12.30 million (Min: 6.41K, Max: 726.60M)  
- **Dominant Demographics:** BANGALORE, Male  
- **Analysis:** Upper-mid-tier customers with stable transaction patterns.

**Cluster 2:**  
- **Avg CustAccountBalance:** 160.59 billion (Min: 119.90K, Max: 75.35T)  
- **Avg TransactionTime:** 7.97 billion  
- **Avg TransactionAmount:** 10.65 million (Min: 1.68K, Max: 871.91M)  
- **Dominant Demographics:** GURGAON, Male  
- **Analysis:** Highest average balance but smaller average transaction amounts; possibly selective high-value customers.

**Clusters 3-9 follow similar analytical patterns, each with distinct characteristics based on geographic location and financial behavior.**

## Classification Analysis

Using the cluster assignments as target labels, we developed classification models to predict customer segments based on their transaction behavior.

### Model Building

For classification, we implemented both Random Forest and XGBoost models:

1. **Data Preparation**:
   ```python
   # Split data into training, evaluation, and test sets
   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
   X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
   
   # Preprocess categorical features
   encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
   X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
   
   # Handle missing values and standardize
   imputer = SimpleImputer(strategy="median")
   X_train = imputer.fit_transform(X_train)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   ```

2. **Class Imbalance Handling**:
   ```python
   # Apply SMOTE to balance classes
   smote = SMOTE(random_state=42)
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
   
   # Calculate class weights
   class_weights = compute_class_weight("balanced", classes=np.unique(y_train_resampled), 
                                      y=y_train_resampled)
   class_weight_dict = {cls: weight for cls, weight in 
                       zip(np.unique(y_train_resampled), class_weights)}
   ```

3. **Model Training**:
   ```python
   # Random Forest model
   rf_model = RandomForestClassifier(n_estimators=200, random_state=42, 
                                   class_weight=class_weight_dict)
   rf_model.fit(X_train_resampled, y_train_resampled)
   
   # XGBoost model
   xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                            random_state=42)
   xgb_model.fit(X_train_resampled, y_train_resampled)
   ```

### Model Evaluation

We evaluated both models using standard classification metrics:

```python
# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
print("Classification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("Classification Report - XGBoost:")
print(classification_report(y_test, y_pred_xgb))
```

### Model Tuning

We performed hyperparameter tuning using GridSearchCV:

```python
# Random Forest tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=64, 
                                               class_weight=class_weight_dict), 
                         param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_rf_model = grid_search.best_estimator_

# Similar process for XGBoost
```

### Results Analysis

**Performance Comparison:**

| **Model**  | **Accuracy Before** | **Accuracy After** | **Change** |
|------------|---------------------|---------------------|------------|
| **Random Forest** | 94% | 94% | **No change** |
| **XGBoost** | 93% | 93% | **No change** |

**Key Insights:**
- Random Forest achieved the highest accuracy (94%)
- Hyperparameter tuning did not significantly improve either model
- Some classes showed low recall, particularly classes with fewer samples
- SMOTE was effective but didn't completely resolve class imbalance issues

**Recommendations for Further Improvement:**
1. Explore a wider hyperparameter space:
   ```python
   param_grid = {
       'n_estimators': [500, 700, 900],  
       'max_depth': [50, 70, 90, None],  
       'min_samples_split': [2, 5, 10, 15, 20],  
       'min_samples_leaf': [1, 2, 4, 8],  
       'max_features': ['sqrt', 'log2'],  
       'bootstrap': [True, False]
   }
   ```
2. Try alternative models like LightGBM or CatBoost
3. Perform additional feature engineering to extract more information from the data

## Conclusion

This project successfully demonstrated:
1. The application of K-Means clustering to identify natural customer segments in banking transaction data
2. The development of classification models to predict these segments with high accuracy (94%)
3. The effectiveness of feature selection, data preprocessing, and class imbalance handling techniques

The optimal Random Forest model provides a robust framework for predicting customer segments, which can be valuable for targeted marketing campaigns, risk assessment, and service personalization in the banking sector.
