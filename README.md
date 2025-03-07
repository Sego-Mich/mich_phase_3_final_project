# Business Understanding

## Project Overview
 Analysis of Vaccination Patterns from the National 2009 H1N1 Flu Survey
 
## Business problem
A vaccine for the H1N1 flu virus became publicly available in October 2009. In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. A better understanding of how these characteristics are associated with personal vaccination patterns can provide guidance for future public health efforts.
 
## Project objectives:
## Main Objective  
To analyze the demographic characteristics of respondents, including age, education, income, employment, and household composition.  

## Specific Objectives  
1. **Age Distribution** – Examine the age group distribution among respondents.  
  - **Feature Used:** age_group  

2. **Educational Attainment** – Analyze the levels of education across different respondents.  
  - **Feature Used:** education  

3. **Income and Employment Status** – Assess variations in income levels and employment status.  
  - **Features Used:** income_poverty, employment_status, employment_industry, employment_occupation  

4. **Household Composition** – Investigate household structure based on marital status, homeownership, and number of adults/children.  
  - **Features Used:** marital_status, rent_or_own, household_adults, household_children  

5. **Geographic Demographics** – Identify demographic variations across different regions.  
  - **Features Used:** hhs_geo_region, census_msa  


# Data Understanding 

## Data collection
The data for this competition comes from the National 2009 H1N1 Flu Survey (NHFS).

In their own words:

>The National 2009 H1N1 Flu Survey (NHFS) was sponsored by the National Center for Immunization and Respiratory Diseases (NCIRD) and conducted jointly by NCIRD and the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC). The NHFS was a list-assisted random-digit-dialing telephone survey of households, designed to monitor influenza immunization coverage in the 2009-10 season.

The target population for the NHFS was all persons 6 months or older living in the United States at the time of the interview. Data from the NHFS were used to produce timely estimates of vaccination coverage rates for both the monovalent pH1N1 and trivalent seasonal influenza vaccines.

The NHFS was conducted between October 2009 and June 2010. It was one-time survey designed specifically to monitor vaccination during the 2009-2010 flu season in response to the 2009 H1N1 pandemic. The CDC has other ongoing programs for annual phone surveys that continue to monitor seasonal flu vaccination.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")

**Loading Dataset**

Features for training

# Features Training
train_features_df = pd.read_csv("data/training_set_features.csv")

Features for testing

#Features Test
test_features_df = pd.read_csv("data/test_set_features.csv")

Training Labels

train_labels_df = pd.read_csv("data/training_set_labels.csv")

**training_set_features.csv**

train_features_df.info()

## Numerical Columns (26)
- respondent_id  
- h1n1_concern  
- h1n1_knowledge  
- behavioral_antiviral_meds  
- behavioral_avoidance  
- behavioral_face_mask  
- behavioral_wash_hands  
- behavioral_large_gatherings  
- behavioral_outside_home  
- behavioral_touch_face  
- doctor_recc_h1n1  
- doctor_recc_seasonal  
- chronic_med_condition  
- child_under_6_months  
- health_worker  
- health_insurance  
- opinion_h1n1_vacc_effective  
- opinion_h1n1_risk  
- opinion_h1n1_sick_from_vacc  
- opinion_seas_vacc_effective  
- opinion_seas_risk  
- opinion_seas_sick_from_vacc  
- household_adults  
- household_children  

## Categorical Columns (10)
- age_group  
- education  
- race  
- sex  
- income_poverty  
- marital_status  
- rent_or_own  
- employment_status  
- hhs_geo_region  
- census_msa  
- employment_industry  
- employment_occupation  


**Select the feautures**

Features selected are to explore how demographic factors effect 

train_df = pd.merge(train_features_df, train_labels_df, on="respondent_id")
train_df.drop(columns="respondent_id", inplace=True)

selected_features = ["age_group", "education", "income_poverty", "employment_status",
    "race", "sex", "marital_status", "rent_or_own","household_adults", "household_children","hhs_geo_region", "census_msa", "h1n1_vaccine","seasonal_vaccine"]
train_df = train_df[selected_features]

train_df.describe()

train_df.info()

train_df.head()

train_df.tail()

train_df.sample(5)

**testing_set_features.csv**

test_features_df.info()

**Select the feautures**

Features selected are to explore how demographic factors effect 

test_features_df = test_features_df[["respondent_id","age_group", "education", "income_poverty", "employment_status",
    "race", "sex", "marital_status", "rent_or_own"]]

test_features_df.describe()

test_features_df.info()

test_features_df.head()

test_features_df.tail()

test_features_df.sample(5)

**training_set_labels.csv**

train_labels_df.info()

train_labels_df.describe()

train_labels_df.info()

train_labels_df.head()

train_labels_df.tail()

train_labels_df.sample(5)

# Data Cleaning

## Correct formats

**training_set_features.csv**

train_df.info()

Formats are as expected

**testing_set_features.csv**

test_features_df.info()

Formats are as expected

**training_set_labels.csv**

train_labels_df.info()

Formats are as expected

### Missing Values

The follow are missing values: ```education, income_poverty, employment_status marital status and rent_or_own```. Checking values

train_df.isna().sum()

missing = ["education", "income_poverty", "employment_status", "marital_status"
           ,"rent_or_own","household_adults","household_children"]
for col in missing:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

train_df.isna().sum()

test_features_df.isna().sum()

The follow are missing values: ```education, income_poverty, employment_status marital status and rent_or_own```. Checking values

missing = [ "education", "income_poverty", "employment_status", "marital_status", "rent_or_own"]
 
for col in missing:
    print(f"{test_features_df[col].value_counts()}\n")

Replacing missing values with mode since many are categorical columns

for col in missing:
    test_features_df[col].fillna(test_features_df[col].mode(), inplace=True)

test_features_df.isna().sum()

train_labels_df.isna().sum()

### Changing Columns

train_df.columns

test_features_df.columns

train_labels_df.columns

Columns are in the desired format

### Checking Duplicates

#Checking there duplicates in training set 
print(train_df.shape[0])
train_features_df[col].duplicated().sum()

train_features_df[col].drop_duplicates(inplace=True)
print(train_features_df.shape[0])

#Checking there duplicates in testing set
#not adding respondent id since there would be no duplicates
col = ["age_group", "education", "income_poverty", "employment_status",
    "race", "sex", "marital_status", "rent_or_own"]
test_features_df = test_features_df[col]
print(test_features_df.shape[0])
test_features_df.duplicated().sum()

test_features_df.drop_duplicates(subset=col,inplace=True)
print(test_features_df.shape[0])

### Feature engineering



### Checking Outliers

numeric_df =train_df.select_dtypes(include = ['number'])

#calculate the number of fig to fit height 
grid=(numeric_df.shape[1]+1)//2 
#allocating each plot a height of 5
plt.figure(figsize=(12, grid * 5))

count=0
for col in numeric_df:
    count += 1 
    plt.subplot(grid,2,count)
    sns.boxplot(y=train_df[col]) 

### Saving Dataset

train_df.to_csv("train_clean.csv")

# Explanatory Analysis

## Univariate Analysis

age_count =  train_df['age_group'].value_counts()
age_count.plot(kind="bar")
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('Age Group')
plt.title("Number of Respondents to Age Group")
plt.show()

**65+ years** are the most respondents while **35-44 Years** are the least

education_count =  train_df['education'].value_counts()
sns.countplot(x=train_df['education'],order=education_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('Education level')
plt.title("Number of Respondents to Education Level")
plt.show()

The survey had many **College Graduate** than other level.With the least be **<12 Years**

income_poverty_count =  train_df['income_poverty'].value_counts()
sns.countplot(x=train_df['income_poverty'],order=income_poverty_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('Income Poverty level')
plt.title("Number of Respondents to Income Poverty Level")
plt.show()

The survey had many **Above Poverty** respondents than other level.With the least be **Below Poverty**

employment_status_count =  train_df['employment_status'].value_counts()
sns.countplot(x=train_df['employment_status'],order=employment_status_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('Employment Status level')
plt.title("Number of Respondents to Employment Status Level")
plt.show()

The survey had many **Employed** respondents than other level.With the least be **Unemployed**

race_count =  train_df['race'].value_counts()
sns.countplot(x=train_df['race'],order=race_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('Race')
plt.title("Number of Respondents to Race")
plt.show()

The survey had many **White** respondents than other level.With the least be **Other or Multiple**

sex_count =  train_df['sex'].value_counts()
sns.countplot(x=train_df['sex'],order=sex_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('Sex')
plt.title("Number of Respondents to Sex")
plt.show()

The survey had many **Female** respondents.With the least be **Male**

marital_status_count =  train_df['marital_status'].value_counts()
sns.countplot(x=train_df['marital_status'],order=marital_status_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('marital_status')
plt.title("Number of Respondents to marital_status")
plt.show()

The survey had many **Married** respondents than other level.With the least be **Not Married**

rent_or_own_count =  train_df['rent_or_own'].value_counts()
sns.countplot(x=train_df['rent_or_own'],order=rent_or_own_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('rent_or_own')
plt.title("Number of Respondents to rent_or_own")
plt.show()

The survey had many **Own** respondents than other level.With the least be **Rent**

seasonal_vaccine_count =  train_df['seasonal_vaccine'].value_counts()
sns.countplot(x=train_df['seasonal_vaccine'],order=seasonal_vaccine_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('seasonal_vaccine')
plt.title("Number of Respondents to seasonal_vaccine")
plt.show()

The survey had many **not vaccinated** respondents for seasonal vaccine.With the least be **vaccinated**

h1n1_vaccine_count =  train_df['h1n1_vaccine'].value_counts()
sns.countplot(x=train_df['h1n1_vaccine'],order=h1n1_vaccine_count.index,palette='Blues_r')
plt.ylabel('No of Respondents')
plt.xticks(rotation=45)
plt.xlabel('h1n1_vaccine')
plt.title("Number of Respondents to h1n1_vaccine")
plt.show()

The survey had many **not vaccinated** respondents for h1n1 vaccine.With the least be **vaccinated**

## Bivariate Analysis

selected_features = ["age_group", "education", "income_poverty", "employment_status",
    "race", "sex", "marital_status", "rent_or_own","household_adults", "household_children","hhs_geo_region", "census_msa" ]

fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))  
 
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    # creating pivot table to aggregate the feature by seasonal_vaccine
    feature_seasonal_vaccine = train_df.pivot_table(index=feature, values='seasonal_vaccine', aggfunc='sum')
    
   
    sns.barplot(x=feature_seasonal_vaccine.index, 
                y=feature_seasonal_vaccine['seasonal_vaccine'], color='#2a5783', 
                ci=None, ax=axes[i])
     
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Seasonal Vaccine')
    axes[i].set_title(f'Comparison of Seasonal Vaccine to {feature}')
    axes[i].tick_params(axis='x', rotation=45)  
    
plt.tight_layout()
plt.show()


1. Took most seasonal vaccines compared to other age groups.
- **65+ Years** 
- **College Graduate**
- **Female**
- **White**
- **Own House**
- **Married**
- **<=75000 Above Property**
2. Took least seasonal vaccines compared to other age groups.
- **35 - 44 Years** 
- **<12 Years** of education
- **Male**
- **Black,Hispanic and Other**
- **Rent House**
- **Not Married**
- **Below Property**

 # Create subplots, setting the number of rows and columns
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    # Filter the train_df to only include rows where seasonal_vaccine == 0
    feature_seasonal_vaccine_0 = train_df[train_df['seasonal_vaccine'] == 0]

    # Creating pivot table to aggregate the feature by seasonal_vaccine == 0
    feature_seasonal_vaccine = feature_seasonal_vaccine_0.pivot_table(index=feature, values='seasonal_vaccine', aggfunc='count')
    
    # Plotting data
    sns.barplot(x=feature_seasonal_vaccine.index, 
                y=feature_seasonal_vaccine['seasonal_vaccine'], color='#2a5783', 
                ci=None, ax=axes[i])
    
    # Set labels and titles for each plot
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Non-Vaccinated Respondents')
    axes[i].set_title(f'Comparison of Non-Vaccinated Respondents to {feature}')
    axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels if needed

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

1. Most Non-vaccinated seasonal vaccines compared to other age groups.
- **18 - 24 Years** 
- **College Graduate**
- **Female**
- **White**
- **Employed**
- **Own House**
- **Married**
- **>=75000 Above Property** 

## Multivariate Analysis

fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))  
 
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    # creating pivot table to aggregate the feature by seasonal_vaccine
   
    sns.countplot(x=train_df[feature],hue=train_df['seasonal_vaccine'],palette='Blues_r',
                  ax=axes[i])
     
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Seasonal Vaccine')
    axes[i].set_title(f'Comparison of Seasonal Vaccine to {feature}')
    axes[i].tick_params(axis='x', rotation=45)  
    
plt.tight_layout()
plt.show()

1. All age groups except 65 Years and 55 - 64 Years which have more are vaccinated than non vaccinated.
2. All education level except College Graduate which have more are vaccinated than non vaccinated.
3. All income levels which have more are non-vaccinated than vaccinated.
4. All employment status except not in labor force which have more are vaccinated than non vaccinated.
5. All sex, marital status and rent status which have more are non-vaccinated than vaccinated.

 # Filter dataset where seasonal_vaccine was given
df_vaccinated = train_df[train_df['seasonal_vaccine'] == 1]

# Create subplots dynamically based on the number of features
n_cols = 2  # Number of columns in the subplot
n_rows = (len(selected_features) + n_cols - 1) // n_cols  # Calculate required rows dynamically

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    sns.countplot(x=df_vaccinated[feature], hue=df_vaccinated['age_group'], 
                  palette='Blues_r', ax=axes[i])
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Seasonal Vaccine Recipients')
    axes[i].set_title(f'Comparison of Seasonal Vaccine Recipients by {feature}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots
for j in range(len(col), len(axes)):  
    fig.delaxes(axes[j])  

# Adjust layout and display
plt.tight_layout()
plt.show()


# Filter dataset where seasonal_vaccine was given
df_vaccinated = train_df[train_df['seasonal_vaccine'] == 1]

# Create subplots dynamically based on the number of features
n_cols = 2  # Number of columns in the subplot
n_rows = (len(col) + n_cols - 1) // n_cols  # Calculate required rows dynamically

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    sns.countplot(x=df_vaccinated[feature], hue=df_vaccinated["income_poverty"], 
                  palette='Blues_r', ax=axes[i])
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Seasonal Vaccine Recipients')
    axes[i].set_title(f'Comparison of Seasonal Vaccine Recipients by {feature}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots
for j in range(len(col), len(axes)):  
    fig.delaxes(axes[j])  

# Adjust layout and display
plt.tight_layout()
plt.show()

# Filter dataset where seasonal_vaccine was given
df_vaccinated = train_df[train_df['seasonal_vaccine'] == 1]

# Create subplots dynamically based on the number of features
n_cols = 2  # Number of columns in the subplot
n_rows = (len(selected_features) + n_cols - 1) // n_cols  # Calculate required rows dynamically

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    sns.countplot(x=df_vaccinated[feature], hue=df_vaccinated["education"], 
                  palette='Blues_r', ax=axes[i])
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Seasonal Vaccine Recipients')
    axes[i].set_title(f'Comparison of Seasonal Vaccine Recipients by {feature}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots
for j in range(len(col), len(axes)):  
    fig.delaxes(axes[j])  

# Adjust layout and display
plt.tight_layout()
plt.show()


# Filter dataset where seasonal_vaccine was given
df_vaccinated = train_df[train_df['seasonal_vaccine'] == 1]

# Create subplots dynamically based on the number of features
n_cols = 2  # Number of columns in the subplot
n_rows = (len(selected_features) + n_cols - 1) // n_cols  # Calculate required rows dynamically

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Loop through each column in the list
for i, feature in enumerate(selected_features):
    sns.countplot(x=df_vaccinated[feature], hue=df_vaccinated["race"], 
                  palette='Blues_r', ax=axes[i])
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Number of Seasonal Vaccine Recipients')
    axes[i].set_title(f'Comparison of Seasonal Vaccine Recipients by {feature}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots
for j in range(len(col), len(axes)):  
    fig.delaxes(axes[j])  

# Adjust layout and display
plt.tight_layout()
plt.show()


# Preprocessing

# columns to encode
categorical_columns = ["age_group", "education", "income_poverty", "employment_status",
                       "race", "sex", "marital_status", "rent_or_own","hhs_geo_region", "census_msa"]

label_encoders = {}

# encoding to each categorical column
for col in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(train_df[col])
    train_df[col] = encoder.transform(train_df[col]) 
    label_encoders[col] = encoder 

train_df.head()


X = train_df.drop(columns=["seasonal_vaccine","h1n1_vaccine"])
y = train_df.seasonal_vaccine
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.8, random_state = 42)

# Modeling

# train logreg model   
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

model = LogisticRegression()

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Use the best model to predict
logreg  = grid_search.best_estimator_ 

from sklearn.model_selection import cross_validate
cross_validate(logreg, X, y, cv=10)

cross_validate(logreg, X, y, return_train_score=True)

# train dt model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV 


# train rf model 
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train) 

# train svc model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)


model = SVC()

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Use the best model to predict
best_model = grid_search.best_estimator_

# naive bayes classifier
nb_classifier = MultinomialNB()
# Train the model
nb_classifier.fit(X_train, y_train)


# Evaluation

# logreg model
y_predict = logreg.predict(X_test)
# evaluate the logreg model
logreg_accuracy = accuracy_score(y_test,y_predict)

# dt model
y_pred = dt_model.predict(X_test)
# Evaluate the dt model
dt_model_accuracy = accuracy_score(y_test, y_pred) 

#rf model
y_pred = rf_model.predict(X_test)
# evaluate the rf model
rf_model_accuracy = accuracy_score(y_test, y_pred)

# svm model
y_pred = svm_model.predict(X_test)
# evaluate the svm model
svm_model_accuracy = accuracy_score(y_test, y_pred)
 
# naive bayes classifier
y_pred = nb_classifier.predict(X_test)
# evaluate the svm model
nb_classifier_accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy = ", logreg_accuracy ) 
print("Decision Tree Classifier Accuracy = ", dt_model_accuracy)
print("Random Forest Accuracy = ",  rf_model_accuracy)
print("SVM Accuracy = ", svm_model_accuracy)
print("Naive Bayes Classifier = ", nb_classifier_accuracy)

# Conclusion

 
- **Logistic Regression**: 0.63 Accuracy
- **Decision Tree Classifier**: 0.59 Accuracy
- **Random Forest**: 0.60 Accuracy
- **Support Vector Machine (SVM)**: 0.62 Accuracy
- **Naive Bayes**: 0.62 Accuracy
Logistic Regression achieved the highest accuracy at **0.63**, making it the best-performing model. Decision Tree Classifier had the lowest accuracy at **0.59**. Random Forest, Naive Bayes and SVM performed similarly with accuracies of **0.60**, **0.62** and **0.62**, respectively.
