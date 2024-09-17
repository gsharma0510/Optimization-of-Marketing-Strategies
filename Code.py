#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import pygwalker as pyg 

# Load data
data = pd.read_csv(r'C:\PDS Project/Analysis.csv', sep=';')

# Binning ages
bins = list(range(15, 100, 5))  # Bins from 15 to 100 with step 5
data['age_group'] = pd.cut(data['age'], bins=bins, right=False, include_lowest=True)

# Plot the number of calls by age group
age_group_counts = data['age_group'].value_counts().sort_index()
plt.figure(figsize=(9, 5))
age_group_counts.plot(kind='bar', width=0.8)
plt.xlabel('Age Group')
plt.ylabel('Number of Calls')
plt.title('Number of Calls in Each Age Group')
plt.xticks(rotation=30, ha='right')
plt.show()

# Data for conversions
data_age = data[data['y'] == 'yes']
data_age['age_group'] = pd.cut(data_age['age'], bins=bins, right=False, include_lowest=True)
age_group_counts = data_age['age_group'].value_counts().sort_index()

# Plot the number of conversions by age group
plt.figure(figsize=(9, 5))
age_group_counts.plot(kind='bar', width=0.8, color='g')
plt.xlabel('Age Group')
plt.ylabel('Number of Conversions')
plt.title('Conversion of Calls in Each Age Group')
plt.xticks(rotation=30, ha='right')
plt.show()

# Job analysis
job_graph = data['job'].value_counts().to_frame().reset_index()
plt.figure(figsize=(9, 5))
plt.bar(job_graph['job'], job_graph['count'], label='Number of Calls to Job Holders')
plt.xlabel('Job Type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Calls')
plt.title('Number of Calls to Job Holders')
plt.legend()
plt.show()

# Conversions by job
dj = data[data['y'] == 'yes']
job_graph_yes = dj['job'].value_counts().to_frame().reset_index()
plt.figure(figsize=(9, 5))
plt.bar(job_graph_yes['job'], job_graph_yes['count'], label='Conversions from Job Holders', color='g')
plt.xlabel('Job Type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Conversions')
plt.title('Number of Conversions from Job Holders')
plt.legend()
plt.show()

# Hue encoding by job
plt.figure(figsize=(9, 5))
sns.countplot(x='job', hue='y', data=data, palette='Set2')
plt.xlabel('Job Category')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.title('Count of Customers by Job Category with Hue Encoding for Joining')
plt.show()

# Stacked bar for job conversion percentages
percentage_df = data.groupby(['job', 'y']).size().unstack().div(data.groupby('job').size(), axis=0) * 100
plt.figure(figsize=(10, 6))
percentage_df.plot(kind='bar', stacked=True, color=['#001F3F', 'g'], edgecolor='black')
plt.xlabel('Job Category')
plt.ylabel('Percentage')
plt.title('Percentage of Conversion by Job Category')
plt.legend(title='Joining (y)', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()

# Education analysis
count_df = data.groupby(['education', 'y']).size().unstack().fillna(0)
plt.figure(figsize=(9, 5))
count_df.plot(kind='bar', stacked=True, color=['#3498db', '#2ecc71'])
plt.xlabel('Education Category')
plt.ylabel('Count')
plt.title('Count of Calls and Subscribers by Education Category')
plt.legend(title='Joining (y)', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.show()

# Pie chart of 'y' counts
count_df = data['y'].value_counts()
explode = [0.1, 0]  # Exploding the first slice
colors = ['#3498db', '#2ecc71']
plt.figure(figsize=(5, 5))
plt.pie(count_df.values, explode=explode, labels=count_df.index, colors=colors, autopct='%1.2f%%', startangle=90, shadow=True)
plt.title('Pie Chart for "Yes" and "No" Counts')
plt.show()

# Box plot of bank balance by 'y'
plt.figure(figsize=(8, 6))
sns.boxplot(x='y', y='balance', data=data, palette=['#1f78b4', '#33a02c'])
plt.title('Box Plot of Bank Balance by Y')
plt.show()

# 3D bar chart for 'y' counts
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.bar(count_df.index, count_df.values, width=0.8, color=['#3498db', '#2ecc71'], alpha=0.8)
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_title('3D Bar Chart for "Yes" and "No" Counts')
plt.show()

# Heatmap for balance vs. 'y'
df = data[data['balance'] >= 0]
plt.figure(figsize=(8, 4))
heatmap_data = df.groupby(['y', pd.cut(df['balance'], bins=15)]).size().unstack().fillna(0)
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='viridis')
plt.xlabel('Balance Bins')
plt.ylabel('Y')
plt.title('Heatmap of Balance with Respect to Y')
plt.show()

# Pygwalker Dashboard
gwalker = pyg.walk(data)

# Logistic Regression Model
X = data.drop(columns=['y'])
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# One-hot encoding
onehot = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day', 'poutcome']
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, axis=1)
        df = pd.concat([df, dummies], axis=1)
    return df

X_train_one = dummy_df(X_train, onehot)
X_test_one = dummy_df(X_test, onehot)

# Logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train_one, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test_one)
print(metrics.accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"]), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SMOTE oversampling
oversample = SMOTE()
X_train_smote, y_train_smote = oversample.fit_resample(X_train_one, y_train)

# Re-train logistic regression with SMOTE data
clf.fit(X_train_smote, y_train_smote)

# Prediction and evaluation after SMOTE
y_pred_smote = clf.predict(X_test_one)
print(metrics.accuracy_score(y_test, y_pred_smote))
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(cm_smote)
print(classification_report(y_test, y_pred_smote))

# Plot confusion matrix after SMOTE
plt.figure(figsize=(6, 6))
sns.heatmap(pd.DataFrame(cm_smote, index=["No", "Yes"], columns=["No", "Yes"]), annot=True, fmt="d")
plt.title("Confusion Matrix After SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
