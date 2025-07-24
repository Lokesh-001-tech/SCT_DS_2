
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/lokes/Downloads/gender_submission.csv")
print("First 5 rows of the dataset:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values:\n", df.isnull().sum())

if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

if 'Embarked' in df.columns:
    df.dropna(subset=['Embarked'], inplace=True)

if 'Survived' in df.columns:
    survived_counts = df['Survived'].value_counts()
    survived_counts.plot(kind='bar', color=['red', 'green'])
    plt.title('Survival Count')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Number of Passengers')
    plt.xticks(rotation=0)
    plt.show()

if 'Sex' in df.columns and 'Survived' in df.columns:
    gender_survival = df.groupby('Sex')['Survived'].value_counts().unstack()
    gender_survival.plot(kind='bar', stacked=True)
    plt.title('Survival by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Survived')
    plt.show()

if 'Age' in df.columns and 'Survived' in df.columns:
    df[df['Survived'] == 0]['Age'].plot.hist(alpha=0.5, label='Not Survived', bins=20)
    df[df['Survived'] == 1]['Age'].plot.hist(alpha=0.5, label='Survived', bins=20)
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.legend()
    plt.show()

if 'Fare' in df.columns and 'Pclass' in df.columns:
    df.boxplot(column='Fare', by='Pclass')
    plt.title('Fare Distribution by Passenger Class')
    plt.suptitle('')  # Remove automatic title
    plt.xlabel('Passenger Class')
    plt.ylabel('F6are')
    plt.show()

import numpy as np

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap='coolwarm')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
plt.title("Correlation Heatmap", pad=20)
plt.show()
