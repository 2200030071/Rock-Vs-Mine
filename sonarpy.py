import matplotlib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from pickle import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set display options for Pandas
pd.set_option('display.width', 100)
pd.set_option('display.precision', 5)

# Load dataset
filename = 'sonar.csv'
df = pd.read_csv(filename, header=None)

# Display dataset info
df.info()

# Check for missing values
print(f"Missing values: {df.isnull().values.any()}")

# Encode 'M' as 1 and 'R' as 0
df[60] = df[60].replace({'M': 1, 'R': 0}).astype(int)

# Plot histograms and density plots
df.hist(figsize=(16, 12))
df.plot(kind='density', subplots=True, layout=(8, 8), sharex=False, figsize=(18, 14))

# Correlation matrix
correlations = df.corr()
fig = plt.figure(figsize=(18, 16))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, 60, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()

# Prepare training and testing datasets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Model evaluation function
def eval_algorithms(models, show_boxplots=True):
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=False)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean() * 100.0:.2f}% ({cv_results.std() * 100.0:.2f}%)")

    if show_boxplots:
        fig = plt.figure(figsize=(14, 12))
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

# Define models
models = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=200)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='scale'))
]

# Evaluate models
eval_algorithms(models)

# Standardized models
pipelines = [
    ('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression(solver='lbfgs', max_iter=200))])),
    ('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])),
    ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])),
    ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])),
    ('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])),
    ('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC(gamma='scale'))]))
]

eval_algorithms(pipelines)

# KNN hyperparameter tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', cv=KFold(n_splits=10, shuffle=False))
grid_result = grid.fit(rescaledX, Y_train)

print(f"Best KNN model: {grid_result.best_score_:.2f} using {grid_result.best_params_}")

# SVM hyperparameter tuning
param_grid = {'C': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid = GridSearchCV(SVC(gamma='auto'), param_grid, scoring='accuracy', cv=KFold(n_splits=10, shuffle=False))
grid_result = grid.fit(rescaledX, Y_train)

print(f"Best SVM model: {grid_result.best_score_:.2f} using {grid_result.best_params_}")

# Ensemble methods
ensembles = [
    ('AB', AdaBoostClassifier(algorithm='SAMME')),
    ('GBM', GradientBoostingClassifier()),
    ('RF', RandomForestClassifier(n_estimators=100)),
    ('ET', ExtraTreesClassifier(n_estimators=100))
]

eval_algorithms(ensembles)

# Train final SVM model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
clf = SVC(C=1.5, kernel='rbf', gamma='auto')
clf.fit(rescaledX, Y_train)

# Evaluate on test data
rescaledTestX = scaler.transform(X_test)
pred = clf.predict(rescaledTestX)

print(f'Accuracy Score: {accuracy_score(Y_test, pred):.2f}')
print(f'\nConfusion Matrix:\n{confusion_matrix(Y_test, pred)}')
print(f'\nClassification Report:\n{classification_report(Y_test, pred)}')

# Save the model
filename = 'finalized_model.sav'
dump(clf, open(filename, 'wb'))

# Load and test saved model
loaded_clf = load(open(filename, 'rb'))
print(f"Loaded Model Accuracy: {loaded_clf.score(rescaledTestX, Y_test):.2f}")
