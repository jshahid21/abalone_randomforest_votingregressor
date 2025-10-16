# Step 1 - Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import tree

# Step 2 - Read data into a dataframe
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_csv('/content/abalone.data', names=names, sep=',')

df.head()

df = df.drop('Sex', axis=1)

df.columns

# Step 4 - Define X and y
X = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight']]
y = df['Rings']

# Step 5 - Split X and y into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

#Define object of RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100, random_state=42)

#fit the algorithm to X_train,y_train
clf.fit(X_train, y_train)

fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=800)
tree.plot_tree(clf.estimators_[0],feature_names=X.columns,filled=True, max_depth=3)

fig,axes = plt.subplots(nrows=1,ncols=5,figsize=(10,2),dpi=900)
for i in range(0,5):
    tree.plot_tree(clf.estimators_[i],feature_names=X.columns,filled=True,ax=axes[i], max_depth=3)

clf.score(X_train,y_train)

# Evaluate the RandomForestRegressor on the test set
test_score = clf.score(X_test, y_test)
print(f"R-squared score on the test set: {test_score}")

!pip install dtreeviz

import dtreeviz

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)

from sklearn.tree import DecisionTreeRegressor
import dtreeviz

# Train a new Decision Tree Regressor with max_depth=3
dt_simplified = DecisionTreeRegressor(max_depth=3)
dt_simplified.fit(X_train, y_train)

# Visualize the simplified tree using dtreeviz
viz_model_simplified = dtreeviz.model(dt_simplified,
                                      X_train,
                                      y_train,
                                      feature_names=X.columns,
                                      target_name='Rings')

viz_model_simplified.view(fontname="monospace")

#import all the required libraries for voting classifier to use with abalone data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression # Import Linear Regression
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor is already imported, but good to be explicit
from sklearn.svm import SVR # Import Support Vector Regressor

# Read abalone data into data frame
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_csv('/content/abalone.data', names=names, sep=',')

# Preprocess data
df = df.drop('Sex', axis=1)

df.columns

# Define X and y
X = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight']]
y = df['Rings']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train

y_train

from sklearn.linear_model import LinearRegression # Import Linear Regression
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor is already imported, but good to be explicit
from sklearn.svm import SVR # Import Support Vector Regressor

algos = []
algos.append(('lr',LinearRegression())) # Replaced LogisticRegression with LinearRegression
algos.append(('rf',RandomForestRegressor())) # Replaced RandomForestClassifier with RandomForestRegressor
algos.append(('svr',SVR())) # Replaced SVC with SVR
algos

voting = VotingRegressor(algos)
voting.fit(X_train,y_train)

voting.score(X_train,y_train)

# Evaluate the VotingRegressor on the test set
voting_test_score = voting.score(X_test, y_test)
print(f"R-squared score on the test set for Voting Regressor: {voting_test_score}")

# Show the performance of individual regression models
for name, model in algos:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"R-squared score for {name}: {score}")

# Predict using the VotingRegressor
# The input should be a 2D array-like structure with 7 features (matching X)
y_pred = voting.predict([[0.5, 0.4, 0.15, 0.7, 0.3, 0.15, 0.2]])
y_pred

for name, model in algos: # Use the 'model' variable from the loop
    model.fit(X_train,y_train)
    y_train_pred = model.predict([[0.5, 0.4, 0.15, 0.7, 0.3, 0.15, 0.2]]) # Corrected from clf.predict to model.predict
    print(f"Prediction from {name}: {y_train_pred}") # Added name for clarity

voting = VotingRegressor(algos)
voting.fit(X_train,y_train)
