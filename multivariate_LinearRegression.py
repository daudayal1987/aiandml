# -*- coding: utf-8 -*-
"""MultiVariate_LinearRegression.ipynb

# Build a multivariate linear regression model that can predict the product sales based on the advertising budget allocated to different channels. The features are TV Budget, Radio Budget, Newspaper Budget and the output is Sales (units)

### Step1: Import required libraries and frameworks
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, root_mean_squared_error as rmse, mean_absolute_error as mae
import matplotlib.pyplot as plt

"""### Step2: Load CSV data and show data stastistics

#### Step 2.1: Load CSV
"""

df = pd.read_csv("budget_and_sales.csv")

"""#### Step 2.2: Show part of data using panda's head command"""

df.head()

"""#### Step 2.3: Describe (count, mean, std, min, max, 25%, 50% and 75%) data"""

df.describe()

"""### Step 3: Prepare data for training and testing"""

features = df[["TV Budget ($)", "Radio Budget ($)", "Newspaper Budget ($)"]]
output = df["Sales (units)"]

features_train, features_test, output_train, output_test = train_test_split(features, output, test_size=.3, random_state=7)

"""### Step 4: Fit model in data"""

model = LinearRegression()
model.fit(features_train, output_train)

"""### Step 5: Analyze model"""

output_predicted = model.predict(features_test)
analyze_df = pd.DataFrame(features_test)
analyze_df["actual_output"] = output_test
analyze_df["predicted_output"] = output_predicted

analyze_df.head()

"""### Step 6: Check regression metrics

#### Step 6.1: Check Mean Squared Error (MSE)
"""

mse(output_test, output_predicted)

"""#### Step 6.2: Check Root Mean Squared Error (RMSE)"""

rmse(output_test, output_predicted)

"""#### Step 6.3: Check Mean Absolute Error (MAE)"""

mae(output_test, output_predicted)

"""### Step 7: Draw pyplot for all features vs output"""

for column in df.columns[:-1]:
  plt.figure(figsize=(4,3))
  plt.scatter(df[column], df.iloc[:,-1])
  plt.title(f'Feature ${column} vs output ${df.columns[-1]}')
  plt.xlabel(column)
  plt.ylabel(df.columns[-1])
  plt.grid(True)
  plt.show()

