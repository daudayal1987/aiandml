"""house_3d_model.ipynb

## Write a python program to draw the 3d plot for the model developed for house price prediction using suitable python based 3d plotting libraries

### import required libraries
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""### Load CSV data in pandas dataframe with column names as CSV file do not have column names"""

column_names = ["square_feet", "no_of_bedrooms", "price"]
df = pd.read_csv("https://raw.githubusercontent.com/daudayal1987/aiandml/refs/heads/main/home_regression.csv", header=None, names=column_names)

"""### Display few dataset items"""

df.head()

"""### Describe dataset for min, max, count, std and other stats using describe function"""

df.describe()

"""### Draw 3D plot using matplotlib"""

x_data = df['square_feet']
y_data = df['no_of_bedrooms']
z_data = df['price']

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x_data, y_data, z_data)
ax.set_xlabel("Square Feet")
ax.set_ylabel("No of Bedrooms")
ax.set_zlabel("Price")
plt.title("3D Model for house price using Sqft and Bedrooms")
plt.show()