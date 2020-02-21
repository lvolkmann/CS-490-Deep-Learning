# @author: Landon Volkmann

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')

# QUESTION 1
# Clean outlier garage data
garage_df = train.get('GarageArea')
print(garage_df)
z = np.abs(stats.zscore(garage_df))
print(z)
outliers = []
pos_to_drop = []
for i, score in enumerate(z):
    if score > 3:
        outliers.append((i, score))
        pos_to_drop.append(i)
        print(i, score)
print(outliers)

plt.scatter(train.get('GarageArea'), train.get('SalePrice'), alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.title('Dirty')
plt.show()


print(train)
train.drop(train.index[pos_to_drop], inplace=True)
print(train)

plt.scatter(train.get('GarageArea'), train.get('SalePrice'), alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.title('Clean')
plt.show()
