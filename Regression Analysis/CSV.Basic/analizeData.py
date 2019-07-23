import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as seaborninstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('/Users/kalharaperera/Desktop/Projects/Data Sets/us-weather-history/KCLT.csv')

# print(dataset.shape)
print(dataset.describe())

dataset.plot(x='actual_min_temp',y='actual_max_temp',style = 'o')
plt.title('Min temp Vs Max temp')
plt.xlabel('Min temp')
plt.ylabel('Max temp')
plt.show()

plt.figure(figsize=(15,10))
# plt.tight_layout()
seaborninstance.distplot(dataset['actual_max_temp'])
plt.show()

#data splicing basically spliting your data into training data and testing data
X = dataset['actual_min_temp'].values.reshape(-1,1)
Y = dataset['actual_max_temp'].values.reshape(-1,1)


#assigning 20% of the data to test data and the others to training data
X_train , X_test,Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


regressor = LinearRegression()
#training the algorithem
regressor.fit(X_train, Y_train)

#to receive the intercept
print('Intercept',regressor.intercept_)

#for every single change in value in x (minimum temparature)causes coefficient times of change in y values
#thing with Beta
print('Coefficient' , regressor.coef_)


#passing data to traing the model and to make the prediction
y_prediction = regressor.predict(X_test)

#storing the data in a dataframe and printing the dataframe
df = pd.DataFrame({'Actual' : Y_test.flatten(),'Predicted': y_prediction.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')

plt.scatter(X_test,Y_test,color='grey')
# plt.plot(X_test,y_prediction,color='red',linewidth=2)
plt.title('check')
plt.show()

# added
dataset.plot(x='actual_min_temp',y='actual_max_temp',style = 'o')
plt.title('Min temp Vs Max temp')
plt.xlabel('Min temp')
plt.ylabel('Max temp')
plt.plot(X_test,y_prediction,color='red',linewidth=2)
plt.show()

print('Mean absolute error :' , metrics.mean_absolute_error(Y_test,y_prediction))
print('Mean squared error :' , metrics.mean_squared_error(Y_test,y_prediction))
print('Root Mean absolute error :' , np.sqrt(metrics.mean_squared_error(Y_test,y_prediction)))


