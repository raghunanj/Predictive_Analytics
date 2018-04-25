import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data_input16 = pd.read_csv("data_2016.csv")
data_input17 = pd.read_csv("data_2017.csv")
data_input15 = pd.read_csv("data_2015.csv")
data_input14 = pd.read_csv("data_2014.csv")
data_input13 = pd.read_csv("data_2013.csv")
data_input12 = pd.read_csv("data_2012.csv")
data_input11 = pd.read_csv("data_2011.csv")
data_input10 = pd.read_csv("data_2010.csv")

data_needed16 = data_input16[['FG%','ORB','DRB','TRB','PTS']]


def Beta_hatFunction(X,Y):
	var_x = X.T.dot(X)
    var_y = X.T.dot(Y)
    var_x = np.linalg.inv(var_x)
    beta_hat = var_x.dot(var_y)
    return(beta_hat)

x_axis = np.array(data_needed16[['FG%','TRB']])
y_axis = np.array(data_needed16[['PTS']])
Beta_hat = Beta_hatFunction(x_axis,y_axis)

print('Beta_hat:',Beta_hat)

'''
regr = linear_model.LinearRegression()

regr.fit(x_axis, y_axis)

m = regr.coef_[0]
b = regr.intercept_

print(' y = {0} * x + {1}'.format(m, b))



#print (data_input16)


plt.scatter(x_axis, y_axis, color='blue')  # you can use test_data_X and test_data_Y instead.
plt.plot([min(x_axis), max(x_axis)], [b, m*max(x_axis) + b], 'r')
plt.title('Fitted linear regression', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)

'''