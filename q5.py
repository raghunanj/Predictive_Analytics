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
data_needed17 = data_input17[['FG%','ORB','DRB','TRB','PTS']]
data_needed15 = data_input16[['FG%','ORB','DRB','TRB','PTS']]
data_needed14 = data_input16[['FG%','ORB','DRB','TRB','PTS']]
data_needed13 = data_input16[['FG%','ORB','DRB','TRB','PTS']]
data_needed12 = data_input16[['FG%','ORB','DRB','TRB','PTS']]
data_needed11 = data_input16[['FG%','ORB','DRB','TRB','PTS']]
data_needed10 = data_input16[['FG%','ORB','DRB','TRB','PTS']]


def Beta_hatFunction(X,Y):
	var_x = X.T.dot(X)
	var_y = X.T.dot(Y)
	var_x = np.linalg.inv(var_x)
	beta_hat = var_x.dot(var_y)
	return(beta_hat)

def SSE(f, f_hat):
	error = 0
	for i in range(len(f)):
		error += np.square(f[i]-f_hat[i])
	return (error)

def MAPE (f, f_hat):
	var = 0
	for i in range(len(f)):
		var += abs((f[i]-f_hat[i])/f[i])
	var = (100/len(f))*var
	return (var)

#---------------5a-----------------------------------------------------------------	
x_axis = np.array(data_needed16[['FG%','TRB']])
y_axis = np.array(data_needed16[['PTS']])
Beta_hat = Beta_hatFunction(x_axis,y_axis)

print('Beta_hat for 5a:',Beta_hat)


#######Inference#########
# TRB contribution of 1.006 is less to PTS when compared to FG% which is 129.63 
#########################


#-----------------5b--------------------------------------------------------------
x_axis = np.array(data_needed16[['FG%','TRB','ORB','DRB']])
y_axis = np.array(data_needed16[['PTS']])
Beta_hat = Beta_hatFunction(x_axis,y_axis)

print('Beta_hat for 5b:',Beta_hat)

#######Inference#########
# TRB contribution is negative in this case. This is caused by the Multicollinearity, 
# Multicollinearity occurs in this case because we clearly see that ORB + DRB = TRB. 
#########################



#-----------------5c--------------------------------------------------------------
x_axis = np.array(data_needed16[['FG%','TRB']])
dump = np.ones(x_axis.shape[0]).reshape(-1,1)
x_axis = np.hstack((dump,x_axis))
y_axis = np.array(data_needed16[['PTS']])
Beta_hat = Beta_hatFunction(x_axis, y_axis)
print('Beta_hat for 5c(i)',Beta_hat)




x_test = np.array(data_needed17[['FG%','TRB']])
dump = np.ones(x_test.shape[0]).reshape(-1,1)
x_test = np.hstack((dump,x_test))
main_result = x_test.dot(Beta_hat)



expected = np.array(data_needed17[['PTS']])
sse = SSE(expected, main_result)
print("SSE of 5c(i):",sse)

mape = MAPE(expected, main_result)
print("MAPE of 5c(i):",mape)



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