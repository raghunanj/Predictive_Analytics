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
data_needed15 = data_input15[['FG%','ORB','DRB','TRB','PTS']]
data_needed14 = data_input14[['FG%','ORB','DRB','TRB','PTS']]
data_needed13 = data_input13[['FG%','ORB','DRB','TRB','PTS']]
data_needed12 = data_input12[['FG%','ORB','DRB','TRB','PTS']]
data_needed11 = data_input11[['FG%','ORB','DRB','TRB','PTS']]
data_needed10 = data_input10[['FG%','ORB','DRB','TRB','PTS']]


def Beta_hatFunction(X,Y):
	var_x = X.T.dot(X)
	var_y = X.T.dot(Y)
	var_x = np.linalg.inv(var_x)
	beta_hat = var_x.dot(var_y)
	return(beta_hat)

def ResidualFn(f, f_hat):
	return (f-f_hat)

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
main_result = 0
main_result2 = 0
main_result3 = 0


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


x_axis1 = np.array(data_needed15[['FG%','TRB']])
x_axis2 = np.array(data_needed16[['FG%','TRB']])
x_axis1 = np.vstack((x_axis1, x_axis2))
dump = np.ones(x_axis1.shape[0]).reshape(-1,1)
x_axis1 = np.hstack((dump,x_axis1))

y_axis1 = np.array(data_needed15[['PTS']])
y_axis2 = np.array(data_needed16[['PTS']])
y_axis1 = np.vstack((y_axis1, y_axis2))
Beta_hat2 = Beta_hatFunction(x_axis1, y_axis1)
print('Beta_hat for 5c(ii)',Beta_hat2)


x_test = np.array(data_needed17[['FG%','TRB']])
dump = np.ones(x_test.shape[0]).reshape(-1,1)
x_test = np.hstack((dump,x_test))
main_result2 = x_test.dot(Beta_hat2)

expected = np.array(data_needed17[['PTS']])
sse2 = SSE(expected, main_result2)
print("SSE of 5c(ii):",SSE(expected, main_result2))
mape2 = MAPE(expected, main_result2)
print("MAPE of 5c(ii):",mape2)



x_axis1 = np.array(data_needed10[['FG%','TRB']])
x_axis2 = np.array(data_needed11[['FG%','TRB']])
x_axis3 = np.array(data_needed12[['FG%','TRB']])
x_axis4 = np.array(data_needed13[['FG%','TRB']])
x_axis5 = np.array(data_needed14[['FG%','TRB']])
x_axis6 = np.array(data_needed15[['FG%','TRB']])
x_axis7 = np.array(data_needed16[['FG%','TRB']])

x_axis1 = np.vstack((x_axis1,x_axis2,x_axis3,x_axis4,x_axis5,x_axis6,x_axis7))

dump = np.ones(x_axis1.shape[0]).reshape(-1,1)
x_axis1 = np.hstack((dump,x_axis1))


y_axis1 = np.array(data_needed10[['PTS']])
y_axis2 = np.array(data_needed11[['PTS']])
y_axis3 = np.array(data_needed12[['PTS']])
y_axis4 = np.array(data_needed13[['PTS']])
y_axis5 = np.array(data_needed14[['PTS']])
y_axis6 = np.array(data_needed15[['PTS']])
y_axis7 = np.array(data_needed16[['PTS']])
y_axis1 = np.vstack((y_axis1,y_axis2,y_axis3,y_axis4,y_axis5,y_axis6,y_axis7))
Beta_hat3 = Beta_hatFunction(x_axis1, y_axis1)

x_test = np.array(data_needed17[['FG%','TRB']])
dump = np.ones(x_test.shape[0]).reshape(-1,1)
x_test = np.hstack((dump,x_test))
main_result3 = x_test.dot(Beta_hat3)

expected = np.array(data_needed17[['PTS']])
sse3 = SSE(expected, main_result3)
print("SSE of 5c(iii):",sse3)
mape3 = MAPE(expected, main_result3)
print("MAPE of 5c(iii):",mape3)


#-----------------5d--------------------------------------------------------------

residual1 = ResidualFn(expected, main_result)
residual2 = ResidualFn(expected, main_result2)
residual3 = ResidualFn(expected, main_result3)


plt.figure('5d(i)', figsize=(6,6))
plt.scatter(main_result, residual1)
plt.xlabel('PTS')
plt.ylabel('Residuals - 5d(i)')
plt.show()


plt.figure('5d(ii)',figsize=(6,6))
plt.scatter(main_result2, residual2)
plt.xlabel('PTS')
plt.ylabel('Residuals - 5d(ii)')
plt.show()

plt.figure('5d(iii)',figsize=(6,6))
plt.scatter(main_result3, residual3)
plt.xlabel('PTS')
plt.ylabel('Residuals - - 5d(iii)')
plt.show()

#-----------------5e--------------------------------------------------------------


plt.figure('5e(i)', figsize=(6,6))
plt.hist(residual1, density=True)
plt.xlabel('Residuals - 5e(i)')
plt.ylabel('PDF')
plt.show()


plt.figure('5e(ii)',figsize=(6,6))
plt.hist(residual2, density=True)
plt.xlabel('Residuals - 5e(ii)')
plt.ylabel('PDF')
plt.show()

plt.figure('5e(iii)',figsize=(6,6))
plt.hist(residual3, density=True)
plt.xlabel('Residuals - 5e(iii)')
plt.ylabel('PDF')
plt.show()

##########Inference####################################################
#Yes, the plots are normal. 
#Normality is very necessary for residuals. 
#As for non normal residuals, the error is not consistent over the 
#entire data set which implies that the prediction is not same across all the dependent variables which is not seen in this case. 
# Also as we know, the inference varies on the dependencies and the variables associated with it.
################################################
