# Question 2
 #finish part 2, fix inline loops, make a a get phi method

import operator
import numpy as np 
import matplotlib.pyplot as plt


def read_data(filename):
	my_data = np.genfromtxt(filename, delimiter = ',')
	# assign first column to input variable and second to target
	x_train, Y_train = my_data[:,0], my_data[:,1]
	# we need to order the array wrt x and keep each x paired with the corresponding Y
	my_sorted_data = sorted(zip(x_train, Y_train), key = operator.itemgetter(0))
	sorted_x_train, sorted_Y_train = zip(*my_sorted_data)
	return sorted_x_train, sorted_Y_train

def get_weight_vector(x_vector, Y_vector, degree):
	phi = np.zeros((degree+1)*len(x_vector)).reshape(len(x_vector), (degree+1))
	for i in range (len(x_vector)): # plus one because we start from zero
		for j in range (degree+1):
			phi[i][j] = x_vector[i]**j
	#phi = np.array([[x_vector[i]**j for j in range(degree+1)] for i in range(len(x_vector))])
	# w = (φ^Τ * φ)^-1 * φ^Τ * y
	weight_vector = np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(Y_vector)
	return weight_vector

def pred_Y(x, weight_vector):
	y = np.zeros(len(x))
	for i in range(len(x)):
		for j in range (len(weight_vector)):
			y[i] += weight_vector[j] * x[i] ** j
	return y.T
	#return np.array([sum([weight_vector[i] * x[j] ** i for i in range(len(weight_vector))]) for j in range(len(x))]).T


def MSE(Y,Y_pred):
    return sum([(Y_pred[i] - Y[i]) ** 2 for i in range(len(Y))])/len(Y)

def L2_get_weight_vector(x_vector, Y_vector, degree, lambda_value):
	# w = (x^T * x + λI)^-1 * x^T * y
	return  np.linalg.inv(x_vector.T.dot(x_vector) + lambda_value*np.identity(degree + 1)).dot(x_vector.T).dot(Y_vector)

def L2_regularization(x_vector, Y_vector, degree):
	weight_list = []
	lambda_value_list = []
	phi = np.zeros((degree+1)*len(x_vector)).reshape(len(x_vector), (degree+1))
	for i in range (len(x_vector)): # plus one because we start from zero
		for j in range (degree+1):
			phi[i][j] = x_vector[i]**j
	lambda_value = 1e-4
	while lambda_value <= 1:
		lambda_value_list.append(lambda_value)
		weight_vector = L2_get_weight_vector(phi,Y_vector, degree, lambda_value)
		weight_list.append(weight_vector)
		lambda_value += 1e-4
	return lambda_value_list, weight_list

def L2_get_MSE(x_vector,Y_vector, weight_list):
	MSE_list = []
	for weight_vector in weight_list:
		MSE_list.append(MSE(Y_vector,pred_Y(x_vector, weight_vector)))
	return MSE_list

def plot_fit(x_train, Y_train, x_validation, Y_validation, weight, degree):
	plt.title('Regression Fit without Regularization for Polynomial of Degree {}'.format(degree))
	plt.xlabel('x values')
	plt.ylabel('Y values')
	plt.plot(x_train, Y_train, 'bo', label = "Training Set")
	plt.plot(x_validation, Y_validation, 'ro', label = "Validation Set")
	plt.plot(x_validation, pred_Y(x_validation, weight))
	plt.legend(loc = 'upper left')
	plt.axis([-1.1,1.1,-20, 50])
	plt.show()

def plot_L2_MSE(x_train, Y_train, x_validation, Y_validation, weight, degree):
	plt.title("Mean Squared Error vs. Regularization Parameter Lambda")
	plt.xlabel('Lambda')
	plt.ylabel('MSE')
	lambda_value_list, weight_list = L2_regularization(x_train, Y_train, degree)
	lambdas = np.asarray(lambda_value_list)
	weights = np.asarray(weight_list)
	MSE_train_list = L2_get_MSE(x_train, Y_train, weights)
	MSE_validation_list = L2_get_MSE(x_validation, Y_validation, weights)
	plt.plot(lambdas, MSE_train_list, 'bo', label = 'Training')
	plt.plot(lambdas, MSE_validation_list, 'ro', label = 'Validation')
	plt.legend(loc = 'upper right')
	plt.show()

def plot_test_fit(x_train, Y_train, x_test, Y_test, degree):#best lambda is 0.0202
	plt.title('Fit of test Data')
	plt.xlabel('x values')
	plt.ylabel('Y values')
	lambda_value_list, weight_list = L2_regularization(x_train, Y_train, degree)
	lambdas = np.asarray(lambda_value_list)
	weights = np.asarray(weight_list)
	MSE_train_list = L2_get_MSE(x_train, Y_train, weights)
	MSE_validation_list = L2_get_MSE(x_validation, Y_validation, weights)
	print('Optimal Lambda =', lambdas[MSE_validation_list.index(min(MSE_validation_list))], 'with MSE = ', min(MSE_validation_list))	
	plt.plot(x_test, Y_test, 'go', label = 'Test')
	plt.plot(x_test, pred_Y(x_test, weights[MSE_validation_list.index(min(MSE_validation_list))]))
	plt.legend(loc = 'upper left')
	plt.show()

if __name__ == "__main__":
	degree = 20
	x_train, Y_train = read_data("Dataset_1_train.csv")
	x_validation, Y_validation = read_data("Dataset_1_valid.csv")
	x_test, Y_test = read_data("Dataset_1_test.csv")
	weight = get_weight_vector(x_train, Y_train, degree)
	print("Training MSE:", MSE(Y_train, pred_Y(x_train, weight)))
	print("Validation MSE:", MSE(Y_validation, pred_Y(x_validation,weight)))
	plot_fit(x_train, Y_train, x_validation, Y_validation, weight, degree)
	plot_L2_MSE(x_train, Y_train, x_validation, Y_validation, weight, degree)
	plot_test_fit(x_train, Y_train, x_test, Y_test, degree)