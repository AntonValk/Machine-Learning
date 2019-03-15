# Question 3
import numpy as np
import matplotlib.pyplot as plt
import operator

ALPHA = 1e-6
weights = np.zeros(2,dtype=float)

def read_data(filename):
	my_data = np.genfromtxt(filename, delimiter = ',')
	# assign first column to input variable and second to target
	x, Y = my_data[:,0], my_data[:,1]
	# we need to order the array wrt x and keep each x paired with the corresponding Y
	my_sorted_data = sorted(zip(x, Y), key = operator.itemgetter(0))
	sorted_x, sorted_Y = zip(*my_sorted_data)
	return sorted_x, sorted_Y

def predict_Y(x):
	return weights[0] + weights[1] * x

def MSE(true_Y, estimated_Y):
	return sum([(estimated_Y[i] - true_Y[i]) ** 2 for i in range (len(true_Y))])/len(true_Y)

def sgd(x, y, weights, learning_rate):
	weights[0] -= learning_rate*(predict_Y(x) - y)
	weights[1] -= learning_rate*(predict_Y(x) - y)*x

def sgd_optimizer(x,y, learning_rate):
	for i in range(len(x)):
		sgd(x[i], y[i], weights, learning_rate)

def sgd_mse_list(x,y, learning_rate):
	mse_list = []
	x_list = []
	y_list = []
	for i in range(len(np.array(x))):
		x_list.append(x[i])
		y_list.append(y[i])
		sgd(x[i], y[i], weights, learning_rate)
		estimated_Y = [predict_Y(j) for j in x_list]
		mse_list.append(MSE(y_list, estimated_Y))
		mse_array = np.asarray(mse_list)
	return np.mean(mse_array)

def mse_log_graph(x_train,Y_train,x_valid,Y_valid,learning_rate,epochs):
	mse_list_val = []
	mse_list_train = []
	plt.figure(1)
	for i in range(epochs + 1):
		sgd_optimizer(x_train,Y_train, learning_rate)
		mse_list_val.append(np.log10(sgd_mse_list(x_valid,Y_valid, ALPHA)))
	plt.plot(mse_list_val)
	plt.title('Validation Logarithm of the Mean Squared Error vs. number of Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('Log10(MSE)')
	plt.show()
	plt.figure(2)
	for i in range(epochs + 1):
		sgd_optimizer(x_train,Y_train, learning_rate)
		mse_list_train.append(np.log10(sgd_mse_list(x_valid,Y_valid, ALPHA)))
	plt.plot(mse_list_train)
	plt.title('Training Logarithm of the Mean Squared Error vs. number of Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('Log10(MSE)')
	plt.show()

def mse_graph(x_train,Y_train,x_valid,Y_valid,learning_rate,epochs):
	mse_list_val = []
	mse_list_train = []
	plt.figure(1)
	for i in range(epochs + 1):
		sgd_optimizer(x_train,Y_train, learning_rate)
		mse_list_val.append(sgd_mse_list(x_valid,Y_valid, ALPHA))
	plt.plot(mse_list_val)
	plt.title('Validation Mean Squared Error vs. number of Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('MSE')
	plt.show()
	plt.figure(2)
	for i in range(epochs + 1):
		sgd_optimizer(x_train,Y_train, learning_rate)
		mse_list_train.append(sgd_mse_list(x_train,Y_train, ALPHA))
	print("MSE reached after",epochs,"epochs is:",mse_list_val[-1])
	plt.plot(mse_list_train)
	plt.title('Training Mean Squared Error vs. number of Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('MSE')
	plt.show()

def fit_visualization(x_valid, Y_valid, learning_rate, epochs):
	weights = [0.0, 0.0]
	fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
	fig.suptitle("Validation Fit as the Number of Epochs Increases", fontsize=18)
	pred_list_0 = []
	pred_list_1 = []
	pred_list_2 = []
	pred_list_3 = []
	pred_list_4 = []
	pred_list_5 = []
	#mse_list_val = []

	for i in range(epochs + 1):
		sgd_optimizer(x_train,Y_train, learning_rate)
		#mse_list_val.append(sgd_mse_list(x_valid,Y_valid, ALPHA))
		if i == 0:
			for i in range(len(x_valid)):
				pred_list_0.append(predict_Y(x_valid[i]))
			plt.subplot(2,3,1)
			plt.plot(x_valid,Y_valid, 'ro')
			plt.plot(x_valid, pred_list_0)
			plt.title('Fit after Epoch 1', fontsize = 9)
		if i == 200:
			for i in range(len(x_valid)):
				pred_list_1.append(predict_Y(x_valid[i]))
			plt.subplot(2,3,2)
			plt.plot(x_valid,Y_valid, 'ro')
			plt.plot(x_valid, pred_list_1)
			plt.title('Fit after Epoch 200', fontsize = 9)
		if i == 500:
			for i in range(len(x_valid)):
				pred_list_2.append(predict_Y(x_valid[i]))
			plt.subplot(2,3,3)
			plt.plot(x_valid,Y_valid, 'ro')
			plt.plot(x_valid, pred_list_2)
			plt.title('Fit after Epoch 500', fontsize = 9)
		if i == 1500:
			for i in range(len(x_valid)):
				pred_list_3.append(predict_Y(x_valid[i]))
			plt.subplot(2,3,4)
			plt.plot(x_valid,Y_valid, 'ro')
			plt.plot(x_valid, pred_list_3)
			plt.title('Fit after Epoch 1500', fontsize = 9)
		if i == 4000:
			for i in range(len(x_valid)):
				pred_list_4.append(predict_Y(x_valid[i]))
			plt.subplot(2,3,5)
			plt.plot(x_valid,Y_valid, 'ro')
			plt.plot(x_valid, pred_list_4)
			plt.title('Fit after Epoch 4000', fontsize = 9)
		if i == 9000:
			for i in range(len(x_valid)):
				pred_list_5.append(predict_Y(x_valid[i]))
			plt.subplot(2,3,6)
			plt.plot(x_valid,Y_valid, 'ro')
			plt.plot(x_valid, pred_list_5)
			plt.title('Fit after Epoch 9000', fontsize = 9)
	plt.subplots_adjust(hspace=0.3, wspace = 0.25)
	plt.show()


if __name__ == '__main__':
	x_train, Y_train = read_data("Dataset_2_train.csv")
	x_valid, Y_valid = read_data("Dataset_2_valid.csv")
	#mse_graph(x_train,Y_train,x_valid,Y_valid,ALPHA,5000)
	#mse_log_graph(x_train,Y_train,x_valid,Y_valid,ALPHA,5000)
	# plt.plot(x_train,Y_train, 'bo')
	# plt.plot(x_valid,Y_valid, 'ro')
	#pred_list = []
	# for i in range(len(x_valid)):
	# 	pred_list.append(predict_Y(x_valid[i]))
	# plt.plot(x_valid, pred_list)
	# plt.show()
	# print(weights)
	fit_visualization(x_valid,Y_valid,ALPHA,9000)