#Question 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

na_values = ['?']

# drop columns 1 to 5 because txt file explains that
# they provide information unuable for predictive purposes
# and only for reference purposes
def read_data(filename):
	df = pd.read_csv(filename, header = None, na_values = na_values)
	print(df.shape)
	print (df.head())
	#drop(df.columns[[1, 69]], axis=1, inplace=True)
	df.drop(df.columns[[0,1,2,3,4]], axis = 1, inplace=True)
	print (df.head())
	print(df.shape)
	return df

def replace_nan_mean(dataframe):
	dataframe.fillna(dataframe.mean(axis = 1), inplace = True)
	dataframe.to_csv("Datasets/communities_data_mean.csv")

def replace_nan_median(dataframe):
	dataframe.fillna(dataframe.median(axis = 1), inplace = True)
	dataframe.to_csv("Datasets/communities_data_median.csv")

def k_fold_split(dataset, k):
    for index in range(1, k+1):
        mask = np.random.rand(len(dataset)) < 0.8
        train = dataset[mask]
        test = dataset[~mask]
        train.to_csv('Datasets/CandC-train{}.csv'.format(index))
        test.to_csv('Datasets/CandC-test{}.csv'.format(index))

def split_data(train_data):
    np.random.shuffle(train_data)
    slice = int(len(train_data) * .80)
    return (train_data[:slice],train_data[slice:])

def MSE(true_Y, estimated_Y):
	return sum([(estimated_Y[i] - true_Y[i]) ** 2 for i in range (len(true_Y))])/len(true_Y)

def get_weights(x,Y):
	return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(Y)

def predict_Y(x, weight):
	return x.dot(weight)

def average_mse():
	average_mse = 0
	weights = []
	for i in range (1,6):
		train_data = pd.read_csv('Datasets/CandC-train{}.csv'.format(i))
		test_data = pd.read_csv('Datasets/CandC-test{}.csv'.format(i))
		train_matrix = np.array(train_data.as_matrix())
		test_matrix = np.array(test_data.as_matrix())
		x_train = train_matrix[:,:-1]
		Y_train = train_matrix[:,-1]
		x_test = test_matrix[:,:-1]
		Y_test = test_matrix[:,-1]
		weight = get_weights(x_train,Y_train)
		weights.append(weight)
		mse = MSE(Y_test, predict_Y(x_test, weight))
		average_mse += mse
		print ("MSE {}: {}".format(i, mse))

	average_mse /= 5
	print("Average MSE",average_mse)
	with open("Assignment1_260672034_4_2.txt",'w+') as w:
		for value in weights:
			w.write(str(value)+'\n\n')


def ridge_regression_weights(x,Y,lam):
	return np.linalg.inv(x.T.dot(x) + lam*np.identity(len(x[1]))).dot(x.T).dot(Y)

def ridge_mse(lam):
	average_mse = 0
	weights = []
	for i in range (1,6):
		train_data = pd.read_csv('Datasets/CandC-train{}.csv'.format(i))
		test_data = pd.read_csv('Datasets/CandC-test{}.csv'.format(i))
		train_matrix = np.array(train_data.as_matrix())
		test_matrix = np.array(test_data.as_matrix())
		x_train = train_matrix[:,:-1]
		Y_train = train_matrix[:,-1]
		x_test = test_matrix[:,:-1]
		Y_test = test_matrix[:,-1]
		weight = ridge_regression_weights(x_train,Y_train,lam)
		weights.append(weight)
		mse = MSE(Y_test, predict_Y(x_test, weight))
		average_mse += mse
	average_mse /= 5
	return average_mse

def optimize_lambda():
	lam = 0
	optimal_lambda = 0
	least_mse = 100
	increment = 1e-1
	mse_list = []
	stop = 10
	while lam <= 10:
		mse_list.append(ridge_mse(lam))
		if mse_list[-1] < least_mse:
			least_mse = mse_list[-1]
			optimal_lambda = lam
		lam += increment
	plt.plot(np.linspace(0,stop,int(stop/increment+1)), mse_list, label = 'Mean Squared Error')
	plt.plot(optimal_lambda, least_mse, 'rx', label = 'Minimum MSE')
	plt.legend(loc = 'upper right')
	plt.xlabel('lambda')
	plt.ylabel('MSE')
	plt.title('Test MSE for varying lambda')
	print("Best fit achieved for lamda = {} with MSE = {}".format(optimal_lambda,least_mse))
	plt.show()

def feature_selection(x,Y,lam,remove_num):
	if remove_num >= len(x):
		return np.zeros(len(x))
	weight = ridge_regression_weights(x,Y,lam)
	#weight_list = list(weight)
	for i in range (remove_num):
		minimum_weight = min(weight, key = abs)
		#minimum_weight = min(weight_list, key = abs)
		#min_index = weight_list.index(minimum_weight)
		min_index = np.where(weight==minimum_weight)
		weight[min_index] = 0
	#print("Length of weight is:",len(weight))
	return weight

def feature_selection_ridge_mse(lam, remove_num):
	average_mse = 0
	for i in range (1,6):
		train_data = pd.read_csv('Datasets/CandC-train{}.csv'.format(i))
		test_data = pd.read_csv('Datasets/CandC-test{}.csv'.format(i))
		train_matrix = np.array(train_data.as_matrix())
		test_matrix = np.array(test_data.as_matrix())
		x_train = train_matrix[:,:-1]
		Y_train = train_matrix[:,-1]
		x_test = test_matrix[:,:-1]
		Y_test = test_matrix[:,-1]
		weight = feature_selection(x_train,Y_train,lam,remove_num)
		mse = MSE(Y_test, predict_Y(x_test, weight))
		average_mse += mse
	average_mse /= 5
	return average_mse	

def optimize_lambda_with_feature_selection(remove_num):
	lam = 0
	optimal_lambda = 0
	least_mse = 100
	increment = 1e-1
	mse_list = []
	stop = 10
	while lam <= 10:
		mse_list.append(feature_selection_ridge_mse(lam,remove_num))
		if mse_list[-1] < least_mse:
			least_mse = mse_list[-1]
			optimal_lambda = lam
		lam += increment
	plt.plot(np.linspace(0,stop,int(stop/increment+1)), mse_list, label = 'Mean Squared Error')
	plt.plot(optimal_lambda, least_mse, 'rx', label = 'Minimum MSE')
	plt.legend(loc = 'upper right')
	plt.xlabel('lambda')
	plt.ylabel('MSE')
	plt.title('Test MSE for varying lambda with {} dropped features'.format(remove_num))
	print("Best fit achieved for lamda = {} with MSE = {}".format(optimal_lambda,least_mse))
	plt.show()


if __name__ == '__main__':
	df = read_data('communities_data.csv')
	replace_nan_median(df)
	k_fold_split(df, 5)
	average_mse()
	optimize_lambda()
	optimize_lambda_with_feature_selection(60)


