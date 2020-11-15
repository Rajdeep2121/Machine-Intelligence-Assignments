'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
# IMPORTING MODULES
import numpy as np		# for mathematical calculations and handling array operations 
import pandas as pd 	# for reading and handling data
from sklearn.model_selection import train_test_split	# to split dataset into test set and training set

# NORMALISING DATAFRAME
def normalize(df):
		result = df.copy()
		for feature_name in df.columns:
				max_value = df[feature_name].max()
				min_value = df[feature_name].min()
				if min_value == max_value:
						result[feature_name] = df[feature_name] / max_value
				else:
						result[feature_name] = (
								df[feature_name] - min_value) / (max_value - min_value)
		return result

def sigmoid(x):
		return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
		o = sigmoid(x)
		return o * (1 - o)

# PREPROCESSING THE DATASET
def preprocessData(df):
	# DROPPING ROWS WITH NAN VALUES
	df.dropna(axis=0, inplace=True)     # removing rows with NaN values

	# SHUFFLING ROWS TO PREVENT ANY SAMPLING ERRORS
	df = df.sample(frac = 1)

	# NORMALIZING COLUMN VALUES
	new_df = normalize(df)
	df = new_df

	# PREPARING INPUT AND OUTPUT DATA 
	community = df['Community'].tolist()
	age = df['Age'].tolist()
	weight = df['Weight'].tolist()
	delphase = df['Delivery phase'].tolist()
	hb = df['HB'].tolist()
	ifa = df['IFA'].tolist()
	bp = df['BP'].tolist()
	education = df['Education'].tolist()
	residence = df['Residence'].tolist()
	y = df['Result'].tolist()

	X = []
	for i in range(df.shape[0]):
			X.append([community[i], age[i], weight[i], delphase[i],
								hb[i], ifa[i], bp[i], education[i], residence[i]])
	return X, y


class NN:
	def __init__(self, layers=[9, 2, 1]):
		self.weights = []
		self.biases = []
		
		# INITIALIZING THE WEIGHTS AND BIASES MATRIX WITH RANDOM VALUES
		for i in range(len(self.layers) -1):
			self.weights.append(np.random.randn(self.layers[i+1], self.layers[i]))
			self.biases.append(np.random.randn(self.layers[i+1], 1))

	def forwardProp(self, X):
		sumLayer = []
		outputLayer = [X]

		# FINDING THE FEEDFORWARD OUTPUT AND SUM OF EVERY LAYER
		for i in range(len(self.weights)):
			sumLayer.append(np.dot(self.weights[i], outputLayer[-1]) + self.biases[i])
			out = sigmoid(sumLayer[-1])
			outputLayer.append(out)
		return sumLayer, outputLayer

	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat

	def CM(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
			


# ----------------MAIN CODE----------------

# READING DATA
data = pd.read_csv('LBW_Dataset.csv')
df = pd.DataFrame(data)

X, y = preprocessData(df)

# SPLITTING DATASET INTO TESTING AND TRAINING SET 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# MODEL CREATION
