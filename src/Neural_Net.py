'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

The hyperparameters used are-
	- np.random.seed(3) - We have used 3 as the seed value as it gave us a better accuracy.
	- def sigmoid - the activation function we have used for our implementation is the sigmoid
		function.
	- def mse_loss - This is our loss function. It takes two arrays and gives us the loss value.
	- layers - We have taken 3 layers for our implementation -
															1) input layer with 9 inputs.
															2) hidden layer with 5 neurons.
															3) output layer with 1 neuron.
	- learning rate - The initial learning rate for our implementation is 0.05. We increase it by 1
	  after training our model with one batch of the dataset. Increasing the learning rate prevented 
	  our model from getting stuck in local minima and also helped in training our model better.
	- epochs- The number of epochs is 200 as it worked well with our model.
	- 0.6 is our deciding factor to determine as to which label it will belong to i.e 1 or 0.
	  If output is greater than or equal to 0.6 then we assign it a value of 1 otherwise 0.
'''
# importing modules
import numpy as np		# for mathematical calculations and handling array operations 
import pandas as pd 	# for reading and handling data
from sklearn.model_selection import train_test_split	# to split dataset into test set and training set

np.seterr(all='ignore')
np.random.seed(3)

# sigmoid activation 
def sigmoid(x):
	x = np.float32(x)
	return 1/(1+ np.exp(-x))

# differential of sigmoid function 
def sigmoidDerivative(x):
	x = np.float32(x)
	o = sigmoid(x)
	return o * (1 - o)

# loss function 
def mse_loss(truey, predy):
  return (((truey - predy) ** 2).mean())

# preprocessing dataset
''' This function takes the dataset, converts each column into a list and the for loop organises it in the proper 
	form of input and output.
	This function makes our dataset ready for our model implementation.
'''
def typecastData(df):
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
	def __init__(self, layers=[9, 5, 1]):
		self.weights = []
		self.biases = []
		self.layers = layers
		x = 0
		for i in self.layers[1:]:
			for k in range(i):
				tlist = []
				for j in range(self.layers[x]):
					tlist.append(np.random.randn())
				self.weights.append(tlist)
			x += 1
		#print(self.weights)

		myiter = sum(self.layers[1:])
		for i in range(myiter):
			self.biases.append(np.random.random())
		#print(self.biases)

	'''This function calculates weight*input + bias value for every neuron and passes it through the activation	function '''
	def forwardProp(self, x):
		sumLayer = []
		a = 0
		for i in range(self.layers[1]):
			tlist = []
			for j in range(len(self.weights[a])):
				tlist.append(self.weights[a][j]*x[j])
			tlist.append(self.biases[a])
			sumLayer.append(sigmoid(sum(tlist)))	#sumLayer stores the value which is to be propogated to the next layer
			a += 1
		tlist2 = []
		for i in range(len(self.weights[a])):
			tlist2.append(self.weights[a][i]*sumLayer[i])	#tlist2 combines the output of neuron with weights to pass to the next layer
		tlist2.append(self.biases[a])
		o = sigmoid(sum(tlist2))
		return o

	def fit(self,X,Y,lr = 0.05):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		epochs = 200
		loss = 1
		for epoch in range(epochs):
			for x,truey in zip(X,Y):
				sumLayer = []
				# FINDING THE FEEDFORWARD OUTPUT AND SUM OF EVERY LAYER
				a = 0
				for i in range(self.layers[1]):
					tlist = []
					for j in range(len(self.weights[a])):
						tlist.append(self.weights[a][j]*x[j])
					tlist.append(self.biases[a])
					sumLayer.append(sigmoid(sum(tlist)))	#tlist contains the values weight*input + biases for layer 1
					a += 1
				tlist2 = []
				for i in range(len(self.weights[a])):
					tlist2.append(self.weights[a][i]*sumLayer[i])
				tlist2.append(self.biases[a])
				oSum = sum(tlist2)
				sumLayer.append(oSum)
				predy = sigmoid(oSum)

				der_predy = -2 * (truey - predy)	#this holds the error

				der_any_wrt_b = []
				for i in range(len(self.biases)):
					der_any_wrt_b.append(sigmoidDerivative(sumLayer[i]))
				
				der_h_wrt_w = []
				for i in range(self.layers[1]):
					temp = []
					for j in range(len(self.weights[i])):
						temp.append(x[j]*sigmoidDerivative(sumLayer[i]))
					der_h_wrt_w.append(temp)
						
				der_predy_wrt_h = []
				for i in range(self.layers[1]):
					der_predy_wrt_h.append(self.weights[-1][i]*sigmoidDerivative(oSum))

				der_predy_wrt_w = []
				for i in range(self.layers[1]):
					der_predy_wrt_w.append(sumLayer[i]*sigmoidDerivative(oSum))
				
				# chain differentiation 
				for i in range(self.layers[1]):
					for j in range(len(self.weights[i])):
						self.weights[i][j] -= lr * der_predy * der_predy_wrt_h[i] * der_h_wrt_w[i][j]
					self.biases[i] -= lr * der_predy * der_predy_wrt_h[i] * der_any_wrt_b[i]

				# updating weights and biases 
				for i in range(self.layers[1]):
					self.weights[-1][i] -= lr * der_predy * der_predy_wrt_w[i]
				self.biases[-1] -= lr * der_predy * der_any_wrt_b[-1]
				
				# printing epoch losses
				if epoch % 10 == 0:
					preds_y = np.apply_along_axis(self.forwardProp, 1, X)
					loss = mse_loss(Y, preds_y)
					print("Epoch %d loss: %.3f" % (epoch, loss))
		return loss
	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		# simple forward propagation 
		yhat = []
		for x in X:
			yhat.append(self.forwardProp(x))
		return yhat


	def CM(self, y_test, y_test_obs):
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

		# applying formulae 
		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		a = (tp+tn)/(tp+tn+fp+fn)

		# displaying out 
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"Accuracy: {a*100}%")

# ----------------MAIN CODE----------------

# reading data 
data = pd.read_csv('../data/cleaned_Dataset.csv')
df = pd.DataFrame(data)

X, y = typecastData(df)

# splitting dataset into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model creation 
model = NN()

lr = 0.05
# training with batches of training data (BATCH PROCESSING)
for i in range(2):
	a,b,c,d = train_test_split(X_train,y_train,test_size = 0.7)
	model.fit(a,c,lr)
	lr = lr + 0.01

# prediction 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print('*'*50)

# confusion matrix 
print("Training Accuracy")
model.CM(y_train, y_pred_train)
print('*'*50)
print("Testing Accuracy")
model.CM(y_test, y_pred_test)