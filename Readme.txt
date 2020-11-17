NEURAL NETWORK FROM SCRATCH IN PYTHON 
Machine Intelligence Course 5th Semester
Assignment 3 Dated 17th November, 2020

TEAM MEMBERS:
	1. SHAASHWAT JAIN(PES1201802346)
	2. SRISHTI SACHAN(PES1201802126)
	3. RAJDEEP SENGUPTA(PES1201800144)


=========================
MODULES USED:

-Numpy => for mathematical calculations & handling array operations 
-Pandas => for reading and handling datasets 
-Sklearn => ONLY USED FOR splitting dataset into training and testing set 

=========================
DATA CLEANING & PREPROCESSING:

-csv data in the dataset file is read using pandas module in python 
-these are converted into pandas dataframes for making operations easy 
-the columns are normalized 
-the outliers are dealt with 
-the dataset is shuffled   
-the NaN values are replaced with the MODE of that column  

=========================
HYPERPARAMETERS USED:

-The Neural Network has, by default, three layers:
	1. input layer => 9 neurons (same dimensions as the number of features in the dataset)
	2. hidden layer (5 neurons)
	3. output layer (1 neuron)
-Activation function used for each neuron is sigmoid function 
-The input data has been fed to the model in batches (BATCH PROCESSING)
-The learning rate by default is given as 0.05 for best results 
-The epochs by default are considered to be 200
-The loss function chosen is the Mean Squared Error Loss(MSE Loss)
-The train:test split ratio is chosen as 80:20 

=========================
FUNCTIONS USED:

-typecast() => the independent and dependent variables are separated from the dataframe 
-sigmoid() => implementation of sigmoid function using numpy 
-sigmoidDerivative() => implementation of derivative of sigmoid function using simple formula 
-mse_loss() => the loss function calculated between predicted and real values 

===============================
NEURAL NETWORK CLASS

The weight and bias matrices are initialized in the __init__() function

Forward Propagation consists of addition of the (weight matrix dotted with the output of previous layer) and the biases of that layer 

Fit function is the main driver of the neural network as it captures the intermediate sums and activated sums of the neurons and calculates the losses and updates weights and biases by reducing the losses(gradient descent method) for every epoch

Predict function just takes the test dataset and performs the forward propagation based on the previously updated weights and biases 

Confusion Matrix function takes in the real and observed outputs and computes the confusion matrix. It also computes the Precision, Recall, F1 Score and Accuracy using the appropriate formulae.

===============================

