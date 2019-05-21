import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv

# Function that reads values from .gz file
def getdata(filename):    
    return read_csv(filename,header=None,dtype=np.float64).values

# Read in the training and test digits
TrainingDigits = getdata('TrainDigitX.csv.gz')
TestDigits1 = getdata('TestDigitX.csv.gz')
TestDigits2 = getdata('TestDigitX2.csv.gz')

# Append the bias (1) to each of the digit vectors, in position 784
TrainingDigits = np.c_[TrainingDigits, np.ones(len(TrainingDigits))]
TestDigits1 = np.c_[TestDigits1, np.ones(len(TestDigits1))]
TestDigits2 = np.c_[TestDigits2, np.ones(len(TestDigits2))]

# Now read in the labels
TrainLabels = read_csv('TrainDigitY.csv',header=None)
TestLabels1 = read_csv('TestDigitY.csv',header=None)

# Following is the sigmoid function returning 1/(1 + e^(-x))

# Form matrix of 0's and 1's from training labels
def to_matrix(L):
    labels = np.array(L)
    q = len(labels)
    p = len(np.unique(labels))
    M = np.zeros((q, p))
    M[np.arange(q), labels.flatten().astype(int)] = 1
    return M

Y = to_matrix(TrainLabels)
Y_Test = to_matrix(TestLabels1)

def Sigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))

# Following is the derivative of the sigmoid function, ASSUMING sig is passed in
def SigDeriv(x):
   return x*(1 - x)     # not using sigmoid(x) because that is the paramenter passed in

# Calculate the softloss vector 
def calcSoftLoss(y):
    sumcomponents = 0
    for i in range(len(y)):
        sumcomponents += np.exp(y[i])
    return np.array([(np.exp(y[i])/sumcomponents) for i in range(len(y))])

def Entropy(alpha):
    sample_size = Y.shape[0]
    var = alpha - Y
    #print("Cross-Entropy",pred,real, res,n_samples)
    return var / sample_size

#Construct unit m-vector with 1 in the n-th index; n <= m - 1
def unit_v(m,n):
  unit_v = [0 for i in range(m)]
  unit_v[n] = 1  
  return(unit_v)

#Calculate mean square error. Here n is the correct digit, x is the output vector
def calcMeanSqErr(x,n):
    return(np.linalg.norm(x - unit_v(len(x),n)))
    
        
# Create randomized m x n weight matrices for starting the forward propagation
def Weights(m,n):
    return np.random.randn(m,n)

# Calculate number of errors after forward propogation using final Alpha
def num_errors(alpha):
    global Y
    err_count = 0
    for i in range(len(alpha)):
        if (Y[i][np.argmax(alpha[i])] == 0):
            err_count += 1
    return err_count

# Calculate number of errors for first set of test digits after forward propogation
# using final Alpha. Also writes the predicted test digits to a csv file
def num_errors_test1(alpha):
    global Y_Test
    err_count = 0
    digits = np.zeros(len(alpha),dtype=int)
    for i in range(len(alpha)):
        pred_digit = np.argmax(alpha[i])
        digits[i] = pred_digit 
        if (Y_Test[i][pred_digit] == 0):
            err_count += 1
    ## Now write the array digits to Test1_Predictions.csv
    np.savetxt("Test1_Predictions.csv",digits,fmt='%01d',delimiter=",")
    return err_count

# This function processes the FIRST set of test digits. It uses the
# final weights and biases that came out of the training phase
def process_test_dig1():
    Z_1 = np.dot(TestDigits1,FinalW1) + FinalBias_1
    Alpha_1 = Sigmoid(Z_1)
    Z_2 = np.dot(Alpha_1,FinalW2) + FinalBias_2
    Alpha_2 = Sigmoid(Z_2)
    Z_3 = np.dot(Alpha_2,FinalW3) + FinalBias_3
    Alpha_3 = Sigmoid(Z_3)                  # Alpha_3 is the final output
    error_count = num_errors_test1(Alpha_3) 
    error_rate = error_count/len(TestDigits1)
    print("Error rate for FIRST set of test digits is",100*error_rate,"%")
    
# This function processes the SECOND set of test digits. It uses the
# final weights and biases that came out of the training phase.
# Also writes the predicted test digits to a csv file
def process_test_dig2():
    Z_1 = np.dot(TestDigits2,FinalW1) + FinalBias_1
    Alpha_1 = Sigmoid(Z_1)
    Z_2 = np.dot(Alpha_1,FinalW2) + FinalBias_2
    Alpha_2 = Sigmoid(Z_2)
    Z_3 = np.dot(Alpha_2,FinalW3) + FinalBias_3
    Alpha_3 = Sigmoid(Z_3)                  # Alpha_3 is the final output
    digits = np.zeros(len(Alpha_3),dtype=int)
    for i in range(len(Alpha_3)):
        pred_digit = np.argmax(Alpha_3[i])
        digits[i] = pred_digit 
    ## Now write the array digits to Test2_Predictions.csv
    np.savetxt("Test2_Predictions.csv",digits,fmt='%01d',delimiter=",")
    

# Now construct the neural network with L - 1 hidden layers. Layer 0 is the
# input layer and Layer L is the output layer consisting of 10 neurons
global FinalW1
global FinalW2
global FinalW3
global FinalBias_1
global FinalBias_2
global FinalBias_3
def NeuralNet(lrate, num_epochs):
    global FinalW1
    global FinalW2
    global FinalW3
    global FinalBias_1
    global FinalBias_2
    global FinalBias_3
    
 ## SET UP Neural Net parameters 
    L = 3                               # There are (L - 1) HIDDEN LAYERS. Layer L is output layer of 10 neurons
    Lneurons = [785, 128, 128, 10]       # This needs to be hard coded to match L. 785 and 10 are fixed. Experiment with other numbers.
    eta = lrate                          # This is the LEARNING RATE. Experiment with this number.
    epochs = num_epochs                          # This is the number of EPOCHS. Experiment with this number. 
    # Construct randomized weight matrices, L in number - so add/delete lines of code if L is changed
    # Also construct the bias vectors
    W1 = Weights(Lneurons[0],Lneurons[1])
    W2 = Weights(Lneurons[1],Lneurons[2])
    W3 = Weights(Lneurons[2],Lneurons[3])
    Bias_1 = np.zeros((1, Lneurons[1]))
    Bias_2 = np.zeros((1, Lneurons[2]))
    Bias_3 = np.zeros((1, Lneurons[3]))
    
    
    for eps in range(epochs):               # Start the epochs, each of size batch_size and
        min_error_rate = 1.1
      #### START forward propagation ####
        Z_1 = np.dot(TrainingDigits,W1) + Bias_1
        Alpha_1 = Sigmoid(Z_1)
        Z_2 = np.dot(Alpha_1,W2) + Bias_2
        Alpha_2 = Sigmoid(Z_2)
        Z_3 = np.dot(Alpha_2,W3) + Bias_3
        Alpha_3 = Sigmoid(Z_3)                  # Alpha_3 is the final output
        error_count = num_errors(Alpha_3) 
        error_rate = error_count/len(TrainingDigits)
        if (error_rate < min_error_rate):
            min_error_rate = error_rate
        #print(error_count,Alpha_3.shape,Alpha_3)       # for debugging
        print("Error rate for Epoch", eps + 1, "( of", epochs, ") is",100*error_rate,"%")
      #### END forward progatation   ####
        
      #### START backward propagation ####
        # First calculate the delta's in reverese
        Delta_3 = Entropy(Alpha_3)
        Z_2 = np.dot(Delta_3, W3.T)
        Delta_2 = Z_2 * SigDeriv(Alpha_2)
        Z_1 = np.dot(Delta_2, W2.T)
        Delta_1 = Z_1 * SigDeriv(Alpha_1)
        # print(Delta_1)                  # for debugging
        # Next update the weight matrices and the bias vectors. eta is the learning rate
        W1 -= eta * np.dot(TrainingDigits.T, Delta_1)
        W2 -= eta * np.dot(Alpha_1.T, Delta_2)
        W3 -= eta * np.dot(Alpha_2.T, Delta_3)
        Bias_1 -= eta * np.sum(Delta_1, axis=0)
        Bias_2 -= eta * np.sum(Delta_2, axis=0)
        Bias_3 -= eta * np.sum(Delta_3, axis=0, keepdims=True)
        # print(W3, Bias_3)               # for debugging
     #### END backward propagation   ####
    print("Lowest error rate for all",epochs, "epochs is",100*min_error_rate)  
    # Store the Weights and biases for use with the test digits
    FinalW1 = W1
    FinalW2 = W2
    FinalW3 = W3
    FinalBias_1 = Bias_1
    FinalBias_2 = Bias_2
    FinalBias_3 = Bias_3
    return min_error_rate
    
def main():
    
    # Loading of the digits and labels was done earlier in line, so we can start the
    # main program by setting up the neural net. Note that the best prediction rate of 92.346% was realized
    # using learning rate = 0.8 and number of epochs = 1000. We are using a lower number of epochs here, so the 
    # program finishes execution in a couple of minutes. It takes about 20 minutes with epochs set to 1000.
    print("STARTING set up of neural net with each epoch consisting of a forward and backward propagation")
    print("NOTE THAT THIS IS A DEMO ONLY. Please refer to attached write-up for parameters which need to be set for optimal performance.")
    print("FOR BEST PREDICTION RATE CHANGE number of epochs (second parameter) to 1000. This will require about 20 minutes execution time")
    error_rate = NeuralNet(0.8,100)          # FIRST parameter is learning rate; SECOND parameter is number of epochs
    print("Highest successful prediction rate is", 100*(1 - error_rate),"%")
    
 ## BEGIN neural net testing with two sets of test digits
    print("Neural net set up and training complete. Use updated weight matrices and bias vectors to predict test digits\n")
    print("First use neural net to predict FIRST SET of test digits. Output labels are in the Test1_Predictions.csv file.\n  ")
    process_test_dig1()    
    print("\nNext use neural net to predict SECOND SET of test digits.")
    process_test_dig2() 
    print("Second set prediction complete - predicted labels for SECOND SET of test digits are in the Test2_Predictions.csv file.\n")
    print("                                 THE PROGRAM HAS NOW COMPLETED                                 ")
 ## END Neural Net testing