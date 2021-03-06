# Multilayer Perceptron Testing

## Project for neural networks in python - MLPperceptron testing based on scikit library

### Project Description
The aim of this project is to train and test linear neural multiperceptron network class 
basing on MNIST database of handwritten digits.	As a neural network core engine scikit-learn library
has been used. It has a very flexible "MLPClassifier" class which allows for a number of parameters to be
set as the following ones: 
```
MLPClassifier(
	hidden_layer_sizes=(30,),
	activation='relu',
	max_iter=30,
	alpha=1e-4,
	batch_size='auto',
	solver='sgd',
	verbose=0,
	random_state=1,
	learning_rate_init=0.1,
) 
```
The MNIST database of handwritten digits, has a training set of 60,000 examples, 
and a test set of 10,000 examples. The digits dataset consists of 28x28 pixel images of digits. 
The images attribute of the dataset stores 28x28 arrays of grayscale values for each image. 
Images are transformed to a vectors of 28 x 28 = 784 values and converted to floating points 
expressed in between 0 and 1. This set is used for neural network optimization and testing
	There are 12 test cases constructed with different sets of parameters in order to asses 
neural network loss function and optimizer fitting. And to find the proper configuration 
of parameters so that neural network has the highest fit scores.	

![image](https://user-images.githubusercontent.com/75398126/152051085-fbabf17c-00d9-4f06-8c7f-0b1e66c9b4e9.png)
![image](https://user-images.githubusercontent.com/75398126/152051282-6d9e71a2-8069-467b-802c-ffa0fe0ce172.png)


### Technologies Used:
	
- Python 3.10
- scikit API

### Starting project:

1. Please Clone project: `git clone {url}`
2. Please Install necessery Python libraries with pip 
	e.g.	WINDOWS: `pip install -U scikit-learn`
			LINUX:	`pip3 install -U scikit-learn`
3. When you have needed application and libraries installed 
	
Please unzip these files:

	`mnist_test.zip --> mnist_test.csv`
	`mnist_train.zip --> mnist_train.csv`

4. Please run main.py - the whole process is done automatickly

During the first main.py run 
(The first part of program),
- It will import data from files:

	`mnist_test.csv`
	`mnist_train.csv`

( 2 images are printed out just for verification purpose, please close one by one to go to the next steps)
- Process them accordingly and store data in binary file, 
  so that data could be loaded faster then from csv
  
	`pickled_mnist.pkl`

	(Since the binary file is created, this program skip these first steps during each next run)
	
The second part of program
- Program loads data from `pickled_mnist.pkl`
- It will start multilayer perceptron testing basing on defined test parameters
	12 Test Cases are defined
		Please check code for test case definition
		New test cases could be added or the current one customized
- Testing Progress and all data are printed out and also saved in file
e.g.
```
TEST 1.1
	Algorythm Optymalizacji: sgd , Funkcja aktywacji: relu , Liczba iteracji: 4
	Wielko??ci macierzy (ilo??c wartsw: 1 , ilo???? neuron??w: (30,) )
	Iteration 1, loss = 0.34969233
	Iteration 2, loss = 0.18941272
	Iteration 3, loss = 0.15737218
	Iteration 4, loss = 0.13680897
	Zbior trenujacy dopasowanie: 0.965467
	Zbior testowy   dopasowanie: 0.958200
```
All test results beyond of loss function are saved in file which is created automaticaly based date and time
e.g. 
	`01022022_102042.txt`
	
### Description of files:
- mnist_test.zip - contain mnist_test.csv
- mnist_train.zip - contain mnist_train.csv
- main.py
- readme.md

### Bibliography
	
http://yann.lecun.com/exdb/mnist/

https://scikit-learn.org/
