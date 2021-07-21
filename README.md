# hugs-and-kisses
project to analyse sets of raw data using  neural network machine learning and python data visualisation tools
This was an assignment I worked on for my "Machine Learning and Neural Networks" module during my course in maynooth.

Train.txt contains a dataset of 11 columns, 10 raw data, and 1 with a binary designator
  This set is used to train the machine learning model
  
Test.txt contains a dataset of 10 columns, all raw data
  The generated model is run on this set to predict the binary designators
  
Key.txt cotains the actual binary designators of the Test.txt rows
  This set is used to measure the accuracy of the model's predictions
  
a.txt is an array, which when the Test.txt data is multiplied by its inverse transforms the data into coordinate pairs for graphing

make.py is the final program which generates a machine learning model to process the test.txt data. 
The program then creates two graphs from the coordinate pairs generated from test.txt, one with the key.txt designators, and one with the model's designators.

Program execution can be found in READMESUBMISSION.md
