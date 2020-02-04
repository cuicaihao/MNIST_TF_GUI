# MNIST_TF_GUI README File
Author: Caihao Cui
Location: Melbourne, VIC, AU.

Description: This repository demonstrates one python program including training, testing, and retraining ML/DL models with TensorFlow and Keras with an extra GUI by Qt5.
 
Dependencies: install Anaconda python 3, TensorFlow, Keras and all the package related. 
TF2.0 works but with few warninig. 
 
## Train the model
Use the **Train_MNIST_Model.py** to build the model, model will be saved at **./models/xxx.xxx.xxx.h5**, the training histroy (acc, loss) will be saved as **history.pickle**.  

## Test the model
Use the **Test_MNIST_Model.py** to test the model and compare the training and testing accuracy. 

## Retrain the model
Use the **Retrain_MNIST_Model.py** to load the trained model and retrain it with different optimization methods.

## Play with GUI
Run **GUI.py** or **MNIST_Window.py** to play with the interactive desktop application. 
### Sample images:
You can change the details by editing the `class of MNIST_Window` 
 
![Input Check 2](./images/GUI_input_2.PNG)

Sample Images:
![Test 0](./images/GUI_Test_Input_0.PNG)
![Test 1](./images/GUI_Test_Input_1.PNG)
![Test 2](./images/GUI_Test_Input_2.PNG)
![Test 3](./images/GUI_Test_Input_3.PNG)
![Test 4](./images/GUI_Test_Input_4.PNG)
![Test 5](./images/GUI_Test_Input_5.PNG)
![Test 6](./images/GUI_Test_Input_6.PNG)
![Test 7](./images/GUI_Test_Input_7.PNG)
![Test 8](./images/GUI_Test_Input_8.PNG)
![Test 9](./images/GUI_Test_Input_9.PNG)

## Reference
This repository is inspried by the CSDN Blog: https://blog.csdn.net/u011389706/article/details/81460820.