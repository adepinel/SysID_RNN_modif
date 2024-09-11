# **Single RNN Model for System Identification**

This code implements a single RNN model for system identification of a 3-tank system. The model is trained on input-output data and is used to predict the output of the system for validation data.


## **Usage**

1. Clone the repository to your local machine.
2. Navigate to the `src` directory.
3. Run the `main.py` file using the command `python main.py`.

## **Inputs**

The code reads input-output data from a MATLAB file `dataset_sysID_3tanks_final.mat`. The file contains the following variables:

- `dExp`: Input data for training.
- `yExp`: Output data for training.
- `dExp_val`: Input data for validation.
- `yExp_val`: Output data for validation.
- `Ts`: Sampling time.

## **Outputs**

The code outputs the following:

- Plots of the loss over epochs, and the predicted output for each tank for the training and validation data.
- A MATLAB file `data_singleRNN.mat` containing the predicted output for the validation data.

## **Code Structure**

The code is structured as follows:

1. Import necessary libraries.
2. Load data from file.
3. Initialize input and output tensors.
4. Initialize RNN model.
5. Define loss function and optimization method.
6. Train the RNN model.
7. Get RNN output for validation data.
8. Calculate loss for validation data.
9. Plot loss over epochs and predicted output for each tank for training and validation data.
10. Save predicted output and validation data to file.

## **License**

This code is licensed under the MIT License.