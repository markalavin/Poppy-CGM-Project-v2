This project is a joint effort of Mark Lavin (human) and Gemini (Google AI) and depends on the FreeStyle Libre 3 Continuous Glucose Monitor (CGM) made by Abbot Laboratories.  Its objective is to develop a system based on CGM and other data that can model and predict glucose levels in Canines.

# 1. Introduction
A couple of months ago, our dog Poppy, a fifteen year old mixed breed, developed diabetes, presumably based on inability to synthesize enough Insulin to modulate blood glucose levels, so that her glucose levels were consistent higher than advised by Veteranarians.  We found that establishing a regimen of regular meals (including interventions or "minimeals") in order to stabilize Poppy's glucose levels in the healthy range of 150 mg/dL and 250 mg/dL was very difficult, and we were looking for guidance on when and how much we fed Poppy and how much insulin and exercise we give her.  In addition, I was taking a course from Coursera/Johns Hopkins University on Nvidia CUDA programming and was looking for a capstone project.  Fortunately, our Vet advised fitting Poppy with a CGM so that we could see her glucose levels over time in a smartphone app.  The CGM data are a key element in helping to control Poppy's glucose levels, and since these are readily available through a Web API, we had the basis for our project.

# 2. Objectives of the Project
We had two objectives in mind:
## 2.1. Model Training
The first objective was to train a model that could represent the CGM data we were observing.  Rather than building an *ad hoc* application, we decided to use Machine Learning, where we would use observed data -- Poppy's glucose levels and "records" or "reports" of events like meals and exercise.  That objective has been met and we have a working model of Poppy's glucose levels that is extremely accurate, with a Root Mean Squared Error less than 5 mg/dL.

## 2.2. Model-based Prediction
The second objective was to use the trained model to make predictions about Poppy's glucose levels.  Specifically, we used the most recently-acquired CGM data and "records" data for the last six hours to predict a two-hour window of expected glucose levels.  That objective has been partially met in that we have fully implemented a "Prediction Loop" and the results look "plausible", with the predicted results following "reasonably" the subsequently-observed data; we have planned but not yet implemented additional programs to compare the results of the predictions with the "actuals" corresponding to the predictions.

# 3. Implementation
The project was implemented with a Python 3 program on a Lenovo P1 laptop running Windows 11 and equipped with a T100 GPU that was programmable in both CUDA and indirectly through the use of the Python PyTorch package that provides Neural Net capabilities implemented via CUDA on the GPU.

## 3.1. Modelling with LSTM
To first order, glucose level prediction is a standard time-series problem based on the CGM data and "record" data, where we're given a "context" time series (the previous six hours of CGM data) and want to extend the time series with two hours of *predicted* CGM data.  There are many machine learning techniques for predicting time-series, but we used Artificial Neural Networks (ANNs), specifically, Long Short-Term Memory (LSTM) recurrent neural networks.  The LSTM consists of a series of modules that the sequential input data flow through and that exchange information to predicate the output, namely a series of predicted glucose levels.  Using PyTorch, we could implement an ANN consisting of two LSTMs followed by a fully-connected network that produces the output "predicted continuation" of the time series.

### 3.1.1. Data Input
#### CGM data API
The CGM data are first acquired from the Abbot LibreView server, downloaded as .csv files, one for each 5-minute glucose sample.  The CGM data are continuously and automatically updated by the Libre 3 app, and accessed by the Python ```pylibrelinkup``` package available through ```pip```.
#### "Record" data API
*In principle*, the record data (meals, insulin, etc.) could be gathered from the Libre 3 app's "Report" capability that allows a user to report these events along with measure information like "units of insulin".  For now, we rely on a data chain of handwritten notes -> .csv files -> dataframes -> tensors where it's the user's responsibility to do the data entry.  The required manual intervention is a weak link in our project and will hopefully be addressed later.
#### Input Tensors
The last step in the data chain, whether for training or prediction, is to merge the CGM and Record data aligned in the time axis.  These merged data are then converted to a PyTorch *tensor* that consists of one row per input sample (approximately 19,000 so far) and seven columns ```['Glucose', 'Insulin', 'Meal', 'Minimeal', 'Karo', 'Sin_T', 'Cos_T']```, where "minimeal" and "karo" indicate the amount (currently fixed) of food or karo (sugar) syrup given to Poppy according to a report.   'Sin_T' and 'Cos_T' are the sin and cosine of the time of day (in hours since midnight) that the sample was taken.  All quantities get normalized during the dadta preparation to lie in the range [0.0, 1.0].  Note that for the vast majority of samples, there is only glucose data and thus all columns except 'Glucose', 'Sin_T' and 'Cos_T' are 0s.
### 3.1.2. Machine Learning
The LSTM model that predicts future glucose levels (for two hours or 24 five-minute samples) based on the preceding six hours or 72 samples glucose levels and record data (meals, insulin, etc.) uses standard neural network training techniques:  Input data are broken up into 32 sample *batches* and then two hundred batches are processed.  For each batch, the 72 "input" or "context" samples are used to predict the subsequent 24 samples, the result is compared with the input "actuals", and the amount and gradient of the error between them is used to adjust the internal state of the model (weights associated with the elements of the LSTM neural net) until minimum error is achieved.
## 3.2. Prediction
Prediction uses similar data and methods to the model training:  Given a prediction time (defaults to "now" but can be set earlier), the program first gathers the data for prediction:  the 72 samples (six hours) preceding the prediction time, then uses the model to predict the subsequent output of 24 samples (two hours) following the prediction time.  The Prediction function prints and plots the results and saves them away for subsequent validation:  waiting for two hours and then comparing the "actuals" of Poppy's glucose levels with the predictions; this capability will be added in the future.

# 4. Usage
The project has been set up so that it could be viewed and even tried out by others.  There is one qualification:  At the moment, the only repository of CGM data I have access to is Poppy's.  I have set things up so that during training or prediction, you'll be accessing Poppy's data, which run from November 2025 through early January 2026.  In any case, if you run into problems trying to set up or use the project, do no hesitate to contact me via the GitHub project 
## 4.1. Code Manufacture
I developed the code (with extensive help from Gemini) using the PyCharm Python IDE.  You should be able to use any Python environment such as Jupyter or even just a stand-alone interpreter.  Here are the steps to follow (this is on Windows; adapt as needed for Linux, MacOS, etc.):
1.  Set up a directory to hold the project.
2.  
## 4.2. Training
## 4.3. Prediction
![](https://github.com/markalavin/Poppy-CGM-Project/blob/main/data/Poppy_Forecast_2026_01_16_20_35.png)

# 5. Implications for CUDA and GPU programming

# 6. Future Work

# Appendix A:  Project Contents
<dl>
  <dt><code>data/Poppy CGM.csv</code></dt>
  <dd>Training CGM data</dd>
  <dt><code>data/Poppy Reports.csv</code></dt>
  <dd>Training Reports data</dd>
  <dt><code>data/prediction_history.csv</code></dt>
  <dd>Log of all Predictions to date</dd>
  <dt><code>srt/Application_Parameters.py</code></dt>
  <dd>Constant definitions</dd>
  <dt><code>src/Check_Input_Tensors.py</code></dt>
  Construct and validate Tensor containing all training data -- CGM and "Records"
  <dd></dd>
  <dt><code>src/Checks.py</code></dt>
  <dd>Check availability of GPU.</dd>
  <dt><code>src/Create_Windows.py</code></dt>
  <dd>Gather CGM and Record data for Training</dd>
  <dt><code>src/Get_Prediction_Data.py</code></dt>
  <dd>Gathers CGM data from <code>pylibrelinkup</code> and Record data from user input and combine for Prediction.</dd>
  <dt><code>src/Logging.py</code></dt>
  <dd>Saves result of Predictions in log CSV file</dd>
  <dt><code>src/Merge_Poppy_data.py</code></dt>
  <dd>Combines CGM and Record data using Pandas' <code>merge...asof</code> capability</dd>
  <dt><code>src/Model_Architecture.py</code></dt>
  <dd></dd>
  <dt><code>src/Predict.py</code></dt>
  <dd>Predict glucose data after "now" or specified time using information gathered from <code>pylibrelinkup</code> and user input.</dd>
  <dt><code>src/Process_CGM_Data.py</code></dt>
  <dd>Processes and analyzes CGM data from .csv file</dd>
  <dt><code>src/Process_Report_Data.py</code></dt>
  <dd>Processes Report data from .csv file</dd>
  <dt><code>src/Train_LSTM.py</code></dt>
  <dd>Perform training on LSTM model based on input tensors</dd>
  <dt><code>src/poppy_model_best.pth.saved</code></dt>
  <dd>Current saved model of Poppy's glucose behavior</dd>
</dl>

