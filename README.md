This project is a joint effort of Mark Lavin (human) and Gemini (Google AI) and depends on the FreeStyle Libre 3 Continuous Glucose Monitor (CGM) made by Abbot Laboratories.  Its objective is to develop a system based on CGM and other data that can model and predict glucose levels in Canines.

# Introduction
A couple of months ago, our dog Poppy, a fifteen year old mixed breed, developed diabetes, presumably based on inability to synthesize enough Insulin to modulate blood glucose levels, so that her glucose levels were consistent higher than advised by Veteranarians.  We found that establishing a regimen of regular meals (including interventions or "minimeals") in order to stabilize Poppy's glucose levels in the healthy range of 150 mg/dL and 250 mg/dL was very difficult, and we were looking for guidance on when and how much we fed Poppy and how much insulin and exercise we give her.  In addition, I was taking a course from Coursera/Johns Hopkins University on Nvidia CUDA programming and was looking for a capstone project.  Fortunately, our Vet advised fitting Poppy with a CGM so that we could see her glucose levels over time in a smartphone app.  The CGM data are a key element in helping to control Poppy's glucose levels, and since these are readily available through a Web API, we had the basis for our project.

# Objectives of the Project
We had two objectives in mind:
## Model Training
The first objective was to train a model that could represent the CGM data we were observing.  Rather than building an *ad hoc* application, we decided to use Machine Learning, where we would use observed data -- Poppy's glucose levels and "records" of events like meals and exercise.  That objective has been met and we have a working model of Poppy's glucose levels that is extremely accurate, with a Root Mean Squared Error less than 5 mg/dL.

## Model-based Prediction
The second objective was to use the trained model to make predictions about Poppy's glucose levels.  Specifically, we used the most recently-acquired CGM data and "records" data for the last six hours to predict a two-hour window of expected glucose levels.  That objective has been partially met in that we have fully implemented a "Prediction Loop" and the results look "plausible", with the predicted results following "reasonably" the subsequently-observed data; we have planned but not yet implemented additional programs to compare the results of the predictions with the "actuals" corresponding to the predictions.

# Implementation
The project was implemented with a Python 3 program on a Lenovo P1 laptop running Windows 11 and equipped with a T100 GPU that was programmable in both CUDA and indirectly through the use of the Python PyTorch package that provides Neural Net capabilities implemented via CUDA on the GPU.

## Modelling with LSTM
To first order, glucose level prediction is a standard time-series problem based on the CGM data and "record" data, where we're given a "context" time series (the previous six hours of CGM data) and want to extend the time series with two hours of *predicted* CGM data.  There are many machine learning techniques for predicting time-series, but we used Artificial Neural Networks (ANNs), specifically, Long Short-Term Memory (LSTM) recurrent neural networks.  The LSTM consists of a series of modules that the sequential input data flow through and that exchange information to predicate the output, namely a series of predicted glucose levels.  Using PyTorch, we could implement an ANN consisting of two LSTMs followed by a fully-connected network that produces the output "predicted continuation" of the time series.
### Data Input
#### CGM data API
#### "Record" data API
### Machine Learning
#### Model Structure
#### Model Training
## Prediction

# Usage
## Code Manufacture
## Training
## Prediction

# Implications for CUDA and GPU programming
# Future Work
