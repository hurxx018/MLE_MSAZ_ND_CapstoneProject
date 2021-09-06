*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Capstone: Prediction of Heart Failure 

*TODO:* Write a short introduction to your project.
Cardiovascular diseases (CVDs) are the number 1 cause of death globally. 
CVDs commonly cause heart failures. 
Early detection of heart failure is one way of addressing the problem. 
Here we used machine learning approach to build a classification model relying on a Heart Failure prediction dataset. 
This dataset is available in Kaggle. 
The dataset consists of 12 features that are cardiovascular disease, hypertension, diabetes and so on.

Here, we applied two methods: HyperDrive and AutoML. 
Logistic regression algorithm was run with HyperDrive where two hyperparameters of C and max-iter were tested.
AutoML was run with random parameter sampling and bandit early stopping policy. 

The best model of VotingEnsemble was determined by the AutoML run. 
Its accuracy was ~0.91 which was larger than the best accuracy ~0.7 of logistic regression with HyperDrive.

The VotingEnsemble model was deployed for the consumption.


## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The dataset called Heart Failure clinical records was obtained from Kaggle. 

It has 12 features such as cardiovascular disease, hypertension, diabetes and so on. 
DEATH_EVENT was a predicted attribute for our classification model. 

age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time,DEATH_EVENT

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
The goal was to build a classification model of predicting "DEATH_EVENT".

### Access
*TODO*: Explain how you are accessing the data in your workspace.
Data of heart_failure_clinical_records_dataset.csv was uploaded to Datastore in Azure with the name of "HeartFailurePrediction".

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Logistic regression was used to build the binary classification model that is the simplest approach. 
Logistic regression allowed us to a kind of banchmarking result that we can estimate the accuracy of our classification model.
Two hyperparameters of the inverse of regularization (C) and the maximum number of iterations (max-iter) were chosen to run HyperDrive.
C is chosen to control the effective size of parameters.
It was randomly selected from a uniform distribution of (0.1, 1.0).
max-iter was tested to determine how long the training should be done to find an optimal model.
It was randomly selected from four values of 50, 100, 500, and 1000.

For each run, the two parameters were randomly selected (RandomParameter Sampling).
For the early stopping policy, we chose BanditPolicy that seems to be efficient of choosing the early stopping point.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The best model of logistic regression algorithm was determined with C=~ and max-iter=####.

I will run HyperDrive after reducing the range of C. 
For example, C is one choice.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

RunDails of HyperDrive
()

Best model of logistic regression 
()

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The best model of VotingEnsemble algorithm was determined by AutoML. 
This was deployed.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
