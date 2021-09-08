# Capstone: Prediction of Heart Failure 

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
Its accuracy was ~0.86 that was larger than the best accuracy ~0.7 of logistic regression with HyperDrive.

The VotingEnsemble model was deployed for the consumption.


## Project Set Up and Installation
This project was completed in the defalut setting of the AzureML.

## Dataset

### Overview
The dataset called Heart Failure clinical records was obtained from Kaggle. 

It has 12 features such as cardiovascular age, disease, hypertension, diabetes and so on. 
DEATH_EVENT was a predicted attribute for our classification model. 

Input features are the following:
age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time

Output is DEATH_EVENT.

### Task
The goal was to build a binary classification model of predicting "DEATH_EVENT". 
The model was developed with using the other 11 input features.

### Access
The data of heart_failure_clinical_records_dataset.csv was uploaded to Datastore in Azure with the name of "HeartFailurePrediction".

## Automated ML

The AutoML for the binary classification was run with the primary metric of accuracy. 
The cross validation was applied to avoid the overfitting. 
The running time was limited to be within 30 minutes. 
The concurrent iteration was applied. 
The ONNX model save mode was enabled. 
Early stopping policy was enabled. 
Automatic featurization was applied.  

Code block of AutoML configuration is shown below.  
![AutoML_1](./images/01_automl_settings.png)  

### Results

The AutoML found out the best performance model of VotingEnsemble with the accuracy of 0.86. 
It consists of 7 sub-models which are weighted by a constant value 1/7~0.14

Screenshots of the `RunDetails` widget display how the AutoML make a progress in finding out the best model.  
![Rundetail_1](./images/automl_rundetails_01.png)
  
The accuracy of each tested ML model is shown in terms of Run ID. 
The best accuracy is described by a orange solid line.  
![Rundetail_2](./images/automl_rundetails_02.png)

The best performance model of VotingEnsemble is described with its accuracy and ID below.  
![Rundetail_3](./images/automl_bestmodel_votingensemble_03.png)


## Hyperparameter Tuning

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

The best model of logistic regression algorithm was determined with the two hyperparameters of C~0.94 and max-iter=50.


RunDails of HyperDrive show the progress of the HyperDrive run.  
![HyperDrive_Rundetail_1](./images/hyperdrive_rundetails_01.png)  

The accuracy of each tested ML model with its pair of hyperparameters is shown in terms of Run ID. 
The best accuracy is described by a orange solid line.  
![HyperDrive_Rundetail_2](./images/hyperdrive_rundetails_02.png)  

Each point represents an accuracy by color for a give pair of (C, max-iter). 
The maximum value of the accuracy is about 0.7 shown to be yellow.  
![HyperDrive_Rundetail_3](./images/hyperdrive_rundetails_03.png)

The Best model of logistic regression was determined with two hyperparameters of C=~0.94 and max-iter=50. 
Its accuracy is about 0.7.  
![HyperDrive_Rundetail_](./images/hyperdrive_bestmodel_04.png)

The Best model of logistic regression is shown in Azure ML Studion.  
![HyperDrive_Rundetail_](./images/hyperdrive_bestmodel_05.png)

## Model Deployment

The best model of VotingEnsemble algorithm was determined by AutoML. 
This model was deployed as a web service.

The example of querying the endpoint is provided with endpoint.py, 
where scoring_uri and key must be updated. 
Here is a command: 
python ./endpoint.py

## Future directions
The neural network will be included in the AutoML to see if the accuracy is improved.  

I will run HyperDrive after adjusting the range of C to be between 0.9 and 1.5. 
This will provide us with the lower bound of the accuracy that the model can be considered to be trained in the future.  

The AutoML training pipeline will be redesigned to interact with other services. 
I will make the pipeline respond to the anomalies in the streaming data.


## Screen Recording
a link to a screen recording of the project in action:
https://youtu.be/Pyx4JLoXeA4

Here are some extra statements to support my screencast.  

**Demo of the deployed model**  
The best performant model of VotingEnsemble algorithm was deployed as a web service in this project. 
This model was registered. 
The local environment was copied from the workspace. 
The Azure Container instance (AciWebservice) was used with 1 cpu and 1 GB memory. 
Authorization and Application insights were programmatically enabled.

The model is successfully deployed to be shown below.  
![Success in deployment](./images/model_deployment_success_08.png)

The model is working.  
![Success in deployment](./images/11_model_deployed_active.png)  

**Demo of a sample request sent to the endpoint and its response**  
Get scoring_uri. 
Make sure that key is available, 
because authentication was enabled in the deployment.
Data was set up to be a dictionary with the key "data" and its value list with two elements. 
That data is converted to a JSON format and stored in a json file (data.json). 
A dictionary headers is defined to include content-type and authorization key. 
These three values (scoring_uri, data, and headers) are used as input for requests.post method.  

This is the screenshot of the query and its result.  
![Output of the deployed model](./images/09_output_deploymentmodel_1.png)


