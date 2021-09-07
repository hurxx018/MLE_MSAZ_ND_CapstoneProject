import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = ''
# If the service is authenticated, set the key or token
key = ''

# Two sets of data to score, so we get two results back
data = {"data":
        [
            {
                "age" : "65", 
                "anaemia" : "0",
                "creatinine_phosphokinase" : "146",
                "diabetes" : "0", 
                "ejection_fraction" : "20",
                "high_blood_pressure" : "0",
                "platelets" : "162000",
                "serum_creatinine" : "1.3", 
                "serum_sodium" : "129",
                "sex" : "1",
                "smoking" : "1",
                "time" : "7"
            },
            {
                "age" : "55", 
                "anaemia" : "0",
                "creatinine_phosphokinase" : "146",
                "diabetes" : "1", 
                "ejection_fraction" : "20",
                "high_blood_pressure" : "0",
                "platelets" : "162000",
                "serum_creatinine" : "1.34", 
                "serum_sodium" : "129",
                "sex" : "0",
                "smoking" : "0",
                "time" : "6"
            },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


