import pandas as pd

dm_inputdf = pd.read_csv('node_data.csv')
dm_interval_input = ["churned", "account_length","area_code","number_customer_service_calls","number_vmail_messages",
                "total_day_calls","total_day_charge","total_day_minutes","total_eve_calls","total_eve_charge",
                "total_eve_minutes","total_intl_calls","total_intl_charge","total_intl_minutes","total_night_calls",
                "total_night_charge","total_night_minutes"]

########################################################################################################################
# Open-Source Node From Here
########################################################################################################################

import json
import requests
import pandas as pd

# The API expects floats for all numeric variables
dm_inputdf[dm_interval_input] = dm_inputdf[dm_interval_input].apply(pd.to_numeric, errors='coerce', axis=1)

# Score data
response = requests.post(
    "http://cdswserver:8080/api/altus-ds-1/models/call-model",
    headers={"Content-Type": "application/json"},
    data=json.dumps({
        "accessKey": "**** secret cdsw model access key ****",
        "request": dm_inputdf.drop(['churned', '_dmIndex_', '_PartInd_', 'M_FILTER'], axis=1).to_dict('records')
    }))

# Populate scores dataframe
dm_scoreddf = pd.DataFrame.from_records(response.json().get("response"))
dm_scoreddf.drop(["churned"], axis=1, inplace=True)

########################################################################################################################
# Open-Source Node Until Here
########################################################################################################################

print(dm_scoreddf)