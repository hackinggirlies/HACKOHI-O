import pandas as pd
import requests
from llamaapi import LlamaAPI
import json
api_url = "https://llama-api.com"
api_token = "LA-bd5a523be4434864be0371ee34e190a3e5d4af335cc84fd28558ef22cc29dac0"  # Replace with your actual token
llama = LlamaAPI(api_token)
df = pd.read_csv("CORE_HackOhio_subset_cleaned_downsampled 1.csv")
print(df.head()) 
#DATETIME_DTM	Datetime, date and time when observation was recorded
#PNT_NM	Point name, the safety criteria being assessed. Forms a primary key when combined with OBSRVN_NB (observation number)
#QUALIFIER_TXT	Qualifier text, list of predetermined observations chosen by the reviewer based on the point being assessed
#PNT_ATRISKNOTES_TX	Point at-risk notes text, comments left by observer regarding unsafe conditions they found
#PNT_ATRISKFOLWUPNTS_TX	Point at-risk follow up notes text, recommended remediation for at-risk conditions observed

date_time = df['DATETIME_DTM']
#Date time format looks like this: 3/15/2023 11:01
safety_criteria = df['PNT_NM']
#format looks like this: Equipment - Tools &Equipment
#a few of these are formatted as questions like this: 
#Did you recognize additional sprain or strain hazards that had not already been recognized and mitigated? If so, please select or describe these hazards in the At-Risk notes. 1
qualifier = df['QUALIFIER_TXT']
#examples: Assistance requested, Personal voltage detector
comments = df['PNT_ATRISKNOTES_TX']
followups = df['PNT_ATRISKFOLWUPNTS_TX']

def get_conversation_history(username):
    headers = {"Authorization": f"Bearer {api_token}"}
    #response = requests.get(f"{api_url}/conversations/{username}", headers=headers)
    response = requests.post(f"{api_url}/chat/completions", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def start_chat():
    username = "user65f7c6e8053d4e3a"
    conversation_history = get_conversation_history(username)
    print(conversation_history)

if __name__ == "__main__":
    start_chat()