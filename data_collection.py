## Data used in the project is final_data.csv available at
# https://drive.google.com/open?id=17m_myHlEYv4z06xkdEYQh3nN3kgYn-xk


import pandas as pd
import requests
import json


df = pd.read_csv("NYPD_Complaint_Data_Historic.csv")

complaint_num = ['CMPLNT_NUM']
complaint_date = [ 'CMPLNT_FR_DT', 'CMPLNT_FR_TM']
complaint_locations = ['Latitude','Longitude','ADDR_PCT_CD', 'BORO_NM','PATROL_BORO']
crime_details = ['KY_CD','PD_CD','LAW_CAT_CD'] #,'OFNS_DESC','CRM_ATPT_CPTD_CD','PD_DESC']
surrounding_desc = ['PARKS_NM','PREM_TYP_DESC']
suspect_data = ['SUSP_AGE_GROUP','SUSP_RACE','SUSP_SEX']
victim_data = ['VIC_AGE_GROUP','VIC_RACE','VIC_SEX']

df_locations = df[complaint_num + complaint_locations + surrounding_desc]
df_crime_details = df[complaint_num + suspect_data + victim_data]
df_data = df[complaint_num + complaint_date + complaint_locations + crime_details + surrounding_desc + suspect_data + victim_data]

# blocks['Tract'] = blocks.BlockCode // 10000
# The block code takes the form XXYYYZZZZZZAAAA where X=State, Y=County, Z=Tract, A=Block. 
# The associated tract code is identical but without the block part.
df_census = pd.read_csv("nyc_census_tracts.csv")

df_crime_data = df_data.iloc[:60000]

count=5001
ans_list = []

URL = "https://geo.fcc.gov/api/census/area"
def get_block_code(row):
    x = row['Latitude']
    y = row['Longitude']
    global count
    global ans_list
    ans = -1
    params = {"lat" : str(x), "lon" : str(y), "format" : "json"}
    r = requests.get(url = URL, params = params)
    if count%100 == 0:
        print(count, r)
    if count%1000 == 0:
        ans_df = pd.DataFrame(ans_list)
        ans_df.to_csv('chkpt_'+str(count)+'.csv')
    count+=1
    if r.status_code == 200:
        ans =  r.json()['results'][0]['block_fips']
    ans_list.append(ans)
    

df_data_sample_1 = df_crime_data
df_data_sample_1['BlockCode'] = df_data_sample_1.apply(get_block_code, axis=1)

df_ds1 = df_crime_data
df_ds1['CensusTract'] = df_ds1.apply(lambda row : int(row['BlockCode'])//10000,axis=1)

df_merged = pd.merge(df_ds1, df_census, on='CensusTract')
df_merged.to_csv('final_data.csv')
