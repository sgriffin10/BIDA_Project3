import pandas as pd

#Import CSV to dataframe
df = pd.read_csv("car_insurance_claim.csv", sep = ',') 
print(df) 



# Drop Irrelevant Columns 
df = df.drop(columns=["ID","BIRTH","HOMEKIDS","TIF","RED_CAR"]) 
# print(df)

#Drops rows with missing values
df = df.dropna() 
print(df.isnull().any()) 
print(df) 

#Removes "$" and "," from rows
for column in df.columns[:]:
    df[column] = df[column].astype(str) 
    df[column] = df[column].str.replace('$', '')
    df[column] = df[column].str.replace(',', '')
# print(df)
# print(df.dtypes)


#Creates Dummy Variables (either 0 or 1) for variables with only 2 choices
dummy_req_list = ["PARENT1","MSTATUS","GENDER","CAR_USE","REVOKED","URBANICITY"]
for column in dummy_req_list:
    df[column] = pd.get_dummies(df[column])
# print(df)
# print(df.dtypes)

#Creates Categorical Variables (0,1,....9,10,etc.)
cat_req_list = ["EDUCATION","OCCUPATION","CAR_TYPE"]
for column in cat_req_list:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes
# print(df)
# print(df.dtypes)

# Changes Data Types from Object to Int
for column in df.columns[:]:
    if df[column].dtypes == "object":
        # df[column] = df[column].astype(str) 
        df[column] = df[column].astype(float)
# print(df) 
# print(df.dtypes)

#Saves to a new file
# df.to_csv("car_insurance_claim_noNAs_2.csv")
