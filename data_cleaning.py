import pandas as pd

df = pd.read_csv("car_insurance_claim.csv", sep = ',') #reads csv
print(df) #check if prints

# df = df.drop(df.columns[32:49], axis = 1)


# #### Cleaning Code #####
df = df.drop(columns=["BIRTH","HOMEKIDS","BLUEBOOK","TIF","RED_CAR"]) #drops the 6 useless columns
print(df)
# df.to_csv("Final_Project/Data/US_Accidents_May19_cleaned_2.csv")

# count = 0 
# for column in df.columns[:]: 
#     if df[column].isnull().any() == True: 


df = df.dropna() #if you want to remove all rows with missing values
print(df.isnull().any()) #check
# print(df) #visual of df after dropped rows

# print(df.types)
for column in df.columns[:]:
    df[column] = df[column].astype(str) 
    df[column] = df[column].str.replace('$', '')
    df[column] = df[column].str.replace(',', '')
# print(df)
# print(df.dtypes)


#Creating Dummy Variables
dummy_req_list = ["PARENT1","MSTATUS","GENDER","CAR_USE","REVOKED","URBANICITY"]
for column in dummy_req_list:
    df[column] = pd.get_dummies(df[column])
# print(df)
# print(df.dtypes)

#Creating Dummy Variables
cat_req_list = ["EDUCATION","OCCUPATION","CAR_TYPE"]
for column in cat_req_list:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes
print(df)
# print(df.dtypes)

# Change object to int
for column in df.columns[:]:
    if df[column].dtypes == "object":
        # df[column] = df[column].astype(str) 
        df[column] = df[column].astype(float)
print(df) 
print(df.dtypes)




df.to_csv("car_insurance_claim_noNAs_2.csv")
