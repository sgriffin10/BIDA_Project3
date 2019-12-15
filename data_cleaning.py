import pandas as pd

df = pd.read_csv("car_insurance_claim.csv", sep = ',') #reads csv
print(df) #check if prints

# df = df.drop(df.columns[32:49], axis = 1)


# #### Cleaning Code #####
df = df.drop(columns=["HOMEKIDS","KIDSDRIV","BLUEBOOK","TIF","RED_CAR"]) #drops the 6 useless columns
print(df)
# df.to_csv("Final_Project/Data/US_Accidents_May19_cleaned_2.csv")

# count = 0 
# for column in df.columns[:]: 
#     if df[column].isnull().any() == True: 


df = df.dropna() #if you want to remove all rows with missing values
print(df.isnull().any()) #check
print(df) #visual of df after dropped rows

# print(df.types)
for column in df.columns[:]:
    df[column] = df[column].astype(str) 
    df[column] = df[column].str.replace('$', '')
print(df)
print(df.dtypes)

df.to_csv("car_insurance_claim.csv_noNAs_1.csv")
