import pandas as pd

df = pd.read_csv("US_Accidents_May19_cleaned_2.csv", sep = ',') #reads csv
# print(df) #check if prints

# df = df.drop(df.columns[32:49], axis = 1)

# #### Cleaning Code #####
# df = df.drop(columns=["TMC","Source","End_Lat","End_Lng","Number","Country"]) #drops the 6 useless columns
# # print(df)
# df.to_csv("Final_Project/Data/US_Accidents_May19_cleaned_2.csv")

# count = 0 
# for column in df.columns[:]: 
#     if df[column].isnull().any() == True: 


df = df.dropna() #if you want to remove all rows with missing values
print(df.isnull().any()) #check
print(df) #visual of df after dropped rows
# df.to_csv("US_Accidents_May19_cleaned_noNAs_1.csv")
