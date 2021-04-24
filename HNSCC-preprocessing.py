# Importing the libraries
import pandas as pd

#show all data from all columns
pd.set_option('max_columns', None)

# Importing the dataset
dataset = pd.read_csv('HNSCC_Clinical_data.csv')

#patient information
patient_info = dataset[["Age", "Sex", "Smoking History", "Current Smoker", "HPV status"]]
patient_info = pd.get_dummies(patient_info, columns = ["Sex", "HPV status"])


#temporal Data (time between different analyses and follow-up)
time = dataset[[
                "Time from preRT image to start RT (month)",
                "Time from RT stop to follow up imaging (months)",
                "Follow up duration (month)",
                "Survival  (months)",
                "Disease-free interval (months)"
                ]]
treatment = dataset[[
                "RT Total Dose (Gy)",
                "Unplanned Additional Oncologic Treatment",
                "Number of Fractions",
                "Received Concurrent Chemoradiotherapy?",
                "CCRT Chemotherapy Regimen"
                ]]
treatment = pd.get_dummies(treatment, columns = ["Unplanned Additional Oncologic Treatment", 
                                                 "Received Concurrent Chemoradiotherapy?", 
                                                 "CCRT Chemotherapy Regimen" ])

#clinical measurements
pre_treatment = dataset[[
                            "Pre-RT L3 Skeletal Muscle Cross Sectional Area (cm2)",
                            "Pre-RT L3 Adipose Tissue Cross Sectional Area (cm2)",
                            "Pre-RT L3 Skeletal Muscle Index (cm2/m2)",
                            "Pre-RT L3 Adiposity Index (cm2/m2)",
                            "Pre-RT CT-derived lean body mass (kg)",
                            "Pre-RT CT-derived fat body mass (kg)",
                            "BMI start treat (kg/m2)"
                        ]]

post_treatment = dataset[[
                            "Post-RT L3 Skeletal Muscle Cross Sectional Area (cm2)",
                            "Post-RT L3 Adipose Tissue Cross Sectional Area (cm2)",
                            "Post-RT L3 Skeletal Muscle Index (cm2/m2)",
                            "Post-RT L3 Adiposity Index (cm2/m2)",
                            "Post-RT CT-derived lean body mass (kg)",
                            "Post-RT CT-derived fat body mass (kg)",
                            "BMI stop treat (kg/m2)",
                            "Site of recurrence (Distal/Local/ Locoregional)"
                            
                        ]]
post_treatment = pd.get_dummies(post_treatment, columns = ["Site of recurrence (Distal/Local/ Locoregional)"])

outcome = dataset[[
                    "Disease Specific Survival Censor"
                 ]]


def print_info():
    data = [patient_info, pre_treatment, treatment, post_treatment, outcome]
    for d in data:
        print(d.info())
        print(d.shape)
        print(d.head())
        print()
        
def get_data():
    data = [patient_info.reset_index(drop=True), treatment.reset_index(drop=True), pre_treatment.reset_index(drop=True), post_treatment.reset_index(drop=True), outcome.reset_index(drop=True)]
    df = pd.concat(data, axis=1, join="inner")
    return df

#print(dataset.info())
#print(outcome.info())

ann_data = get_data()
print(ann_data.info())
#ann_data.to_csv('ann_dataset_treatment1.csv', index=False)
