# %%
import pandas as pd
import os
import numpy as np
# Show all dataframe columns
pd.set_option('display.max_columns', None)

# %%
video_survey_match = pd.read_excel("data/video_survey_match.xlsx")


# Dropn "new_" from the 'Name' column
video_survey_match['Name'] = video_survey_match['Name'].str.replace('new_', '')

# Split the 'Name' column into a list of strings and extract just the first element (file number)
video_survey_match['file_name'] = video_survey_match['Name'].str.split('_').str[0].astype(int)

video_survey_match['offset'] = video_survey_match['Name'].str.split('.mp4').str[0]
video_survey_match['offset'] = video_survey_match['offset'].str.split('_off').str[1].astype(int)/100

video_survey_match = video_survey_match.loc[:,['file_name','offset','ID', 'Name']]
# video_survey_match = video_survey_match.rename(columns={'offset': 'file_name'})

metadata = pd.read_csv("data/metadata.csv")

# Change column name
metadata = metadata.rename(columns={'file name': 'file_name'})

# Merge dataframes
metadata_merged = pd.merge(video_survey_match, metadata, on='file_name', how='inner')

metadata_cleaned = metadata_merged.drop(columns=['Name', 'file_name', 'time of event', 'time of alert', 'video path',
                                                  'event', 'bad video', 'time of first appearance', 'event time', 'crash occurance'])
# metadata_cleaned

# %%
df_long_CL = pd.read_csv("data/df_long.csv")

# Attention check
# df_long_CL_AC = df_long_CL[df_long_CL['While watching the video, which type of vehicle did you imagine yourself riding in?'] == df_long_CL['display'] + 1]
df_long_CL.shape

# %%
new_columns = [
    # Metadata Columns
    'start_date',
    'end_date',

    'reponse_type',
    'ip_address',
    'progress',

    'duration_seconds',
    'finished',

    'recorded_date',
    'response_id',
    'last_name',
    'first_name',
    'email',
    'data_reference',

    'latitude',
    'longitude',

    'distribution_channel',
    'user_language',

    # Attention Check Questions
    'q_attention_direction',      # "What best describes where your attention was directed..."
    'q_attention_check',     # "This is an attention check... select the number '2'..."

    # Main Survey Questions
    'q_vehicle_imagined',
    
    # Trust Questions
    'q_trust_taxi_skill',
    'q_safe_taxi_overall',
    'q_focus_taxi_ride',
    'q_trust_av_skill',
    'q_safe_av_overall',
    'q_focus_av_ride',

    # Demographic and Experience Questions
    'q_passenger_in_sdc',
    'q_primary_transport',
    'q_adas_usage',
    'q_tech_confidence',
    'q_driving_experience_years',
    'q_continent',
    'q_gender',
    'q_gender_other',
    'q_age',

    'q_prolific_id',
    'prolific_pid',

    # Other Columns
    'display_a',
    'display',
    'variable_header',
    'answer_value',
    'ID',
    'question'
]

df_long_CL.columns = new_columns

# %%
# Get a list of columns with constant values
# A column has a constant value if its number of unique values is 1
constant_columns = [col for col in df_long_CL.columns if df_long_CL[col].nunique() <= 1]

# Drop the columns with constant values
df_long_CL = df_long_CL.drop(columns=constant_columns)
df_long_CL.shape

# %%
df_demo = pd.read_csv("data\prolific_export_688279a66160545bd74e3613.csv")
df_demo = df_demo[['Participant id', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Language', 'Student status', 'Employment status']].copy()
df_demo.rename(columns={'Participant id': 'prolific_pid'}, inplace=True)
df_demo

# %%
df_long_CL = df_long_CL.merge(df_demo, on='prolific_pid', how='left')


# Get a list of columns with 'object' dtype
object_columns = df_long_CL.select_dtypes(include=['object']).columns

# Drop these columns from the DataFrame
# Excluding "ResponseID" and "ID"
excluded_columns = ['ResponseID', 'ID', 'Ethnicity simplified', 'Country of residence', 'Student status', 'Employment status']

object_columns = [col for col in object_columns if col not in excluded_columns]
df_cleaned = df_long_CL.drop(columns=object_columns)

# Combine Trust columns
df_cleaned['Combined_Trust'] = df_cleaned['q_trust_taxi_skill'].fillna(0) + df_cleaned['q_trust_av_skill'].fillna(0)

# Combine Safe columns
df_cleaned['Combined_Safe'] = df_cleaned['q_safe_taxi_overall'].fillna(0) + df_cleaned['q_safe_av_overall'].fillna(0) 

# Combine Focus columns
df_cleaned['Combined_Focus'] = df_cleaned['q_focus_taxi_ride'].fillna(0) + df_cleaned['q_focus_av_ride'].fillna(0)

# %%
drop_columns = ['duration_seconds', 'latitude', 'longitude',
                 'q_trust_taxi_skill', 'q_safe_taxi_overall', 'q_focus_taxi_ride',
                   'q_trust_av_skill', 'q_focus_av_ride', 'q_safe_av_overall']

df_cleaned = df_cleaned.drop(columns=drop_columns)

df_final = df_cleaned.merge(metadata_merged, on='ID', how='left')

drop_columns = ["Name", "file_name", "time of event", "time of alert", "video path", "event", "bad video", "time of first appearance", "event time", "crash occurance"]
df_final = df_final.drop(columns=drop_columns)
df_final = df_final.dropna(subset=['offset'])
df_final

# %%

for col in df_final.select_dtypes(include=['object']).columns:
    print(f"\nColumn: {col}")
    # print(df_final[col].value_counts(dropna=False, normalize=False))
    # print("Percentages:")
    print((df_final[col].value_counts(dropna=False, normalize=True) * 100).round(2).astype(str) + '%')

# %%

for col in df_final.select_dtypes(include=['number']).columns:
    print(f"\nColumn: {col}")
    # print(df_final[col].value_counts(dropna=False, normalize=False))
    # print("Percentages:")
    print((df_final[col].value_counts(dropna=False, normalize=True) * 100).round(2).astype(str) + '%')

# %%
# Mean of the three columns, regard the nan values non-exist
df_final['Combined_Trust'] = np.nanmean(df_final[['Combined_Trust', 'Combined_Safe', 'Combined_Focus']], axis=1)
df_final['Combined_Trust']

# df_final.to_pickle('data/main/df_final.pkl')

# %%
import pandas as pd
import numpy as np

# Group 1,2 as 'Low'; 3,4,5 as 'Medium'; 6,7 as 'High'
likert_cols = ['q_tech_confidence']
for col in likert_cols:
    df_final[col] = np.select( # Directly overwrite the original column
        [df_final[col].isin([1, 2]), df_final[col].isin([3, 4, 5]), df_final[col].isin([6, 7])],
        ['Low', 'Medium', 'High'],
        default='Unknown'
    )

# Re-grouping answer_value: 1,2 / 3,4,5 / 6,7
df_final['answer_value'] = np.select( # Directly overwrite the original column
    [df_final['answer_value'].isin([1, 2]), df_final['answer_value'].isin([3, 4, 5]), df_final['answer_value'].isin([6, 7])],
    [0, 1, 2],
    default='Unknown'
)

# Re-grouping for age: ~25 / 25-45 / 45-65 / +65
df_final['Combined_Trust'] = np.select( # Directly overwrite the original column
    [df_final['Combined_Trust'] <= 2.5,
      (df_final['Combined_Trust'] > 2.5) & (df_final['Combined_Trust'] <= 5.5),
      df_final['Combined_Trust'] > 5.5],
    ['Low', 'Medium', 'High'],
    default='Unknown'
)

# Re-grouping for age: ~25 / 25-45 / 45-65 / +65
df_final['q_age'] = np.select( # Directly overwrite the original column
    [df_final['q_age'] <= 25,
      (df_final['q_age'] > 25) & (df_final['q_age'] <= 45),
      (df_final['q_age'] > 45) & (df_final['q_age'] <= 65),
      df_final['q_age'] > 65],
    ['<25', '25-45', '45-65', '>65'],
    default='Unknown'
)

# Re-grouping for driving experience: ~5 / 5-10 / +5
df_final['q_driving_experience_years'] = np.select( # Directly overwrite the original column
    [df_final['q_driving_experience_years'] < 5,
     (df_final['q_driving_experience_years'] >= 5) & (df_final['q_driving_experience_years'] <= 10),
     df_final['q_driving_experience_years'] > 10],
    ['<5', '5-10', '>10'],
    default='Unknown'
)

# Re-grouping for q_adas_usage: 1~4 / 5~6 / 7~10
df_final['q_adas_usage'] = np.select( # Directly overwrite the original column
    [df_final['q_adas_usage'].isin([1,2]), df_final['q_adas_usage'].isin([3,4,5]), df_final['q_adas_usage'].isin([6,7])],
    ['1-4', '5-6', '7-10'],
    default='Unknown'
)

# redifine for Student status: 
df_final['Ethnicity simplified'] = np.select( # Directly overwrite the original column
    [df_final['Ethnicity simplified'].isin(['DATA_EXPIRED', 'CONSENT_REVOKED', 'Other']),
     df_final['Ethnicity simplified'].isna()],
    ['Unknown', 'Unknown'],
    default=df_final['Ethnicity simplified']
)

# redifine for Student status: 
df_final['Student status'] = np.select( # Directly overwrite the original column
    [df_final['Student status'].isin(['DATA_EXPIRED', 'CONSENT_REVOKED']),
     df_final['Student status'].isna()],
    ['Unknown', 'Unknown'],
    default=df_final['Student status']
)

# Employment status: 
df_final['Employment status'] = np.select( # Directly overwrite the original column
    [df_final['Employment status'].isin(['Unemployed (and job seeking)',
                                        "Not in paid work (e.g. homemaker', 'retired or disabled)",
                                        'Due to start a new job within the next month']),
    df_final['Employment status'].isin(['DATA_EXPIRED', 'CONSENT_REVOKED', 'nan', 'Other']),
    df_final['Employment status'].isna()],
    ['Unemployed', 'Unknown', 'Unknown'],
    default=df_final['Employment status']
)

# Country of residence: 
df_final['Country of residence'] = np.select( # Directly overwrite the original column
    [df_final['Country of residence'].isin(['DATA_EXPIRED', 'CONSENT_REVOKED', 'nan', 'Other']),
    df_final['Country of residence'].isna()],
    ['Unknown', 'Unknown'],
    default=df_final['Country of residence']
)

# Display the distribution of the transformed columns
print("\n--- Distribution of Transformed Columns ---")
# List of columns that were transformed
transformed_cols = likert_cols + ['q_age', 'q_driving_experience_years', 'q_adas_usage']

for col in transformed_cols:
    print(f"\n{col}:")
    print(df_final[col].value_counts())

# Show unique values of df_final for each column with object dtype
df_final.select_dtypes(include=['object']).nunique()
for col in df_final.select_dtypes(include=['object']).columns:
    print(f"{col}: {df_final[col].unique()}")

# %%
df = df_final.copy()

target_question = 'answer_value'
# Modefy the target question to 0: 0, 1 ; 1: 2, 3, 4, 5 ; 2: 6, 7
df[target_question] = np.select(
    [df[target_question].isin(['0', '1']), df[target_question].isin(['2'])],
    [0, 1],
    default=np.nan
)

df.columns = df.columns.str.replace('q_', '')
df.columns = df.columns.str.replace('_', ' ')
df.columns = df.columns.str.replace('  ', ' ')
df.columns

# %%
# rename the columns
df.rename(columns={'attention check':'Attention Check',
                   'vehicle imagined' : "Vehicle Imagined",
                    'passenger in sdc': 'AV Experience',
                    'primary transport': 'Primary Mode',
                    'adas usage': 'ADAS Usage',
                    'tech confidence': 'Tech Confidence',
                    'driving experience years': 'Driving Experience',
                    'continent': 'Continent',
                    'gender': 'Gender',
                    'age': 'Age',
                    'display' : 'Display',
                    'Combined Trust': 'Driver Trust',
                    'Combined Safe': 'Passenger Safety',
                    'Combined Focus': 'Passenger Distraction',
                    'Ethnicity simplified': 'Ethnicity',
                    'Country of residence': 'Country',
                    'Student status': 'Student',
                    'Employment status': 'Employment',
                    'weather': 'Weather',
                    'scene': 'Scene',
                    'light conditions': 'Light Conditions',
                    'road type': 'Road Type',
                    'offset': 'Time To Event',
                    'ego fault': 'Ego Fault',
                    'point of impact': 'Point of Impact',
                    'ego vehicle avoidability': 'Ego Avoidability',
                    'ego maneuver': 'Ego Maneuver',
                    'ego speeding': 'Ego Speeding',
                    'ego violation': 'Ego Violation',
                    'counterpart maneuver': 'Other Maneuver',
                    'counterpart speeding': 'Other Speeding',
                    'counterpart body style': 'Other Body Style',
                    'traffic violation': 'Other Violation',
                    'VRU involve' : 'VRU Involvement',
                    'impact severity': 'Impact Severity',
                    'answer value': 'Answer Value'}, inplace=True)
df.columns

# %%


# Change values of the columns
df['Point of Impact'] = df['Point of Impact'].replace({'rear-end': 'Rear-End', 'sideswipe': 'Sideswipe', 'head-on': 'Head-On',
                                                       'T-bone': 'T-bone', 'other': 'Other'})

df['Ego Avoidability'] = df['Ego Avoidability'].replace({'clearly avoidable ': 'Clearly Avoidable',
                                                          'potentially avoidable ': 'Potentially Avoidable', 'unavoidable': 'Unavoidable'})

df['Road Type'] = df['Road Type'].replace({'signalized intersection': 'Signalized Intersection',
                                            'collector road': 'Collector', 'arterial': 'Arterial',
                                              'highway': 'Highway', 'ramp': 'Ramp', 'stop sign intersection': 'Stop Sign Intersection',
                                                'parking lot': 'Parking Lot', 'residential area': 'Residential Area'})
df['Ego Fault'] = df['Ego Fault'].replace({1: 'Yes', 0: 'No'})

df['AV Experience'] = df['AV Experience'].replace({1: 'Yes', 2: 'No'})
df['Primary Mode'] = df['Primary Mode'].replace({1: 'Car', 2: 'Public Transport', 3: 'Bicycle/on foot', 4: 'Motorcycle'})
df['ADAS Usage'] = df['ADAS Usage'].replace({'1-4': 'Low', '5-6': 'Medium', '7-10' : 'High'})

df['Continent'] = df['Continent'].replace({1: 'Africa', 2: 'Asia', 3: 'Europe', 4: 'North America', 5: 'Oceania', 6: 'South America'})

df['Gender'] = df['Gender'].replace({1: 'Male', 2: 'Female', 3: 'Other', 4: 'Other', 5: 'Other'})
df['Age'] = df['Age'].replace({1: '18-24', 2: '25-34', 3: '35-44', 4: '45-54', 5: '55-64', 6: '65+'})
df['Display'] = df['Display'].replace({1: 'Yes', 0: 'No'})

df['Ego Maneuver'] = df['Ego Maneuver'].replace({'proceeding straight': 'Go Straight',
                                                  'turning': 'Turning', 'lane changing': 'Lane Changing',
                                                    'slow down': 'Slow Down', 'stop': 'Stop',
                                                      'merging': 'Merging', 'parking': 'Parking'})
df['Ego Speeding'] = df['Ego Speeding'].replace({'yes': 'Yes', 'no': 'No'})
df['Ego Violation'] = df['Ego Violation'].replace({'no': 'No', 'right-of-way': 'Right-of-way',
                                                    'red-light-running': 'Red Light Running', 'stop sign': 'Stop Sign',
                                                      'centerline': 'Centerline', 'off-road-running': 'Off-road'})

df['Other Maneuver'] = df['Other Maneuver'].replace({'proceeding straight': 'Go Straight',
                                                                  'turning': 'Turning', 'lane changing': 'Lane Changing',
                                                                    'slow down': 'Slow Down', 'stop': 'Stop',
                                                                      'merging': 'Merging', 'parking': 'Parking', 'backing up': 'Backing Up',
                                                                      'open the door': 'Openning Door', 'waiting': 'Waiting', 'merging' : 'Merging',
                                                                      'other': 'Other'})

df['Other Speeding'] = df['Other Speeding'].replace({'yes': 'Yes', 'no': 'No'})
df['Other Body Style'] = df['Other Body Style'].replace({'sedan': 'Sedan', 'van': 'Van', 'truck': 'Truck', 'pickup truck': 'Pickup Truck',
                                                          'suv': 'SUV', 'other': 'Other'})

df['Other Violation'] = df['Other Violation'].replace({'no': 'No', 'right-of-way': 'Right-of-way',
                                                    'red-light-running': 'Red Light Running', 'stop sign': 'Stop Sign',
                                                      'centerline': 'Centerline', 'off-road-running': 'Off-road',
                                                        'jaywalking': 'Jaywalking', 'ilegal parking': 'Illegal Parking',
                                                          'other': 'Other'})

df['VRU Involvement'] = df['VRU Involvement'].replace({'no': 'No', 'yes': 'Yes'})
df['Impact Severity'] = df['Impact Severity'].replace({'minor': 'Minor', 'moderate': 'Moderate', 'major': 'Severe', 'near-miss': 'Near-Miss'})

df['Other Violation'] = df['Other Violation'].replace({'no': 'No', 'right-of-way': 'Right-of-way',
                                                                  'red-light-running': 'Red Light Running', 'stop sign': 'Stop Sign',
                                                                    'centerline': 'Centerline', 'off-road-running': 'Off-road',
                                                                      'other': 'Other'})

# %%
drop_columns = ['Passenger Safety', 'Passenger Distraction']

df = df.drop(columns=drop_columns)

# %%
# Re-grouping for driving experience: ~5 / 5-10 / +5
df['Attention Check'] = np.select( # Directly overwrite the original column
    [df['Attention Check'] == 1,
     df['Attention Check'] == 2],
    ['Pass', 'Fail'],
    default='Unknown'
)

df['Vehicle Imagined'] = np.select( # Directly overwrite the original column
    [df['Vehicle Imagined'] == 2,
     df['Vehicle Imagined'] == 1],
    ['AV', 'HDV'],
    default='Unknown'
)

df['Display'] = np.select( # Directly overwrite the original column
    [df['Display'] == 'Yes',
     df['Display'] == 'No'],
    ['AV', 'HDV'],
    default='Unknown'
)

# %%
df.rename(columns={'ID': 'Video ID'}, inplace=True)

# %%
df.to_csv('data/anonymized_survey+demo+video_matched.csv', index=False)

# %%
df = pd.read_csv('data/anonymized_survey+demo+video_matched.csv')
df

# %%
# convert to string
df['Driving Experience'] = df['Driving Experience'].astype(str)

df['Driving Experience']
# = df['Driving Experience'].astype(str)

# %%

df['Driving Experience']

# %%
df.to_parquet('data/df_final.parquet', index=False)

