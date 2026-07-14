# %%
import pandas as pd
import os
# Show all dataframe columns
pd.set_option('display.max_columns', None)

# Read the CSV file from current directory
# df = pd.read_excel(r"data/2025_0720_1000 Main_July 25, 2025_08.52.xlsx", header=1) # header is second row

# %%
# Save the dataframe to a parquet file for later use
if os.path.exists('data/raw_survey_data.parquet'):
    df = pd.read_parquet('data/raw_survey_data.parquet')
else:
    # df = pd.read_excel(r"data/2025_0720_1000 Main_July 25, 2025_08.52.xlsx")
    df.to_parquet('data/raw_survey_data.parquet', index=False)

# %%
cols_to_drop = [col for col in df.columns if 'Timing - ' in col]

# Drop the columns
df.drop(cols_to_drop, axis=1, inplace=True)

# %%
df.head()

# %%
import re

# --- Identify id_vars and value_vars ---
all_columns = df.columns.tolist()

# value_vars are columns containing 'File.php?F=' and NOT starting with 'Timing - '
# and representing the actual questions.
# The question columns have the structure: 'https://...File.php?F={unique_id} - {Question_Text}'
# The timing columns have the structure: 'Timing - https://...File.php?F={unique_id} - Timing - {Metric}'
# We want to melt the columns that are direct questions.
value_vars = [
    col for col in all_columns
    if 'File.php?F=' in col and not col.startswith('Timing - ')
]

# id_vars are all other columns
id_vars = [col for col in all_columns if col not in value_vars]

print(f"\nNumber of id_vars: {len(id_vars)}")
print(f"\nNumber of value_vars: {len(value_vars)}")

# --- Perform the melt operation ---
df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='VariableHeader', value_name='AnswerValue')

print(f"\nShape of DataFrame after melt: {df_long.shape}")

# --- Extract unique_id and Question from VariableHeader ---
# The pattern is 'https://...File.php?F={unique_id} - {Question}'
# We need to extract the part after 'F=' and before ' - ' as unique_id
# and the part after ' - ' as Question.

def extract_id_question(header):
    match = re.search(r'File\.php\?F=([A-Za-z0-9_]+)\s*-\s*(.+)', header)
    if match:
        return pd.Series([match.group(1), match.group(2).strip()])
    return pd.Series([None, None])

df_long[['unique_id', 'Question']] = df_long['VariableHeader'].apply(
    lambda x: pd.Series(extract_id_question(x))
)

# --- Display some info about the new columns ---
print("\nInfo of the final long DataFrame:")
df_long.info()

print("\nValue counts for extracted 'unique_id':")
print(df_long['unique_id'].value_counts(dropna=False))

print("\nValue counts for extracted 'Question' (first 5):")
print(df_long['Question'].value_counts(dropna=False).head())

# Drop rows where AnswerValue is NaN
df_long = df_long.dropna(subset=['AnswerValue'])

# %%
df_long.shape

# %%
output_file_name = "data/df_long.csv"
df_long.to_csv(output_file_name, index=False)

# %%
# Attention check #1
attn_check_1_question = df_long.columns[18]

attn_check_1 = df_long[attn_check_1_question] == 1
attn_check_1_df = df_long[attn_check_1]
print(attn_check_1_df.shape)

# Attention check #2
attn_check_2_question = df_long.columns[19]
attn_check_2 = df_long[attn_check_2_question].astype(int) == df_long['display'].astype(int) + 1
attn_check_2_df = df_long[attn_check_2]
print(attn_check_2_df.shape)

# Attention check #1 & #2
attn_check_1_2 = attn_check_1 & attn_check_2
attn_check_1_2_df = df_long[attn_check_1_2]
print(attn_check_1_2_df.shape)

# %%
# Perceived risk
df_long_PR = attn_check_1_2_df[attn_check_1_2_df['Question'].str.contains('perceive')]
output_file_name = "data/main/df_long_PR.csv"
df_long_PR.to_csv(output_file_name, index=False)


# # Crash Time
# df_long_CT = attn_check_3_df[attn_check_3_df['Question'].str.contains('When')]
# output_file_name = "data/pre test/df_long_CT.csv"
# df_long_CT.to_csv(output_file_name, index=False)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Duration = End Dat - Start Date
duration = df_long_PR['End Date'] - df_long_PR['Start Date']

# convert duration to minutes
duration = duration.dt.total_seconds() / 60

plt.hist(duration, bins=100)
plt.show()


# %%
# Attention check
df_long_CL_AC = df_long_CL[df_long_CL['While watching the video, which type of vehicle did you imagine yourself riding in?'] == df_long_CL['display'] + 1]
df_long_CL_AC.shape



# %%
df_long_AC

