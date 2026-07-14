# %%
import plotly.graph_objects as go
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# %%
df = pd.read_parquet('data/main/df_final.parquet')

# %%
df['Display'].value_counts()

# %%

# Plot "Driver Trust" histogram
import pandas as pd
import matplotlib.pyplot as plt

# Plot "Driver Trust" histogram by "Display"

# m
df_yes = df[df['Display'] == 'Yes']
df_no = df[df['Display'] == 'No']

print(df_yes['Driver Trust'].value_counts(normalize=True))
print(df_no['Driver Trust'].value_counts(normalize=True))








# %%
col_participant = ['AV Experience', 'Primary Mode',
                    'ADAS Usage', 'Tech Confidence', 'Driving Experience',
                      'Continent', 'Gender', 'Age',
                        'Driver Trust',
                         'Ethnicity', 'Country', 'Student', 'Employment']

col_scenario = ['Weather', 'Scene', 'Light Conditions', 'Road Type', 'Time To Event', 'Ego Fault', 'Point of Impact',
                 'Ego Avoidability', 'Ego Maneuver', 'Ego Speeding', 'Ego Violation',
                   'Other Maneuver', 'Other Speeding', 'Other Body Style',
                     'Other Violation', 'VRU Involvement', 'Impact Severity']

# %%
df_participant = df[col_participant]


print(df_participant.shape)

df_scenario = df[col_scenario]
df_scenario.columns = df_scenario.columns.str.replace('_', ' ')
print(df_scenario.shape)

# %%
df_plot = pd.concat([df_participant, df_scenario], axis=1)

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# --- Set the default renderer for Jupyter Notebook ---
pio.renderers.default = "notebook"

# %%
trans_col = ['AV Experience', 'ADAS Usage', 'Tech Confidence',
                     'Primary Mode', 'Driving Experience',
                        'Driver Trust']

demo_col = ['Gender', 'Age', 'Ethnicity',
             'Continent', 'Student', 'Employment']

env_col = ['Weather', 'Scene', 'Light Conditions', 'Road Type']

crash_col = ['Time To Event', 'Point of Impact', 'VRU Involvement', 'Impact Severity']

crash_env_col = env_col + crash_col

counterpart_col = ['Other Maneuver', 'Other Speeding', 'Other Body Style', 'Other Violation']

ego_col = ['Ego Fault', 'Ego Avoidability', 'Ego Maneuver', 'Ego Speeding', 'Ego Violation']

maneuver_col = counterpart_col + ego_col

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- 3. Initialize Subplots (3 Rows, 2 Columns) ---
fig = make_subplots(
    rows=2, cols=2,  # Changed from 2 rows, 3 cols
    specs=[[{'type': 'domain'}, {'type': 'domain'}],  # Updated specs for 3x2 grid
           [{'type': 'domain'}, {'type': 'domain'}]],
    horizontal_spacing=0.01,
    vertical_spacing=0.01
)

# --- 4. Define Groups, Positions, and Titles ---
column_groups = [trans_col, demo_col, crash_env_col, maneuver_col]

# Updated positions for a 3x2 grid
subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

# Define titles in a list to be used for the sunburst path
titles = ['Mobility<br>Behavior<br>& Attitudes', 'Demographics', 'Crash<br>Environment',
          'Maneuver<br>Properties']

# --- 5. Loop to Create and Add Each Sunburst Chart ---
# The loop remains the same, but uses the updated positions
for columns_to_plot, pos, title in zip(column_groups, subplot_positions, titles):
    # Reshape data from wide to long format
    df_melted = df_plot[columns_to_plot].melt(var_name='variable', value_name='value')
    df_melted['variable'] = df_melted['variable'].str.replace(' ', '<br>')
    df_melted['value'] = df_melted['value'].astype(str).str.replace(' ', '<br>')

    # if title contains 'counterpart' or 'ego', then remove the 'counterpart' or 'ego' in the variable
    if 'Counterpart' in title:
        df_melted['variable'] = df_melted['variable'].str.replace('counterpart', '')
    if 'Ego' in title:
        df_melted['variable'] = df_melted['variable'].str.replace('ego', '')
        df_melted['variable'] = df_melted['variable'].str.replace('vehicle', '')
        
    # The 'title' for the path now comes from our list
    df_melted['title'] = title

    # Create a temporary sunburst figure to generate the trace
    sunburst_fig = px.sunburst(
        df_melted,
        path=['title', 'variable', 'value']
    )

    # --- Custom Text Template Logic ---
    custom_text_templates = []
    for entry in sunburst_fig.data[0].ids:
        if entry.count('/') == 2:
            template = "<b>%{label}</b><br>%{percentParent:.0%}"
        else:
            template = "<b>%{label}</b>"
        custom_text_templates.append(template)

    # Apply the custom templates to the trace
    sunburst_fig.data[0].texttemplate = custom_text_templates

    # Add the customized trace to the main subplot figure
    fig.add_trace(
        sunburst_fig.data[0],
        row=pos[0], col=pos[1]
    )

# --- 6. Final Layout and Trace Customization ---
fig.update_traces(
    insidetextorientation='radial',
    rotation=90
)
fig.update_layout(
    # title_text="Multi-Category Sunburst Chart Analysis",
    height=800, 
    width=800,  
    margin=dict(t=0, l=0, r=0, b=0) 
)


fig.show()
# save the figure with high resolution
fig.write_image("sunburst_chart.png", scale=3, width=800, height=800)

# %% [markdown]
# # Literature Review Flow Analysis

# %%
df

# %%
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

df = pd.read_excel('data\AV_Literature_Review_Categorization_v2.xlsx')

# --- 1. Define the desired order for nodes in each column ---
sankey_cols = ['Focus', 'Methodology', 'Stimuli type']

# Get the order for the first column dynamically
focus_order = sorted(df['Focus'].unique().tolist())

# --- THIS SECTION IS UPDATED WITH YOUR SPECIFIC ORDER ---
method_order = ["Survey", "Qualitative + Survey", "Qualitative", "Experiment"]
stimuli_order = ['No stimuli', 'Field study', 'VR', 'Text', 'Text+Video', "Video"]
# --- END OF UPDATED SECTION ---

# The final list of labels MUST be in this specific order.
# The code will only include labels present in the dataframe to avoid errors.
all_df_labels = set(df[sankey_cols[0]]).union(set(df[sankey_cols[1]])).union(set(df[sankey_cols[2]]))
ordered_labels = [lbl for lbl in (focus_order + method_order + stimuli_order) if lbl in all_df_labels]

# Create a mapping from the label to its final index
label_to_id = {label: i for i, label in enumerate(ordered_labels)}

# --- 2. Prepare Data for Sankey Diagram ---
sources = []
targets = []
values = []

for i in range(len(sankey_cols) - 1):
    grouped = df.groupby([sankey_cols[i], sankey_cols[i + 1]]).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        # Check if both source and target labels are in our ordered list before adding
        if row[sankey_cols[i]] in label_to_id and row[sankey_cols[i+1]] in label_to_id:
            sources.append(label_to_id[row[sankey_cols[i]]])
            targets.append(label_to_id[row[sankey_cols[i + 1]]])
            values.append(row['count'])

# --- 3. Calculate Node Totals (Corrected Method) ---
node_inflow = {i: 0 for i in range(len(ordered_labels))}
node_outflow = {i: 0 for i in range(len(ordered_labels))}

for source, target, value in zip(sources, targets, values):
    node_outflow[source] += value
    node_inflow[target] += value

node_totals = {i: max(node_inflow[i], node_outflow[i]) for i in range(len(ordered_labels))}
new_labels = [f"{label} ({node_totals[label_to_id[label]]})" for label in ordered_labels]

# --- 4. Generate Node X and Y Coordinates ---
node_x = []
node_y = []

# Create helper maps for vertical positioning based on the *actual* labels present
focus_labels_present = [lbl for lbl in focus_order if lbl in all_df_labels]
method_labels_present = [lbl for lbl in method_order if lbl in all_df_labels]
stimuli_labels_present = [lbl for lbl in stimuli_order if lbl in all_df_labels]

focus_pos_map = {label: i for i, label in enumerate(focus_labels_present)}
method_pos_map = {label: i for i, label in enumerate(method_labels_present)}
stimuli_pos_map = {label: i for i, label in enumerate(stimuli_labels_present)}

for label in ordered_labels:
    if label in focus_order:
        node_x.append(0.0)
        y_val = focus_pos_map[label] / (len(focus_labels_present) - 1) if len(focus_labels_present) > 1 else 0.5
        node_y.append(y_val)
    elif label in method_order:
        node_x.append(0.5)
        y_val = method_pos_map[label] / (len(method_labels_present) - 1) if len(method_labels_present) > 1 else 0.5
        node_y.append(y_val)
    elif label in stimuli_order:
        node_x.append(1.0)
        y_val = stimuli_pos_map[label] / (len(stimuli_labels_present) - 1) if len(stimuli_labels_present) > 1 else 0.5
        node_y.append(y_val)


# --- 5. Create and Display the Sankey Diagram ---
fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=10,
        thickness=10,
        line=dict(color="black", width=1),
        label=new_labels,
        color=px.colors.qualitative.Plotly,
        x=node_x,
        y=node_y
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    ))])
# figure size
fig.update_layout(width=800, height=500)
# fig.update_layout(title_text="Literature Review Flow Analysis (Custom Order)", font_size=12)
fig.show()

# %%
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import numpy as np

# --- Helper Function to Clean Stimuli Data ---
def clean_stimuli_count(value):
    """
    Converts a value (int, float, or str like '3-5') into a numeric type.
    Ranges are converted to their average.
    Returns np.nan if conversion is not possible.
    """
    if pd.isna(value):
        return np.nan
    
    s_val = str(value).strip()
    
    # Check for a range like '3-5' or '3 to 5'
    range_match = re.match(r'(\d+)\s*[-–to]+\s*(\d+)', s_val)
    if range_match:
        num1 = int(range_match.group(1))
        num2 = int(range_match.group(2))
        return (num1 + num2) / 2
    
    # Check for a single number (integer or float)
    try:
        return float(s_val)
    except ValueError:
        return np.nan # Return NaN for non-numeric strings

# --- Load Data ---
df = pd.read_excel('data\AV_Literature_Review_Categorization_v2.xlsx')

# (Steps 1-5 for the Sankey plot remain unchanged)

# --- 1. Define the desired order for nodes in each column ---
sankey_cols = ['Focus', 'Methodology', 'Stimuli type']
focus_order = sorted(df['Focus'].unique().tolist())
method_order = ["Survey", "Qualitative + Survey", "Qualitative", "Experiment"]
stimuli_order = ['No stimuli', 'Field study', 'VR', 'Text', 'Text+Video', "Video"]

all_df_labels = set(df[sankey_cols[0]]).union(set(df[sankey_cols[1]])).union(set(df[sankey_cols[2]]))
ordered_labels = [lbl for lbl in (focus_order + method_order + stimuli_order) if lbl in all_df_labels]
label_to_id = {label: i for i, label in enumerate(ordered_labels)}

# --- 2. Prepare Data for Sankey Diagram ---
sources, targets, values = [], [], []
for i in range(len(sankey_cols) - 1):
    grouped = df.groupby([sankey_cols[i], sankey_cols[i + 1]]).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        if row[sankey_cols[i]] in label_to_id and row[sankey_cols[i+1]] in label_to_id:
            sources.append(label_to_id[row[sankey_cols[i]]])
            targets.append(label_to_id[row[sankey_cols[i + 1]]])
            values.append(row['count'])

# --- 3. Calculate Node Totals ---
node_inflow = {i: 0 for i in range(len(ordered_labels))}
node_outflow = {i: 0 for i in range(len(ordered_labels))}
for source, target, value in zip(sources, targets, values):
    node_outflow[source] += value; node_inflow[target] += value
node_totals = {i: max(node_inflow[i], node_outflow[i]) for i in range(len(ordered_labels))}
new_labels = [f"{label} ({node_totals[label_to_id[label]]})" for label in ordered_labels]

# --- 4. Generate Node X and Y Coordinates ---
node_x, node_y = [], []
focus_labels_present = [lbl for lbl in focus_order if lbl in all_df_labels]
method_labels_present = [lbl for lbl in method_order if lbl in all_df_labels]
stimuli_labels_present = [lbl for lbl in stimuli_order if lbl in all_df_labels]
focus_pos_map = {label: i for i, label in enumerate(focus_labels_present)}
method_pos_map = {label: i for i, label in enumerate(method_labels_present)}
stimuli_pos_map = {label: i for i, label in enumerate(stimuli_labels_present)}
for label in ordered_labels:
    if label in focus_order:
        node_x.append(0.0)
        y_val = focus_pos_map[label]/(len(focus_labels_present) - 1) if len(focus_labels_present) > 1 else 0.5
        node_y.append(y_val)
    elif label in method_order:
        node_x.append(0.5)
        y_val = method_pos_map[label]/(len(method_labels_present) - 1) if len(method_labels_present) > 1 else 0.5
        node_y.append(y_val)
    elif label in stimuli_order:
        node_x.append(1.0)
        y_val = stimuli_pos_map[label]/(len(stimuli_labels_present) - 1) if len(stimuli_labels_present) > 1 else 0.5
        node_y.append(y_val)

# --- 5. Create and Display the Sankey Diagram ---
sankey_fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=new_labels, color=px.colors.qualitative.Plotly, x=node_x, y=node_y),
    link=dict(source=sources, target=targets, value=values))])
sankey_fig.update_layout(width=800, height=500)
sankey_fig.show()

## --- 6. Create Bar Chart of Stimuli per Author ---

# NOTE: Verify these column names match your Excel file exactly.
author_col = 'Paper'
stimuli_count_col = 'number of stimuli/scenarios'

if author_col in df.columns and stimuli_count_col in df.columns:
    # Create a clean DataFrame for this plot
    df_plot = df[[author_col, stimuli_count_col]].copy()
    
    # Apply the cleaning function to convert stimuli count to a number
    df_plot['stimuli_numeric'] = df_plot[stimuli_count_col].apply(clean_stimuli_count)
    
    # Remove rows where stimuli count is 0 or could not be converted
    df_plot = df_plot.dropna(subset=['stimuli_numeric'])
    df_plot = df_plot[df_plot['stimuli_numeric'] > 0]
    
    if not df_plot.empty:
        # Sort the DataFrame so the largest bar is at the top
        df_plot = df_plot.sort_values(by='stimuli_numeric', ascending=True)

        # Create the horizontal bar chart
        bar_fig = px.bar(
            df_plot,
            x='stimuli_numeric',  # Changed from author_col
            y=author_col,          # Changed from 'stimuli_numeric'
            orientation='h',       # Added for horizontal chart
            title='Number of Stimuli/Scenarios per Study',
            text_auto=True         # Display the value on each bar
        )
        
        # Improve layout and axis titles
        bar_fig.update_layout(
            xaxis_title="Number of Stimuli/Scenarios", # Swapped
            yaxis_title="Author"                      # Swapped
        )
        bar_fig.update_traces(textposition='outside')
        
        # Display the bar chart
        bar_fig.show()
    else:
        print(f"No valid, non-zero data to plot for '{stimuli_count_col}'.")
else:
    print(f"Warning: Ensure columns '{author_col}' and '{stimuli_count_col}' exist in the Excel file.")

# %%
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import re
import numpy as np

# --- Helper Function to Clean Stimuli Data ---
def clean_stimuli_count(value):
    """
    Converts a value (int, float, or str like '3-5') into a numeric type.
    Ranges are converted to their average.
    Returns np.nan if conversion is not possible.
    """
    if pd.isna(value):
        return np.nan

    s_val = str(value).strip()

    # Check for a range like '3-5' or '3 to 5'
    range_match = re.match(r'(\d+)\s*[-–to]+\s*(\d+)', s_val)
    if range_match:
        num1 = int(range_match.group(1))
        num2 = int(range_match.group(2))
        return (num1 + num2) / 2

    # Check for a single number (integer or float)
    try:
        return float(s_val)
    except ValueError:
        return np.nan # Return NaN for non-numeric strings


# --- 1. Define the desired order for nodes in each column ---
sankey_cols = ['Focus', 'Methodology', 'Stimuli type']
focus_order = sorted(df['Focus'].unique().tolist())
method_order = ["Survey", "Qualitative + Survey", "Qualitative", "Experiment"]
stimuli_order = ['No stimuli', 'Field study', 'VR', 'Text', 'Text+Video', "Video"]

all_df_labels = set(df[sankey_cols[0]]).union(set(df[sankey_cols[1]])).union(set(df[sankey_cols[2]]))
ordered_labels = [lbl for lbl in (focus_order + method_order + stimuli_order) if lbl in all_df_labels]
label_to_id = {label: i for i, label in enumerate(ordered_labels)}

# --- 2. Prepare Data for Sankey Diagram ---
sources, targets, values = [], [], []
for i in range(len(sankey_cols) - 1):
    grouped = df.groupby([sankey_cols[i], sankey_cols[i + 1]]).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        if row[sankey_cols[i]] in label_to_id and row[sankey_cols[i+1]] in label_to_id:
            sources.append(label_to_id[row[sankey_cols[i]]])
            targets.append(label_to_id[row[sankey_cols[i + 1]]])
            values.append(row['count'])

# --- 3. Calculate Node Totals ---
node_inflow = {i: 0 for i in range(len(ordered_labels))}
node_outflow = {i: 0 for i in range(len(ordered_labels))}
for source, target, value in zip(sources, targets, values):
    node_outflow[source] += value; node_inflow[target] += value
node_totals = {i: max(node_inflow[i], node_outflow[i]) for i in range(len(ordered_labels))}
new_labels = [f"{label}<br>({node_totals[label_to_id[label]]})" for label in ordered_labels]

# --- 4. Generate Node X and Y Coordinates ---
node_x, node_y = [], []
focus_labels_present = [lbl for lbl in focus_order if lbl in all_df_labels]
method_labels_present = [lbl for lbl in method_order if lbl in all_df_labels]
stimuli_labels_present = [lbl for lbl in stimuli_order if lbl in all_df_labels]
focus_pos_map = {label: i for i, label in enumerate(focus_labels_present)}
method_pos_map = {label: i for i, label in enumerate(method_labels_present)}
stimuli_pos_map = {label: i for i, label in enumerate(stimuli_labels_present)}
for label in ordered_labels:
    if label in focus_order:
        node_x.append(0.0)
        y_val = focus_pos_map[label]/(len(focus_labels_present) - 1) if len(focus_labels_present) > 1 else 0.5
        node_y.append(y_val)
    elif label in method_order:
        node_x.append(0.5)
        y_val = method_pos_map[label]/(len(method_labels_present) - 1) if len(method_labels_present) > 1 else 0.5
        node_y.append(y_val)
    elif label in stimuli_order:
        node_x.append(1.0)
        y_val = stimuli_pos_map[label]/(len(stimuli_labels_present) - 1) if len(stimuli_labels_present) > 1 else 0.5
        node_y.append(y_val)

# --- 5. Create Traces for Sankey and Bar Chart ---
sankey_trace = go.Sankey(
    arrangement='snap',
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=new_labels, color=px.colors.qualitative.Plotly, x=node_x, y=node_y),
    link=dict(source=sources, target=targets, value=values)
)

# --- 6. Prepare Data and Create Bar Chart Trace ---
author_col = 'Paper'
stimuli_count_col = 'number of stimuli/scenarios'
bar_trace = None

if author_col in df.columns and stimuli_count_col in df.columns:
    df_plot = df[[author_col, stimuli_count_col]].copy()
    df_plot['stimuli_numeric'] = df_plot[stimuli_count_col].apply(clean_stimuli_count)
    df_plot = df_plot.dropna(subset=['stimuli_numeric'])
    df_plot = df_plot[df_plot['stimuli_numeric'] > 0]

    if not df_plot.empty:
        df_plot = df_plot.sort_values(by='stimuli_numeric', ascending=True)
        bar_trace = go.Bar(
            x=df_plot['stimuli_numeric'],
            y=df_plot[author_col],
            orientation='h',
            text=df_plot['stimuli_numeric'],
            textposition='outside'
                            )
    else:
        print(f"No valid, non-zero data to plot for '{stimuli_count_col}'.")
else:
    print(f"Warning: Ensure columns '{author_col}' and '{stimuli_count_col}' exist in the Excel file.")

# --- 7. Create Combined Figure with Subplots ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.6, 0.4],
    specs=[[{"type": "sankey"}, {"type": "xy"}]],
    subplot_titles=("(a) Literature Review Flow", "(b) Stimuli per Study"),
    horizontal_spacing=0.17  
)

fig.add_trace(sankey_trace, row=1, col=1)

if bar_trace:
    fig.add_trace(bar_trace, row=1, col=2)
    fig.update_xaxes(title_text="Number of Stimuli/Scenarios", row=1, col=2)
    fig.update_layout(xaxis=dict(range=[0, 1500]))
    # fig.update_yaxes(title_text="Author", row=1, col=2)


fig.update_layout(
    width=1200,
    height=500,
    showlegend=False
)
# save figure
fig.write_image("scenario_visualization.png", width=1200, height=500, scale=1)
fig.show()

