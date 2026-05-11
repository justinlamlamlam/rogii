import pandas as pd
import numpy as np
import os
import plotly.express as px

path_csv = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"

all_files = os.listdir(path_csv)
csv_files = [f for f in all_files if f.endswith('horizontal_well.csv')]

df_list = []

for csv_file in csv_files:
    file_path = os.path.join(path_csv, csv_file)
    df_temp = pd.read_csv(file_path)
    df_temp['csv_name'] = os.path.splitext(csv_file)[0] # Add csv_name column without extension
    df_list.append(df_temp)

df_combined = pd.concat(df_list, ignore_index=True)

print(f"Combined DataFrame shape: {df_combined.shape}")
print(df_combined.head())


df_combined['TVT_Color'] = np.where(df_combined['TVT_input'].notna(), 'blue', 'red')

print(df_combined[['TVT_input', 'TVT_Color']].head())
print(df_combined['TVT_Color'].value_counts())

# Changed px.scatter_3d to px.line_3d
# Added line_group='csv_name' to ensure lines connect only within individual files
"""fig = px.line_3d(
    data_frame=df_combined, 
    x='X', 
    y='Y', 
    z='Z', 
    color='TVT_Color',
    line_group='csv_name',
    title='3D Line Plot of X, Y, Z by TVT_Color'
)
fig.write_html("3d_plot.html")"""

df_downsampled = df_combined.iloc[::200].copy()

# Plotting the downsampled version
fig_downsampled = px.line_3d(
    data_frame=df_downsampled, 
    x='X', 
    y='Y', 
    z='Z', 
    color='TVT_Color',
    line_group='csv_name',
    title='3D Line Plot (1/200 Sampled Data)'
)

fig_downsampled.write_html("3d_plot_sampled.html")

print(f"Downsampled DataFrame shape: {df_downsampled.shape}")