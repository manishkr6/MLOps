import pandas as pd
import os

# Create a sample DataFrame with column names
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Loss Angeles', 'Chicago']
}

df = pd.DataFrame(data)

# Adding new row to df for V2
new_row_loc = {"Name": 'GF1',
               'Age': 20,
               'City': 'City1'}
df.loc[len(df.index)] = new_row_loc

# 1. Get the exact folder where mycode.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Tell it to make the 'data' folder INSIDE that script directory
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True) # (Also fixes the makedirs typo!)

# 3. Define the file path
file_path = os.path.join(data_dir, 'sample_data.csv')

# Save the DataFrame
df.to_csv(file_path, index=False)

print(f'CSV file saved to {file_path}')