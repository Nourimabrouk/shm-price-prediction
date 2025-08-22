import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/raw/Bit_SHM_data.csv', low_memory=False)

print(f'Dataset shape: {df.shape}')
print(f'Total records: {len(df)}')
print(f'Available columns: {list(df.columns)}')

# Check for 'Sales Price' column (our target)
if 'Sales Price' in df.columns:
    total_value = df['Sales Price'].sum()
    print(f'Total sales value: ${total_value:,.0f}')
    print(f'Total sales value in billions: ${total_value/1e9:.2f}B')
    print(f'Price range: ${df["Sales Price"].min():,.0f} - ${df["Sales Price"].max():,.0f}')
    print(f'Average price: ${df["Sales Price"].mean():,.2f}')
else:
    print('Sales Price column not found')

# Check machine hours missing data
if 'MachineHours CurrentMeter' in df.columns:
    missing_count = df['MachineHours CurrentMeter'].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f'Machine hours missing: {missing_count:,} ({missing_pct:.2f}%)')
else:
    print('MachineHours CurrentMeter column not found')

# Check geographic coverage - need to find the correct column name
geo_cols = [col for col in df.columns if 'state' in col.lower() or 'location' in col.lower()]
print(f'Geographic columns found: {geo_cols}')

if geo_cols:
    geo_col = geo_cols[0]  # Use first geographic column found
    unique_states = df[geo_col].nunique()
    print(f'Geographic coverage: {unique_states} unique locations')