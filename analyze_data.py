import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dirty_transport_dataset_cleaned.csv')

print("="*80)
print("DATA ANALYSIS - INVESTIGATING LOW MODEL PERFORMANCE")
print("="*80)

print("\n1. DELAY STATISTICS:")
print(df['delay_minutes'].describe())
print(f"Variance: {df['delay_minutes'].var():.2f}")
print(f"Std Dev: {df['delay_minutes'].std():.2f}")

print("\n2. DELAY BY ROUTE:")
print(df.groupby('route_id')['delay_minutes'].agg(['mean', 'std', 'min', 'max']))

print("\n3. DELAY BY WEATHER:")
print(df.groupby('weather')['delay_minutes'].agg(['mean', 'std', 'count']))

print("\n4. CORRELATIONS WITH DELAY:")
print(df[['passenger_count', 'latitude', 'longitude', 'delay_minutes']].corr()['delay_minutes'].sort_values())

print("\n5. SAMPLE DATA:")
print(df[['route_id', 'scheduled_time', 'actual_time', 'delay_minutes', 'weather', 'passenger_count']].head(20))

print("\n6. DELAY DISTRIBUTION:")
print(pd.cut(df['delay_minutes'], bins=[-np.inf, 30, 60, 90, 120, np.inf], 
             labels=['<30min', '30-60min', '60-90min', '90-120min', '>120min']).value_counts().sort_index())
