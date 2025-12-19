import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

def clean_transport_data(input_file):
    """
    Clean the transport dataset by addressing various data quality issues.
    
    Args:
        input_file (str): Path to the dirty dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found. Please check the file path.")
    
    input_file_name = input_file.split('.')[0]
    output_file = f'{input_file_name}_cleaned.csv'
    
    # Load the data
    print("Loading dataset...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    print(f"Original dataset shape: {df.shape}")
    
    # Display initial data quality summary
    print("\n=== INITIAL DATA QUALITY SUMMARY ===")
    print("Missing values per column:")
    print(df.isnull().sum())
    print("\nUnique values in key columns:")
    print(f"Route IDs: {df['route_id'].unique()}")
    print(f"Weather values: {df['weather'].unique()}")
    
    # 1. Standardize Route IDs
    print("\n1. Standardizing Route IDs...")
    def standardize_route_id(route_id):
        if pd.isna(route_id):
            return route_id
        route_str = str(route_id).strip()
        
        # Extract number from any format and standardize to R0X format
        numbers = re.findall(r'\d+', route_str)
        if numbers:
            num = int(numbers[0])
            return f'R{num:02d}'
        
        # If no numbers found, return original
        return route_str
    
    df['route_id'] = df['route_id'].apply(standardize_route_id)
    
    # 2. Clean and standardize actual_time
    print("2. Cleaning actual_time column...")
    def clean_time_format(time_str):
        if pd.isna(time_str) or time_str == '':
            return np.nan
        
        time_str = str(time_str).strip()
        
        # Handle AM/PM format
        if 'AM' in time_str or 'PM' in time_str:
            try:
                # Parse formats like "04.56AM", "05.43AM", "07.21PM"
                time_str_clean = time_str.replace('.', ':')
                return pd.to_datetime(time_str_clean, format='%I:%M%p').strftime('%H:%M')
            except:
                return np.nan
        
        # Handle 24-hour format with colon
        if ':' in time_str:
            try:
                parts = time_str.split(':')
                if len(parts) == 2:
                    hour, minute = int(parts[0]), int(parts[1])
                    return f'{hour:02d}:{minute:02d}'
            except:
                return np.nan
        
        # Handle numeric formats like "224", "345", "815", "1115", etc.
        if time_str.isdigit():
            if len(time_str) <= 2:
                # Single or double digit represents minutes past midnight
                minute = int(time_str)
                return f'00:{minute:02d}'
            elif len(time_str) == 3:
                # Format like "345" -> "3:45"
                hour = int(time_str[0])
                minute = int(time_str[1:3])
                return f'{hour:02d}:{minute:02d}'
            elif len(time_str) == 4:
                # Format like "1245" -> "12:45"
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                return f'{hour:02d}:{minute:02d}'
        
        return np.nan
    
    df['actual_time'] = df['actual_time'].apply(clean_time_format)
    
    # 3. Standardize weather column
    print("3. Standardizing weather values...")
    def standardize_weather(weather):
        if pd.isna(weather) or weather == '':
            return np.nan
        
        weather_str = str(weather).strip().lower()
        
        # Fix spelling errors and standardize
        weather_mapping = {
            'sun': 'Sunny',
            'sunny': 'Sunny',
            'cloudy': 'Cloudy',
            'clody': 'Cloudy',  # Fix spelling error
            'rain': 'Rainy',
            'rainy': 'Rainy'
        }
        
        return weather_mapping.get(weather_str, weather_str.title())
    
    df['weather'] = df['weather'].apply(standardize_weather)
    
    # 4. Clean passenger_count (handle negative values and missing data)
    print("4. Cleaning passenger_count...")
    def clean_passenger_count(count):
        if pd.isna(count) or count == '':
            return np.nan
        
        try:
            count_val = float(count)
            # Replace negative values with NaN (invalid passenger counts)
            if count_val < 0:
                return np.nan
            # Cap at reasonable maximum (assuming buses/trains)
            if count_val > 200:
                return 200  # Cap at reasonable maximum
            return int(count_val)
        except:
            return np.nan
    
    df['passenger_count'] = df['passenger_count'].apply(clean_passenger_count)
    
    # 5. Clean coordinates (handle invalid values like 999)
    print("5. Cleaning latitude and longitude...")
    def clean_coordinate(coord, coord_type='lat'):
        if pd.isna(coord) or coord == '' or coord == 999:
            return np.nan
        
        try:
            coord_val = float(coord)
            
            # Validate latitude range
            if coord_type == 'lat' and not (-90 <= coord_val <= 90):
                return np.nan
            # Validate longitude range  
            elif coord_type == 'lon' and not (-180 <= coord_val <= 180):
                return np.nan
                
            return coord_val
        except:
            return np.nan
    
    df['latitude'] = df['latitude'].apply(lambda x: clean_coordinate(x, 'lat'))
    df['longitude'] = df['longitude'].apply(lambda x: clean_coordinate(x, 'lon'))
    
    # 6. Fill missing values with appropriate imputation strategies
    print("6. Filling missing values...")
    
    # Fill missing actual_time with scheduled_time (assumption: on-time arrival when missing)
    def impute_missing_time(row):
        if pd.isna(row['actual_time']) and not pd.isna(row['scheduled_time']):
            # Extract time from scheduled_time and format as HH:MM
            scheduled_dt = pd.to_datetime(row['scheduled_time'])
            return scheduled_dt.strftime('%H:%M')
        return row['actual_time']
    
    df['actual_time'] = df.apply(impute_missing_time, axis=1)
    
    # Fill missing weather with mode (most common weather) by route
    print("   - Filling missing weather values...")
    weather_mode_by_route = df.groupby('route_id')['weather'].apply(lambda x: x.mode()[0] if not x.empty and not x.mode().empty else 'Sunny')
    
    def fill_weather(row):
        if pd.isna(row['weather']):
            return weather_mode_by_route.get(row['route_id'], 'Sunny')
        return row['weather']
    
    df['weather'] = df.apply(fill_weather, axis=1)
    
    # Fill missing passenger_count using multiple strategies
    print("   - Filling missing passenger counts...")
    
    # Strategy 1: Use median by route and weather condition
    passenger_medians = df.groupby(['route_id', 'weather'])['passenger_count'].median()
    
    # Strategy 2: Use overall route median as fallback
    route_medians = df.groupby('route_id')['passenger_count'].median()
    
    # Strategy 3: Use overall median as final fallback
    overall_median = df['passenger_count'].median()
    
    def fill_passenger_count(row):
        if pd.isna(row['passenger_count']):
            # Try route + weather combination first
            key = (row['route_id'], row['weather'])
            if key in passenger_medians and not pd.isna(passenger_medians[key]):
                return int(passenger_medians[key])
            
            # Fallback to route median
            if row['route_id'] in route_medians and not pd.isna(route_medians[row['route_id']]):
                return int(route_medians[row['route_id']])
            
            # Final fallback to overall median
            return int(overall_median) if not pd.isna(overall_median) else 150
        return row['passenger_count']
    
    df['passenger_count'] = df.apply(fill_passenger_count, axis=1)
    
    # Fill missing coordinates using route-based averages
    print("   - Filling missing coordinates...")
    
    # Calculate route centroids (average coordinates per route)
    route_coords = df.groupby('route_id')[['latitude', 'longitude']].median()
    
    def fill_coordinates(row):
        lat, lon = row['latitude'], row['longitude']
        route_id = row['route_id']
        
        # If both are missing, use route centroid
        if pd.isna(lat) and pd.isna(lon) and route_id in route_coords.index:
            lat = route_coords.loc[route_id, 'latitude']
            lon = route_coords.loc[route_id, 'longitude']
        
        # If only latitude is missing
        elif pd.isna(lat) and route_id in route_coords.index:
            lat = route_coords.loc[route_id, 'latitude']
        
        # If only longitude is missing
        elif pd.isna(lon) and route_id in route_coords.index:
            lon = route_coords.loc[route_id, 'longitude']
        
        return pd.Series([lat, lon])
    
    df[['latitude', 'longitude']] = df.apply(fill_coordinates, axis=1)
    
    # 7. Handle completely empty rows (after imputation)
    print("7. Removing rows with too many missing values...")
    # Remove rows where more than 50% of values are missing
    threshold = len(df.columns) * 0.5
    df_cleaned = df.dropna(thresh=threshold)
    
    # Also remove rows with missing actual_time (needed for delay calculation)
    print("   - Removing rows with missing actual_time...")
    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['actual_time'].notna()]
    rows_removed = rows_before - len(df_cleaned)
    if rows_removed > 0:
        print(f"   - Removed {rows_removed} rows with missing actual_time")
    
    # 8. Calculate delay_minutes
    print("8. Calculating delay_minutes...")
    
    # Convert scheduled_time to proper datetime format
    df_cleaned['scheduled_time'] = pd.to_datetime(df_cleaned['scheduled_time'])
    
    # Create a combined datetime for actual_time
    df_cleaned['actual_datetime'] = pd.to_datetime(
        df_cleaned['scheduled_time'].dt.date.astype(str) + ' ' + df_cleaned['actual_time'].astype(str),
        errors='coerce'
    )
    
    # Calculate delay in minutes
    df_cleaned['delay_minutes'] = (df_cleaned['actual_datetime'] - df_cleaned['scheduled_time']).dt.total_seconds() / 60
    
    # Handle cases where actual time is next day (e.g., scheduled 23:00, actual 01:00)
    df_cleaned.loc[df_cleaned['delay_minutes'] < -720, 'delay_minutes'] += 1440  # Add 24 hours
    
    print(f"   - Delay range: {df_cleaned['delay_minutes'].min():.1f} to {df_cleaned['delay_minutes'].max():.1f} minutes")
    print(f"   - Average delay: {df_cleaned['delay_minutes'].mean():.1f} minutes")
    
    # Drop the temporary actual_datetime column
    df_cleaned = df_cleaned.drop(columns=['actual_datetime'])
    
    # 9. Data validation and final cleanup
    print("9. Final validation and cleanup...")
    
    # Sort by scheduled_time and route_id for better organization
    df_cleaned = df_cleaned.sort_values(['scheduled_time', 'route_id']).reset_index(drop=True)
    
    # Generate cleaning report
    print("\n=== CLEANING SUMMARY ===")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")
    
    print("\nMissing values after cleaning:")
    missing_summary = df_cleaned.isnull().sum()
    print(missing_summary)
    
    print(f"\nMissing value percentages:")
    missing_percentages = (missing_summary / len(df_cleaned) * 100).round(2)
    print(missing_percentages)
    
    print(f"\nStandardized route IDs: {sorted(df_cleaned['route_id'].dropna().unique())}")
    print(f"Standardized weather values: {sorted(df_cleaned['weather'].dropna().unique())}")
    
    # Data completeness after imputation
    print("\n=== DATA COMPLETENESS AFTER IMPUTATION ===")
    completeness = ((df_cleaned.notna().sum() / len(df_cleaned)) * 100).round(2)
    for col, pct in completeness.items():
        print(f"{col}: {pct}% complete")
    
    # Save cleaned dataset
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to: {output_file}")
    
    return df_cleaned

def analyze_cleaned_data(df):
    """
    Provide analysis of the cleaned dataset
    """
    print("\n=== CLEANED DATA ANALYSIS ===")
    
    # Basic statistics
    print("Dataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['scheduled_time'].min()} to {df['scheduled_time'].max()}")
    print(f"Number of routes: {df['route_id'].nunique()}")
    
    # Route analysis
    print("\nRoute Statistics:")
    route_stats = df.groupby('route_id').agg({
        'passenger_count': ['count', 'mean', 'std'],
        'weather': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).round(2)
    print(route_stats)
    
    # Weather distribution
    print("\nWeather Distribution:")
    weather_counts = df['weather'].value_counts()
    weather_percentages = (weather_counts / len(df) * 100).round(1)
    for weather, count in weather_counts.items():
        print(f"{weather}: {count} ({weather_percentages[weather]}%)")
    
    # Time analysis
    if df['actual_time'].notna().any():
        print(f"\nTime Data Completeness: {(df['actual_time'].notna().sum() / len(df) * 100):.1f}%")
    
    # Location data completeness
    location_complete = df[['latitude', 'longitude']].notna().all(axis=1).sum()

    print(f"Complete location data: {location_complete} records ({location_complete/len(df)*100:.1f}%)")

if __name__ == "__main__":
    try:
        # Check for CSV files in current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if csv_files:
            print("Available CSV files in current directory:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            print()
        
        # Get input file from user
        input_file = input("Enter the input file name (e.g., dirty_transport_dataset.csv): ").strip()
        
        # If user didn't provide extension, try adding .csv
        if not input_file.endswith('.csv'):
            input_file += '.csv'
        
        # Clean the dataset
        print(f"\nAttempting to clean: {input_file}")
        cleaned_df = clean_transport_data(input_file)
        
        # Analyze the cleaned data
        analyze_cleaned_data(cleaned_df)
        
        print("\n=== DATA CLEANING COMPLETED ===")
        print("The cleaned dataset is ready for analysis!")
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please make sure the file exists in the current directory.")
        if csv_files:
            print(f"Available files: {', '.join(csv_files)}")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your input file and try again.")
