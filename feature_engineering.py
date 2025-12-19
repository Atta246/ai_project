import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the cleaned dataset."""
    print("Loading cleaned dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def create_time_features(df):
    """Extract time-based features from scheduled_time."""
    print("\n" + "="*80)
    print("CREATING TIME-BASED FEATURES")
    print("="*80)
    
    # Convert to datetime if not already
    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    
    # Extract hour
    df['hour'] = df['scheduled_time'].dt.hour
    print("✓ Created 'hour' (0-23)")
    
    # Extract day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['scheduled_time'].dt.dayofweek
    print("✓ Created 'day_of_week' (0=Monday, 6=Sunday)")
    
    # Extract month
    df['month'] = df['scheduled_time'].dt.month
    print("✓ Created 'month' (1-12)")
    
    # Extract day of month
    df['day'] = df['scheduled_time'].dt.day
    print("✓ Created 'day' (1-31)")
    
    # Is weekend (1 for Saturday/Sunday, 0 otherwise)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    print("✓ Created 'is_weekend' (0/1)")
    
    # Time of day categories
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
    print("✓ Created 'time_of_day' (Morning/Afternoon/Evening/Night)")
    
    # Peak hours (7-9 AM and 5-7 PM)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    print("✓ Created 'is_peak_hour' (0/1)")
    
    # Cyclic encoding for hour (captures circular nature of time)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print("✓ Created 'hour_sin' and 'hour_cos' (cyclic time encoding)")
    
    # Day of week cyclic encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    print("✓ Created cyclic day of week encoding")
    
    return df

def create_route_features(df):
    """Create route-based statistical features."""
    print("\n" + "="*80)
    print("CREATING ROUTE-BASED FEATURES")
    print("="*80)
    
    # Average passenger count per route (no leakage)
    route_avg_passengers = df.groupby('route_id')['passenger_count'].mean()
    df['route_avg_passengers'] = df['route_id'].map(route_avg_passengers)
    print("✓ Created 'route_avg_passengers' (typical route capacity)")
    
    # Route frequency (number of trips)
    route_frequency = df.groupby('route_id').size()
    df['route_frequency'] = df['route_id'].map(route_frequency)
    print("✓ Created 'route_frequency' (number of trips per route)")
    
    return df

def create_weather_features(df):
    """Create weather-based features."""
    print("\n" + "="*80)
    print("CREATING WEATHER-BASED FEATURES")
    print("="*80)
    
    # Binary features for specific weather conditions (no leakage)
    df['is_rainy'] = (df['weather'] == 'Rainy').astype(int)
    df['is_sunny'] = (df['weather'] == 'Sunny').astype(int)
    df['is_cloudy'] = (df['weather'] == 'Cloudy').astype(int)
    print("✓ Created binary weather flags (is_rainy, is_sunny, is_cloudy)")
    
    return df

def create_passenger_features(df):
    """Create passenger-related features."""
    print("\n" + "="*80)
    print("CREATING PASSENGER-BASED FEATURES")
    print("="*80)
    
    # Passenger load relative to route average
    df['passenger_ratio'] = df['passenger_count'] / df['route_avg_passengers']
    print("✓ Created 'passenger_ratio' (current vs average passengers)")
    
    # Passenger count categories
    def categorize_passenger_load(count):
        if count < 50:
            return 'Low'
        elif count < 100:
            return 'Medium'
        elif count < 150:
            return 'High'
        else:
            return 'Very High'
    
    df['passenger_load'] = df['passenger_count'].apply(categorize_passenger_load)
    print("✓ Created 'passenger_load' (Low/Medium/High/Very High)")
    
    # Polynomial features for passenger count (capture non-linear relationships)
    df['passenger_count_squared'] = df['passenger_count'] ** 2
    df['passenger_count_log'] = np.log1p(df['passenger_count'])
    print("✓ Created polynomial passenger features")
    
    return df

def create_interaction_features(df):
    """Create interaction features between multiple variables."""
    print("\n" + "="*80)
    print("CREATING INTERACTION FEATURES")
    print("="*80)
    
    # Weather + Peak Hour interaction
    df['rainy_peak'] = df['is_rainy'] * df['is_peak_hour']
    print("✓ Created 'rainy_peak' (rainy weather during peak hours)")
    
    # Weather + Weekend interaction
    df['rainy_weekend'] = df['is_rainy'] * df['is_weekend']
    print("✓ Created 'rainy_weekend' (rainy weather on weekends)")
    
    # Passenger count + Peak hour
    df['passengers_peak'] = df['passenger_count'] * df['is_peak_hour']
    print("✓ Created 'passengers_peak' (passenger count during peak hours)")
    # Route + Weather interaction
    df['route_weather'] = df['route_id_encoded'] * df['weather_encoded']
    print("✓ Created 'route_weather' (route-weather interaction)")
    
    # Route + Hour interaction
    df['route_hour'] = df['route_id_encoded'] * df['hour']
    print("✓ Created 'route_hour' (route-time interaction)")
    
    # Weather + Hour interaction  
    df['weather_hour'] = df['weather_encoded'] * df['hour']
    print("✓ Created 'weather_hour' (weather-time interaction)")
    
    
    # High passenger load + Peak hour
    df['high_load_peak'] = ((df['passenger_count'] > 150).astype(int)) * df['is_peak_hour']
    print("✓ Created 'high_load_peak' (high passenger load during peak hours)")
    
    return df

def encode_categorical_features(df):
    """Encode categorical variables."""
    print("\n" + "="*80)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*80)
    
    # Label encode route_id
    le_route = LabelEncoder()
    df['route_id_encoded'] = le_route.fit_transform(df['route_id'])
    print(f"✓ Encoded 'route_id': {len(le_route.classes_)} unique routes")
    
    # Label encode weather
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    print(f"✓ Encoded 'weather': {len(le_weather.classes_)} weather types")
    
    # Label encode time_of_day
    le_time = LabelEncoder()
    df['time_of_day_encoded'] = le_time.fit_transform(df['time_of_day'])
    print(f"✓ Encoded 'time_of_day': {len(le_time.classes_)} time periods")
    
    # Label encode passenger_load
    le_load = LabelEncoder()
    df['passenger_load_encoded'] = le_load.fit_transform(df['passenger_load'])
    print(f"✓ Encoded 'passenger_load': {len(le_load.classes_)} load categories")
    
    # Save encoders for future use
    encoders = {
        'route_id': le_route,
        'weather': le_weather,
        'time_of_day': le_time,
        'passenger_load': le_load
    }
    
    return df, encoders

def select_features_for_modeling(df):
    """Select final features for modeling."""
    print("\n" + "="*80)
    print("FEATURE SELECTION FOR MODELING")
    print("="*80)
    
    # Define feature columns (X) and target (y)
    feature_columns = [
        # Encoded categorical features
        'route_id_encoded',
        'weather_encoded',
        'time_of_day_encoded',
        'passenger_load_encoded',
        
        # Numerical features
        'passenger_count',
        'latitude',
        'longitude',
        'hour',
        'day_of_week',
        'month',
        'day',
        
        # Cyclic time features
        'hour_sin',
        'hour_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        
        # Binary features
        'is_weekend',
        'is_peak_hour',
        'is_rainy',
        'is_sunny',
        'is_cloudy',
        
        # Statistical features (no leakage)
        'route_avg_passengers',
        'route_frequency',
        'passenger_ratio',
        
        # Polynomial features
        'passenger_count_squared',
        'passenger_count_log',
        
        # Interaction features
        'rainy_peak',
        'rainy_weekend',
        'passengers_peak',
        'high_load_peak',
        'route_weather',
        'route_hour',
        'weather_hour'
    ]
    
    target_column = 'delay_minutes'
    
    print(f"\nSelected {len(feature_columns)} features for modeling:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nTarget variable: {target_column}")
    print("\n✓ Added cyclic encodings and interaction features")
    print("✓ Added polynomial features for non-linear relationships")
    print("⚠ NOTE: Data has very low feature-target correlation")
    print("  Model performance may be limited by data quality")
    
    return feature_columns, target_column

def scale_features(X_train, X_test):
    """Scale numerical features."""
    print("\n" + "="*80)
    print("SCALING FEATURES")
    print("="*80)
    print("\n⚠ NOTE: Removed route_avg_delay, route_delay_std, and weather_avg_delay")
    print("  to prevent data leakage (using target information in features)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Applied StandardScaler to training and test sets")
    print(f"  - Training set shape: {X_train_scaled.shape}")
    print(f"  - Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def create_train_test_split(df, feature_columns, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT")
    print("="*80)
    
    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Split data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(df, X_train_scaled, X_test_scaled, y_train, y_test, 
                        feature_columns, encoders, scaler):
    """Save processed data and artifacts."""
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    # Save full processed dataset
    df.to_csv('processed_transport_dataset.csv', index=False)
    print("✓ Saved: processed_transport_dataset.csv")
    
    # Save train-test splits as CSV for reference
    pd.DataFrame(X_train_scaled, columns=feature_columns).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_columns).to_csv('X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['delay_minutes']).to_csv('y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['delay_minutes']).to_csv('y_test.csv', index=False)
    print("✓ Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    
    # Save artifacts using pickle
    import pickle
    
    artifacts = {
        'feature_columns': feature_columns,
        'encoders': encoders,
        'scaler': scaler
    }
    
    with open('feature_engineering_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("✓ Saved: feature_engineering_artifacts.pkl (encoders and scaler)")

def display_summary(df, feature_columns):
    """Display summary of feature engineering."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    
    print(f"\nOriginal columns: route_id, scheduled_time, actual_time, weather, " +
          f"passenger_count, latitude, longitude, delay_minutes")
    print(f"Total features created: {len(feature_columns)}")
    print(f"Final dataset shape: {df.shape}")
    
    print("\nFeature Categories:")
    print(f"  - Time-based features: 8")
    print(f"  - Route-based features: 5")
    print(f"  - Weather-based features3 (removed leakage features)")
    print(f"  - Weather-based features: 3
    print(f"  - Interaction features: 4")
    print(f"  - Encoded categorical: 4")
    
    print("\nMissing values in final dataset:")
    missing = df[feature_columns].isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  ✓ No missing values!")
    
    print("\nData is ready for model training!")

def main():
    """Main function to run feature engineering pipeline."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING PIPELINE - TRANSPORT DELAY PREDICTION")
    print("="*80)
    
    # Load data
    df = load_data('dirty_transport_dataset_cleaned.csv')
    
    # Create features
    df = create_time_features(df)
    df = create_route_features(df)
    df = create_weather_features(df)
    df = create_passenger_features(df)
    df = create_interaction_features(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Select features for modeling
    feature_columns, target_column = select_features_for_modeling(df)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        df, feature_columns, target_column, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save processed data
    save_processed_data(df, X_train_scaled, X_test_scaled, y_train, y_test,
                       feature_columns, encoders, scaler)
    
    # Display summary
    display_summary(df, feature_columns)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*80)
    print("\nNext step: Model training and evaluation")

if __name__ == "__main__":
    main()
