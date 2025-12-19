import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_cleaned_data(file_path):
    """Load the cleaned dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        
        # Convert scheduled_time to datetime if needed
        if 'scheduled_time' in df.columns:
            df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        
        # Extract date column if not present
        if 'date' not in df.columns and 'scheduled_time' in df.columns:
            df['date'] = pd.to_datetime(df['scheduled_time']).dt.date
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_info(df):
    """Display basic information about the dataset."""
    print("\n" + "="*80)
    print("BASIC DATASET INFORMATION")
    print("="*80)
    
    print("\nDataset Shape:")
    print(f"  - Rows: {df.shape[0]}")
    print(f"  - Columns: {df.shape[1]}")
    
    print("\nColumn Names and Types:")
    print(df.dtypes)
    
    print("\nFirst Few Rows:")
    print(df.head())
    
    print("\nDataset Info:")
    df.info()
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values!")
    
    print("\nDuplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"  Total duplicates: {duplicates}")

def statistical_summary(df):
    """Generate statistical summary of numerical columns."""
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    print("\nNumerical Columns Summary:")
    print(df.describe())
    
    print("\nCategorical Columns Summary:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

def analyze_delay(df):
    """Analyze delay patterns."""
    print("\n" + "="*80)
    print("DELAY ANALYSIS")
    print("="*80)
    
    if 'delay_minutes' in df.columns:
        print("\nDelay Statistics:")
        print(f"  - Mean delay: {df['delay_minutes'].mean():.2f} minutes")
        print(f"  - Median delay: {df['delay_minutes'].median():.2f} minutes")
        print(f"  - Max delay: {df['delay_minutes'].max():.2f} minutes")
        print(f"  - Min delay: {df['delay_minutes'].min():.2f} minutes")
        print(f"  - Std deviation: {df['delay_minutes'].std():.2f} minutes")
        
        # Delay distribution
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['delay_minutes'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Delays')
        plt.axvline(df['delay_minutes'].mean(), color='red', linestyle='--', label=f'Mean: {df["delay_minutes"].mean():.2f}')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.boxplot(df['delay_minutes'])
        plt.ylabel('Delay (minutes)')
        plt.title('Delay Boxplot')
        
        plt.subplot(1, 3, 3)
        delay_categories = pd.cut(df['delay_minutes'], 
                                  bins=[-np.inf, 0, 5, 10, 20, np.inf],
                                  labels=['Early', '0-5 min', '5-10 min', '10-20 min', '>20 min'])
        delay_categories.value_counts().plot(kind='bar', color='coral', edgecolor='black')
        plt.xlabel('Delay Category')
        plt.ylabel('Count')
        plt.title('Delays by Category')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('delay_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: delay_analysis.png")
        plt.show()

def analyze_by_route(df):
    """Analyze delays by route."""
    print("\n" + "="*80)
    print("ROUTE ANALYSIS")
    print("="*80)
    
    if 'route_id' in df.columns and 'delay_minutes' in df.columns:
        route_stats = df.groupby('route_id')['delay_minutes'].agg([
            ('count', 'count'),
            ('mean_delay', 'mean'),
            ('median_delay', 'median'),
            ('max_delay', 'max'),
            ('min_delay', 'min')
        ]).round(2)
        
        print("\nDelay Statistics by Route:")
        print(route_stats)
        
        # Visualizations
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        route_means = df.groupby('route_id')['delay_minutes'].mean().sort_values()
        route_means.plot(kind='barh', color='skyblue', edgecolor='black')
        plt.xlabel('Average Delay (minutes)')
        plt.ylabel('Route ID')
        plt.title('Average Delay by Route')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='delay_minutes', by='route_id', figsize=(14, 5))
        plt.xlabel('Route ID')
        plt.ylabel('Delay (minutes)')
        plt.title('Delay Distribution by Route')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig('route_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: route_analysis.png")
        plt.show()

def analyze_by_weather(df):
    """Analyze delays by weather condition."""
    print("\n" + "="*80)
    print("WEATHER ANALYSIS")
    print("="*80)
    
    if 'weather' in df.columns and 'delay_minutes' in df.columns:
        weather_stats = df.groupby('weather')['delay_minutes'].agg([
            ('count', 'count'),
            ('mean_delay', 'mean'),
            ('median_delay', 'median'),
            ('std_delay', 'std')
        ]).round(2)
        
        print("\nDelay Statistics by Weather:")
        print(weather_stats)
        
        # Visualizations
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        weather_means = df.groupby('weather')['delay_minutes'].mean().sort_values()
        weather_means.plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.xlabel('Weather Condition')
        plt.ylabel('Average Delay (minutes)')
        plt.title('Average Delay by Weather')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='delay_minutes', by='weather', figsize=(14, 5))
        plt.xlabel('Weather Condition')
        plt.ylabel('Delay (minutes)')
        plt.title('Delay Distribution by Weather')
        plt.suptitle('')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('weather_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: weather_analysis.png")
        plt.show()

def analyze_by_traffic(df):
    """Analyze delays by traffic condition."""
    print("\n" + "="*80)
    print("TRAFFIC ANALYSIS")
    print("="*80)
    
    if 'traffic_conditions' not in df.columns:
        print("\n⚠ 'traffic_conditions' column not found in dataset. Skipping traffic analysis.")
        return
    
    if 'traffic_conditions' in df.columns and 'delay_minutes' in df.columns:
        traffic_stats = df.groupby('traffic_conditions')['delay_minutes'].agg([
            ('count', 'count'),
            ('mean_delay', 'mean'),
            ('median_delay', 'median'),
            ('std_delay', 'std')
        ]).round(2)
        
        print("\nDelay Statistics by Traffic:")
        print(traffic_stats)
        
        # Visualizations
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        traffic_means = df.groupby('traffic_conditions')['delay_minutes'].mean().sort_values()
        traffic_means.plot(kind='bar', color='salmon', edgecolor='black')
        plt.xlabel('Traffic Condition')
        plt.ylabel('Average Delay (minutes)')
        plt.title('Average Delay by Traffic')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='delay_minutes', by='traffic_conditions', figsize=(14, 5))
        plt.xlabel('Traffic Condition')
        plt.ylabel('Delay (minutes)')
        plt.title('Delay Distribution by Traffic')
        plt.suptitle('')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('traffic_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: traffic_analysis.png")
        plt.show()

def correlation_analysis(df):
    """Analyze correlations between variables."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: correlation_heatmap.png")
        plt.show()
        
        # Feature correlations with delay
        if 'delay_minutes' in numerical_cols:
            print("\nCorrelation with Delay:")
            delay_corr = correlation_matrix['delay_minutes'].sort_values(ascending=False)
            print(delay_corr)

def time_analysis(df):
    """Analyze time-based patterns."""
    print("\n" + "="*80)
    print("TIME-BASED ANALYSIS")
    print("="*80)
    
    if 'date' in df.columns:
        # Convert to datetime if not already
        if df['date'].dtype != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Extract time features
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        
        print("\nDate Range:")
        print(f"  - From: {df['date'].min()}")
        print(f"  - To: {df['date'].max()}")
        print(f"  - Duration: {(df['date'].max() - df['date'].min()).days} days")
        
        if 'delay_minutes' in df.columns:
            # Delay by day of week
            plt.figure(figsize=(14, 5))
            
            plt.subplot(1, 2, 1)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_delays = df.groupby('day_of_week')['delay_minutes'].mean().reindex(day_order)
            day_delays.plot(kind='bar', color='mediumpurple', edgecolor='black')
            plt.xlabel('Day of Week')
            plt.ylabel('Average Delay (minutes)')
            plt.title('Average Delay by Day of Week')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            # Trend over time
            daily_avg = df.groupby('date')['delay_minutes'].mean()
            plt.plot(daily_avg.index, daily_avg.values, marker='o', linestyle='-', color='darkorange')
            plt.xlabel('Date')
            plt.ylabel('Average Delay (minutes)')
            plt.title('Delay Trend Over Time')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('time_analysis.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: time_analysis.png")
            plt.show()

def combined_analysis(df):
    """Analyze combined effects of multiple factors."""
    # Check which analyses we can perform
    has_traffic = 'traffic_conditions' in df.columns
    has_delay = 'delay_minutes' in df.columns
    has_route = 'route_id' in df.columns
    has_weather = 'weather' in df.columns
    
    if not (has_route and has_weather and has_delay):
        print("\n⚠ Missing required columns for combined analysis. Skipping.")
        return
    
    print("\n" + "="*80)
    print("COMBINED FACTORS ANALYSIS")
    print("="*80)
    
    if all(col in df.columns for col in ['route_id', 'weather', 'traffic_conditions', 'delay_minutes']):
        # Route + Weather
        print("\nAverage Delay by Route and Weather:")
        pivot_route_weather = df.pivot_table(
            values='delay_minutes',
            index='route_id',
            columns='weather',
            aggfunc='mean'
        ).round(2)
        print(pivot_route_weather)
        
        # Route + Traffic
        print("\nAverage Delay by Route and Traffic:")
        pivot_route_traffic = df.pivot_table(
            values='delay_minutes',
            index='route_id',
            columns='traffic_conditions',
            aggfunc='mean'
        ).round(2)
        print(pivot_route_traffic)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(pivot_route_weather, annot=True, fmt='.1f', cmap='YlOrRd', 
                    ax=axes[0], cbar_kws={'label': 'Avg Delay (min)'})
        axes[0].set_title('Average Delay: Route vs Weather')
        axes[0].set_xlabel('Weather')
        axes[0].set_ylabel('Route ID')
        
    
    elif has_route and has_weather and has_delay:
        # If no traffic data, just show route vs weather
        print("\nAverage Delay by Route and Weather:")
        pivot_route_weather = df.pivot_table(
            values='delay_minutes',
            index='route_id',
            columns='weather',
            aggfunc='mean'
        ).round(2)
        print(pivot_route_weather)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_route_weather, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Avg Delay (min)'})
        plt.title('Average Delay: Route vs Weather')
        plt.xlabel('Weather')
        plt.ylabel('Route ID')
        plt.tight_layout()
        plt.savefig('combined_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: combined_analysis.png")
        plt.show()

def key_insights(df):
    """Generate key insights from the analysis."""
    print("\n" + "="*80)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*80)
    
    insights = []
    
    # Delay insights
    if 'delay_minutes' in df.columns:
        avg_delay = df['delay_minutes'].mean()
        max_delay = df['delay_minutes'].max()
        insights.append(f"✓ Average delay across all routes: {avg_delay:.2f} minutes")
        insights.append(f"✓ Maximum delay recorded: {max_delay:.2f} minutes")
        
        # Routes with highest delays
        if 'route_id' in df.columns:
            worst_route = df.groupby('route_id')['delay_minutes'].mean().idxmax()
            worst_delay = df.groupby('route_id')['delay_minutes'].mean().max()
            insights.append(f"✓ Route with highest average delay: {worst_route} ({worst_delay:.2f} min)")
        
        # Weather impact
        if 'weather' in df.columns:
            worst_weather = df.groupby('weather')['delay_minutes'].mean().idxmax()
            weather_delay = df.groupby('weather')['delay_minutes'].mean().max()
            insights.append(f"✓ Weather condition with highest delays: {worst_weather} ({weather_delay:.2f} min)")
        
        # Traffic impact
        if 'traffic_conditions' in df.columns:
            worst_traffic = df.groupby('traffic_conditions')['delay_minutes'].mean().idxmax()
            traffic_delay = df.groupby('traffic_conditions')['delay_minutes'].mean().max()
            insights.append(f"✓ Traffic condition with highest delays: {worst_traffic} ({traffic_delay:.2f} min)")
    
    print("\n")
    for insight in insights:
        print(insight)
    
    print("\n\nRECOMMENDATIONS FOR MODELING:")
    print("1. Focus on routes with high variability in delays")
    print("2. Weather and traffic are likely important predictive features")
    print("3. Consider time-based features (day of week, time of day)")
    print("4. May need to handle outliers or extreme delays separately")
    print("5. Consider ensemble methods given multiple influential factors")

def main():
    """Main function to run all EDA analyses."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - TRANSPORT DELAY PREDICTION")
    print("="*80)
    
    # Load data
    df = load_cleaned_data('dirty_transport_dataset_cleaned.csv')
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Run all analyses
    basic_info(df)
    statistical_summary(df)
    analyze_delay(df)
    analyze_by_route(df)
    analyze_by_weather(df)
    analyze_by_traffic(df)
    correlation_analysis(df)
    time_analysis(df)
    combined_analysis(df)
    key_insights(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  - delay_analysis.png")
    print("  - route_analysis.png")
    print("  - weather_analysis.png")
    print("  - traffic_analysis.png")
    print("  - correlation_heatmap.png")
    print("  - time_analysis.png")
    print("  - combined_analysis.png")
 
if __name__ == "__main__":
    main()
