import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_all_metrics():
    """Load metrics from all trained models."""
    print("Loading metrics from all models...")
    
    models_data = []
    
    try:
        lr_metrics = pd.read_csv('linear_regression_metrics.csv')
        models_data.append(lr_metrics)
        print("✓ Linear Regression metrics loaded")
    except:
        print("⚠ Linear Regression metrics not found")
    
    try:
        rf_metrics = pd.read_csv('random_forest_metrics.csv')
        models_data.append(rf_metrics)
        print("✓ Random Forest metrics loaded")
    except:
        print("⚠ Random Forest metrics not found")
    
    try:
        gb_metrics = pd.read_csv('gradient_boosting_metrics.csv')
        models_data.append(gb_metrics)
        print("✓ Gradient Boosting metrics loaded")
    except:
        print("⚠ Gradient Boosting metrics not found")
    
    if not models_data:
        raise ValueError("No model metrics found. Please train at least one model first.")
    
    all_metrics = pd.concat(models_data, ignore_index=True)
    return all_metrics

def compare_models(metrics_df):
    """Compare all models and display results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print("\nTest Set Performance:")
    print(metrics_df[['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2']].to_string(index=False))
    
    print("\nTraining Set Performance:")
    print(metrics_df[['Model', 'Train_RMSE', 'Train_MAE', 'Train_R2']].to_string(index=False))
    
    # Find best model
    best_model_rmse = metrics_df.loc[metrics_df['Test_RMSE'].idxmin(), 'Model']
    best_model_mae = metrics_df.loc[metrics_df['Test_MAE'].idxmin(), 'Model']
    best_model_r2 = metrics_df.loc[metrics_df['Test_R2'].idxmax(), 'Model']
    
    print("\n" + "="*80)
    print("BEST MODELS")
    print("="*80)
    print(f"Best RMSE: {best_model_rmse}")
    print(f"Best MAE:  {best_model_mae}")
    print(f"Best R²:   {best_model_r2}")

def visualize_comparison(metrics_df):
    """Create visualizations comparing all models."""
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RMSE Comparison
    axes[0, 0].bar(metrics_df['Model'], metrics_df['Test_RMSE'], 
                   color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Test RMSE Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. MAE Comparison
    axes[0, 1].bar(metrics_df['Model'], metrics_df['Test_MAE'], 
                   color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Test MAE Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. R² Comparison
    axes[1, 0].bar(metrics_df['Model'], metrics_df['Test_R2'], 
                   color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Test R² Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Overfitting Analysis (Train vs Test R²)
    x = np.arange(len(metrics_df))
    width = 0.35
    axes[1, 1].bar(x - width/2, metrics_df['Train_R2'], width, 
                   label='Train R²', color='skyblue', edgecolor='black')
    axes[1, 1].bar(x + width/2, metrics_df['Test_R2'], width, 
                   label='Test R²', color='salmon', edgecolor='black')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Train vs Test R² (Overfitting Check)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()
    
    # Create detailed metrics table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = metrics_df[['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2', 
                              'Train_RMSE', 'Train_MAE', 'Train_R2']].round(4)
    table = ax.table(cellText=table_data.values, 
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig('model_metrics_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_metrics_table.png")
    plt.close()

def save_comparison_report(metrics_df):
    """Save a comprehensive comparison report."""
    print("\n" + "="*80)
    print("SAVING COMPARISON REPORT")
    print("="*80)
    
    # Save combined metrics
    metrics_df.to_csv('all_models_comparison.csv', index=False)
    print("✓ Saved: all_models_comparison.csv")
    
    # Create summary report
    with open('model_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT - TRANSPORT DELAY PREDICTION\n")
        f.write("="*80 + "\n\n")
        
        f.write("TEST SET PERFORMANCE:\n")
        f.write("-"*80 + "\n")
        f.write(metrics_df[['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2']].to_string(index=False))
        f.write("\n\n")
        
        f.write("TRAINING SET PERFORMANCE:\n")
        f.write("-"*80 + "\n")
        f.write(metrics_df[['Model', 'Train_RMSE', 'Train_MAE', 'Train_R2']].to_string(index=False))
        f.write("\n\n")
        
        # Best models
        best_rmse = metrics_df.loc[metrics_df['Test_RMSE'].idxmin()]
        best_mae = metrics_df.loc[metrics_df['Test_MAE'].idxmin()]
        best_r2 = metrics_df.loc[metrics_df['Test_R2'].idxmax()]
        
        f.write("BEST MODELS BY METRIC:\n")
        f.write("-"*80 + "\n")
        f.write(f"Best RMSE: {best_rmse['Model']} ({best_rmse['Test_RMSE']:.4f})\n")
        f.write(f"Best MAE:  {best_mae['Model']} ({best_mae['Test_MAE']:.4f})\n")
        f.write(f"Best R²:   {best_r2['Model']} ({best_r2['Test_R2']:.4f})\n")
        f.write("\n")
        
        # Overfitting analysis
        f.write("OVERFITTING ANALYSIS:\n")
        f.write("-"*80 + "\n")
        for _, row in metrics_df.iterrows():
            overfit = row['Train_R2'] - row['Test_R2']
            status = "✓ Good" if overfit < 0.1 else "⚠ Possible overfitting"
            f.write(f"{row['Model']:20s} R² difference: {overfit:.4f} {status}\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        f.write(f"1. The {best_r2['Model']} model shows the best overall performance (R²).\n")
        f.write(f"2. For minimum prediction error, use {best_mae['Model']} (lowest MAE).\n")
        f.write(f"3. Consider ensemble methods combining multiple models for production.\n")
        
    print("✓ Saved: model_comparison_report.txt")

def main():
    """Main function to compare all trained models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON AND ANALYSIS")
    print("="*80)
    
    # Load metrics
    metrics_df = load_all_metrics()
    
    # Compare models
    compare_models(metrics_df)
    
    # Visualize comparison
    visualize_comparison(metrics_df)
    
    # Save comparison report
    save_comparison_report(metrics_df)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - model_comparison.png")
    print("  - model_metrics_table.png")
    print("  - all_models_comparison.csv")
    print("  - model_comparison_report.txt")

if __name__ == "__main__":
    main()
