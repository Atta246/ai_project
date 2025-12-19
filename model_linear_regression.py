import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the preprocessed training and testing data."""
    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    print("\n" + "="*80)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("="*80)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully")
    print(f"  - Number of features: {X_train.shape[1]}")
    print(f"  - Training samples: {X_train.shape[0]}")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance."""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Testing metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Performance:")
    print(f"  - RMSE: {train_rmse:.4f}")
    print(f"  - MAE:  {train_mae:.4f}")
    print(f"  - R²:   {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  - RMSE: {test_rmse:.4f}")
    print(f"  - MAE:  {test_mae:.4f}")
    print(f"  - R²:   {test_r2:.4f}")
    
    # Check for overfitting
    overfitting = train_r2 - test_r2
    print(f"\nOverfitting Check:")
    print(f"  - R² difference: {overfitting:.4f}")
    if overfitting > 0.1:
        print("  ⚠ Model may be overfitting")
    else:
        print("  ✓ Model generalization looks good")
    
    return {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance based on coefficients."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    # Sort by absolute value
    coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, row in coefficients.head(10).iterrows():
        print(f"  {row['Feature']:30s} {row['Coefficient']:10.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    top_features = coefficients.head(15)
    plt.barh(top_features['Feature'], top_features['Coefficient'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Feature Coefficients - Linear Regression')
    plt.tight_layout()
    plt.savefig('linear_regression_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: linear_regression_feature_importance.png")
    plt.close()
    
    return coefficients

def visualize_predictions(y_test, y_test_pred):
    """Visualize actual vs predicted values."""
    print("\n" + "="*80)
    print("PREDICTION VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Delay (minutes)')
    axes[0, 0].set_ylabel('Predicted Delay (minutes)')
    axes[0, 0].set_title('Actual vs Predicted Delay')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, edgecolors='k')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Delay (minutes)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Prediction error distribution
    prediction_errors = abs(residuals)
    axes[1, 1].hist(prediction_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].set_xlabel('Absolute Error (minutes)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Absolute Errors')
    axes[1, 1].axvline(x=prediction_errors.mean(), color='r', linestyle='--', 
                      lw=2, label=f'Mean: {prediction_errors.mean():.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: linear_regression_predictions.png")
    plt.close()

def save_model(model, metrics):
    """Save the trained model and metrics."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    # Save model
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Saved: linear_regression_model.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'Model': 'Linear Regression',
        'Train_RMSE': metrics['train_rmse'],
        'Train_MAE': metrics['train_mae'],
        'Train_R2': metrics['train_r2'],
        'Test_RMSE': metrics['test_rmse'],
        'Test_MAE': metrics['test_mae'],
        'Test_R2': metrics['test_r2']
    }])
    metrics_df.to_csv('linear_regression_metrics.csv', index=False)
    print("✓ Saved: linear_regression_metrics.csv")

def main():
    """Main function to train and evaluate Linear Regression model."""
    print("\n" + "="*80)
    print("LINEAR REGRESSION MODEL - TRANSPORT DELAY PREDICTION")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, X_train.columns)
    
    # Visualize predictions
    visualize_predictions(y_test, metrics['y_test_pred'])
    
    # Save model
    save_model(model, metrics)
    
    print("\n" + "="*80)
    print("LINEAR REGRESSION MODEL TRAINING COMPLETE!")
    print("="*80)
    print("\nModel Summary:")
    print(f"  - Test RMSE: {metrics['test_rmse']:.4f} minutes")
    print(f"  - Test MAE:  {metrics['test_mae']:.4f} minutes")
    print(f"  - Test R²:   {metrics['test_r2']:.4f}")

if __name__ == "__main__":
    main()
