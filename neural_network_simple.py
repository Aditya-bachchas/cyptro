import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesNeuralNetwork:
    def __init__(self, data_path='processed_sales_data.csv'):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'Final_Amount'
        
    def load_data(self):
        """Load processed sales data"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print(f"Loaded {len(self.data)} records for machine learning training")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare features for machine learning training"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        df = self.data.copy()
        
        # Select relevant features for prediction
        feature_columns = [
            'Price', 'Quantity', 'Discount_Rate', 'Year', 'Month', 'Quarter',
            'Day_of_Year', 'Week_of_Year', 'Is_Weekend', 'Is_Holiday_Season', 'Is_Summer',
            'Total_Spent', 'Order_Count', 'Avg_Order_Value', 'Customer_Lifetime_Days',
            'Product_Revenue', 'Product_Orders', 'Product_Avg_Price', 'Product_Quantity_Sold',
            'Category_Revenue', 'Category_Orders', 'Category_Avg_Price', 'Category_Quantity_Sold',
            'Region_Revenue', 'Region_Orders', 'Region_Avg_Order', 'Region_Customers',
            'Channel_Revenue', 'Channel_Orders', 'Channel_Avg_Order',
            'Payment_Revenue', 'Payment_Orders', 'Payment_Avg_Order',
            'Segment_Revenue', 'Segment_Orders', 'Segment_Avg_Order', 'Segment_Customers',
            'Estimated_Cost', 'Estimated_Profit', 'Profit_Margin'
        ]
        
        # Categorical columns to encode
        categorical_columns = [
            'Customer_Segment', 'Region', 'Category', 'Product', 'Sales_Channel', 
            'Payment_Method', 'Customer_Value_Tier', 'Day_of_Week'
        ]
        
        # Create feature dataframe
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Add encoded categorical features
        for col in categorical_columns:
            if col in df.columns:
                X[col] = df[col].astype('category').cat.codes
        
        # Target variable
        y = df[self.target_column].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_columns = feature_columns
        
        print(f"Feature preparation completed:")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {len(feature_columns)}")
        
        return True
    
    def build_model(self, model_type='random_forest'):
        """Build machine learning model"""
        if not hasattr(self, 'X_train'):
            print("Please prepare features first.")
            return False
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            print(f"Unknown model type: {model_type}")
            return False
        
        print(f"Machine learning model built: {model_type}")
        return True
    
    def train_model(self):
        """Train the machine learning model"""
        if self.model is None:
            print("No model built. Please build model first.")
            return False
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed!")
        return True
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None:
            print("No trained model available.")
            return None
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        print("Model Performance Metrics:")
        print(f"  - Mean Squared Error: {mse:.2f}")
        print(f"  - Root Mean Squared Error: {np.sqrt(mse):.2f}")
        print(f"  - Mean Absolute Error: {mae:.2f}")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - Mean Absolute Percentage Error: {mape:.2f}%")
        
        return metrics
    
    def plot_predictions(self):
        """Plot actual vs predicted values"""
        if self.model is None:
            print("No trained model available.")
            return
        
        y_pred = self.model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Sales Amount')
        plt.ylabel('Predicted Sales Amount')
        plt.title('Actual vs Predicted Sales Amounts')
        plt.grid(True)
        plt.savefig('predictions_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if self.model is None:
            print("No trained model available.")
            return
        
        if hasattr(self.model, 'feature_importances_'):
            # For Random Forest
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            # For Linear Regression
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': np.abs(self.model.coef_)
            }).sort_values('Importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return importance_df
    
    def save_model(self, filename='sales_prediction_model.pkl'):
        """Save the trained model"""
        if self.model is not None:
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {filename}")
        else:
            print("No model to save.")
    
    def load_model(self, filename='sales_prediction_model.pkl'):
        """Load a trained model"""
        try:
            import pickle
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_sales(self, input_data):
        """Make sales predictions for new data"""
        if self.model is None:
            print("No trained model available.")
            return None
        
        # Preprocess input data (simplified - would need proper feature engineering)
        if isinstance(input_data, dict):
            # Convert dict to array format
            features = []
            for col in self.feature_columns:
                features.append(input_data.get(col, 0))
            input_array = np.array([features])
        else:
            input_array = input_data
        
        # Scale the input
        input_scaled = self.scaler.transform(input_array)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        
        return prediction[0]
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("Starting Machine Learning Sales Prediction Pipeline...")
        
        if not self.load_data():
            return False
        
        if not self.prepare_features():
            return False
        
        if not self.build_model(model_type='random_forest'):
            return False
        
        if not self.train_model():
            return False
        
        # Evaluate and visualize
        metrics = self.evaluate_model()
        self.plot_predictions()
        self.feature_importance_analysis()
        
        # Save model
        self.save_model()
        
        print("Machine Learning Pipeline Completed Successfully!")
        return True

if __name__ == "__main__":
    # Run the complete machine learning pipeline
    ml_model = SalesNeuralNetwork()
    ml_model.run_complete_pipeline() 