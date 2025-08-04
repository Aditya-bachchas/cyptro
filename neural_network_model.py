import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        self.history = None
        
    def load_data(self):
        """Load processed sales data"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print(f"Loaded {len(self.data)} records for neural network training")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare features for neural network training"""
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
    
    def build_model(self, architecture='complex'):
        """Build neural network model"""
        if not hasattr(self, 'X_train'):
            print("Please prepare features first.")
            return False
        
        input_dim = self.X_train.shape[1]
        
        if architecture == 'simple':
            # Simple architecture
            self.model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_dim=input_dim),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='linear')
            ])
        elif architecture == 'complex':
            # Complex architecture for better performance
            self.model = keras.Sequential([
                keras.layers.Dense(256, activation='relu', input_dim=input_dim),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(64, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                
                keras.layers.Dense(32, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.1),
                
                keras.layers.Dense(1, activation='linear')
            ])
        else:
            # Advanced architecture with residual connections
            inputs = keras.layers.Input(shape=(input_dim,))
            
            # First dense block
            x = keras.layers.Dense(256, activation='relu')(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            
            # Residual connection
            residual = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.Add()([x, residual])
            x = keras.layers.BatchNormalization()(x)
            
            # Second dense block
            x = keras.layers.Dense(128, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            
            # Third dense block
            x = keras.layers.Dense(64, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
            
            # Output
            outputs = keras.layers.Dense(1, activation='linear')(x)
            
            self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Neural network model built with {architecture} architecture")
        self.model.summary()
        
        return True
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the neural network model"""
        if self.model is None:
            print("No model built. Please build model first.")
            return False
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Learning rate reduction
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
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
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
        """Analyze feature importance using permutation importance"""
        if self.model is None:
            print("No trained model available.")
            return
        
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        result = permutation_importance(
            self.model, self.X_test, self.y_test, 
            n_repeats=10, random_state=42
        )
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
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
    
    def save_model(self, filename='sales_prediction_model.h5'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filename)
            print(f"Model saved to {filename}")
        else:
            print("No model to save.")
    
    def load_model(self, filename='sales_prediction_model.h5'):
        """Load a trained model"""
        try:
            self.model = keras.models.load_model(filename)
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
        # This is a placeholder for demonstration
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
        
        return prediction[0][0]
    
    def run_complete_pipeline(self):
        """Run the complete neural network pipeline"""
        print("Starting Neural Network Sales Prediction Pipeline...")
        
        if not self.load_data():
            return False
        
        if not self.prepare_features():
            return False
        
        if not self.build_model(architecture='complex'):
            return False
        
        if not self.train_model(epochs=50):
            return False
        
        # Evaluate and visualize
        metrics = self.evaluate_model()
        self.plot_training_history()
        self.plot_predictions()
        self.feature_importance_analysis()
        
        # Save model
        self.save_model()
        
        print("Neural Network Pipeline Completed Successfully!")
        return True

if __name__ == "__main__":
    # Run the complete neural network pipeline
    nn_model = SalesNeuralNetwork()
    nn_model.run_complete_pipeline() 