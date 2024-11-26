import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from keras.models import save_model
import pandas_market_calendars as mcal
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class StockPricePredictor:
    
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.model = None
        self.stock_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.scaled_data = None
        self.market_calendar = mcal.get_calendar('NYSE')


    def download_data(self):
        print(f"\nDownloading stock data for {self.stock_symbol}...")
        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)
        self.stock_data = yf.download(self.stock_symbol, start=start, end=end)
        self.stock_data = self.stock_data.droplevel('Ticker', axis=1)
        print("Data downloaded successfully!")


    def prepare_data(self):
        print("\nPreparing data for model training...")

        self.scaled_data = self.scaler.fit_transform(self.stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close']])

        x_data = []
        y_data = []

        for i in range(100, len(self.scaled_data)):
            x_data.append(self.scaled_data[i-100:i]) 
            y_data.append(self.scaled_data[i]) 

        x_data, y_data = np.array(x_data), np.array(y_data)
        self.x_data = x_data 
        self.y_data = y_data 

        # Splitting the data into train and test sets (70% for training, 30% for testing)
        splitting_len = int(len(x_data) * 0.7)
        self.x_train = x_data[:splitting_len]
        self.y_train = y_data[:splitting_len]
        self.x_test = x_data[splitting_len:]
        self.y_test = y_data[splitting_len:]

        print("Data preparation completed!")


    def build_and_train_model(self):
        print("\nBuilding and training the model...")

        self.model = Sequential()

        self.model.add(Input(shape=(self.x_train.shape[1], self.x_train.shape[2])))  
        
        # LSTM layers
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(LSTM(64, return_sequences=False))
        # self.model.add(LSTM(32, return_sequences=False))
 
        # Dense layers for learning the non-linear relationship
        self.model.add(Dense(25,activation='relu'))
        # self.model.add(Dense(10,activation='relu'))
        self.model.add(Dense(5,activation='relu'))  

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(self.x_train, self.y_train, batch_size=10, epochs=20)
        print("Model training completed!")


    def save_model(self, model_filename="stock_price_model.keras"):
        self.model.save(model_filename)
        print(f"\nModel saved to {model_filename}.")


    def calculate_rmse(self):
        predictions = self.model.predict(self.x_test)
    
        # Inverse transform the predictions and actual values to get original stock prices
        inv_pred = self.scaler.inverse_transform(predictions)
        inv_y_test = self.scaler.inverse_transform(self.y_test)
    
        # Calculate RMSE for each column
        rmse_values = []
        columns = ['Open', 'High', 'Low', 'Close', 'Adj Close'] 
        
        for i in range(len(columns)): 
            # column_rmse = np.sqrt(np.mean((inv_pred[:, i] - inv_y_test[:, i]) ** 2))  # RMSE for the i-th column
            # column_rmse=max(column_rmse, 0)
            # rmse_values.append(column_rmse)
            mse = mean_squared_error(inv_y_test[:, i], inv_pred[:, i])
            rmse = np.sqrt(mse)  # Taking the square root of MSE to get RMSE
            
            # Ensure RMSE is non-negative
            rmse_values.append(max(rmse, 0))
        
        # Now plot RMSE values using plot_rmse function
        self.plot_rmse(rmse_values, columns)


    def plot_rmse(self, rmse_values, columns):
        print("\n\n")
        fig, axes = plt.subplots(1, len(rmse_values), figsize=(18, 6))  # Adjust figsize as needed

        # Loop through each column and plot a pie chart on the corresponding axis
        for i, ax in enumerate(axes):
            accuracy = 100 - rmse_values[i] 
            sizes = [rmse_values[i], accuracy]  
            labels = [f"RMSE: {rmse_values[i]:.2f}", f"Accuracy: {accuracy:.2f}%"]  
            colors = ['red', 'green']  

            # Plot pie chart on the given axis
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
            ax.axis('equal') 
            ax.set_title(f"RMSE and Accuracy for '{columns[i]}'", fontsize=12)

        plt.tight_layout(pad=4.0)  
        plt.subplots_adjust(top=0.85)  
        plt.savefig('rmse_plot.png')
        # plt.show()
        
        
    def get_next_trading_day(self, start_date):
        next_day = start_date + timedelta(days=1)
        
        while not self.market_calendar.valid_days(start_date=next_day, end_date=next_day).size:
            next_day += timedelta(days=1)  
        return next_day


    def predict_next_7_days(self):
        print("\nPredicting next 7 trading days...")

        # Prepare data to predict future 7 days (using 5 features)
        last_100_days = self.x_data[-1].reshape(1, 100, 5)  # 5 features (Open, High, Low, Close, Adj Close)
        predicted_prices = []

        # Predict prices for the next 7 days
        for _ in range(7):
            pred_price = self.model.predict(last_100_days) 
            predicted_prices.append(pred_price[0]) 
            last_100_days = np.append(last_100_days[:, 1:, :], pred_price.reshape(1, 1, 5), axis=1)

        self.predicted_prices = self.scaler.inverse_transform(np.array(predicted_prices))

        last_date = self.stock_data.index[-1]
        
        # Generate 7 trading dates starting from the day after the last known date
        self.forecast_dates = []
        current_date = last_date
        for _ in range(7):
            current_date = self.get_next_trading_day(current_date)  # Get the next valid trading day
            self.forecast_dates.append(current_date)  # Add it to the forecast dates

        self.prediction_df = pd.DataFrame(self.predicted_prices, columns=['Open', 'High', 'Low', 'Close', 'Adj Close'], index=self.forecast_dates)

        # Print the prediction DataFrame
        print("\nPredicted stock prices for the next 7 trading days:\n")
        self.prediction_df.index = pd.to_datetime(self.prediction_df.index).date
        # self.prediction_df.to_csv('predicted_stock_prices.csv')
        print(self.prediction_df)
        return self.prediction_df


    def plot_predictions(self,prediction_df):
        columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        plt.figure(figsize=(12, 8))
    
        # Loop through each column in prediction_df and plot the predicted values
        for i, column in enumerate(columns):
            plt.subplot(3, 2, i + 1) 
            
            # Plot the predicted values for the valid dates in prediction_df
            plt.plot(prediction_df.index, prediction_df[column], label=f'Predicted {column}', color='orange', marker='x')
            
            # Title and labels
            plt.title(f'{column} - Predicted Prices', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            
            plt.xticks(rotation=45)
            
            plt.legend(loc='upper left', fontsize=10)
    
        plt.tight_layout()
        # plt.show()
        plt.savefig('predicted_7_day_stock_prices.png')


if __name__ == "__main__":
    stock = input("Enter the stock symbol (e.g., 'GOOG'): ")
    
    predictor = StockPricePredictor(stock)
    predictor.download_data()
    predictor.prepare_data()
    predictor.build_and_train_model()
    predictor.save_model("stock_price_model.keras")
    predictor.calculate_rmse()
    prediction_df=predictor.predict_next_7_days()   
    predictor.plot_predictions(prediction_df)