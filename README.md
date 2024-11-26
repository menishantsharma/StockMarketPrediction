# Stock Market Prediction

## File Structure for folder nishant

smp.ipynb - File contains model training

predictor.py - File contains code to predict new output

lstm_stock_model.h5 - Trained Model saved

prediction.png - Graph of predicted price where dot represent the predicted next day price.

loss.png - Represent the change of training and validation loss with number of epochs.

actual_vs_predicted.png - Represent what is actual price and what my model learned.

RMSE: 0.041555813978944624

## File Structure for folder Harsh Kumar
Stock_Market_Prediction.py : Contains the python code for training the model on a particular stock.
                            The model trained for GOOG(Google) stock and download the last 20 years data from yfinance python library and used the sliding window of 7 days.

stock_price_model.keras : Trained model saved for reference.

rmse_plot.png : Contains the RMSE/Accuracy plot for 5 stock features (Open, High, Low, Close, Adj Close).

predicted_7_day_stock_prices.png : Predicting the 5 stock features for upcoming 7 trading days.
