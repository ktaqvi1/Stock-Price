# Stock Prediction using Linear Regression

This project uses machine learning to predict stock closing prices based on historical stock data. We apply a **Linear Regression** model to predict the future closing price based on the previous day's opening price, high, low, and volume data.

## Project Overview

The project uses historical stock price data (open, high, low, close, volume) to build a model that predicts the closing price. The data is taken from a CSV file for a specific company, which is input by the user.

The following steps are taken in the project:

1. **Data Preprocessing**: The dataset is cleaned by removing irrelevant columns and handling missing data.
2. **Feature Engineering**: The model uses features like the opening price, high, low, and volume to predict the closing price.
3. **Model Training**: A Linear Regression model is trained on the data.
4. **Prediction & Visualization**: Predictions are made on the test data, and the results are visualized.
5. **Model Evaluation**: The model is evaluated using R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Requirements

Before running the project, ensure you have the following dependencies installed:

* Python 3.x
* Pandas
* Numpy
* Scikit-learn
* Matplotlib

You can install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## File Structure

```
/stockPredictionProject/
│
├── README.md          # Project description
├── stockPrediction.py # Main Python script
└── <company_name>.csv # Stock data CSV file (input by the user)
```

## How to Run

1. Clone the repository or download the files.
2. Prepare a CSV file containing historical stock data (with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`).
3. Place the CSV file in the project directory and name it with the company name (e.g., `AAPL.csv` for Apple).
4. Run the Python script `stockPrediction.py`:

```bash
python stockPrediction.py
```

5. Enter the company name (the name of the CSV file without the extension) when prompted.
6. The script will load the dataset, preprocess it, train the model, make predictions, and display the results (including visualizations and evaluation metrics).

## Model Evaluation

After running the model, the following evaluation metrics will be displayed:

* **R-squared (R²)**: Measures how well the model fits the data. A value closer to 1 indicates a better fit.
* **Mean Absolute Error (MAE)**: The average of the absolute differences between predicted and actual values.
* **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values.
* **Root Mean Squared Error (RMSE)**: The square root of MSE, representing the model's error in the original scale of the data.

## Visualizations

The script generates the following plots:

1. **Actual Closing Price**: A time-series plot of the actual stock closing prices.
2. **Predicted Closing Price**: A time-series plot of the predicted stock closing prices.
3. **Actual vs Predicted Closing Prices**: A combined plot showing both the actual and predicted closing prices for comparison.
<img width="1302" height="493" alt="Screenshot 2025-12-12 142106" src="https://github.com/user-attachments/assets/247d8ad0-7f56-43be-9ffd-1633810472b3" />
<img width="837" height="407" alt="Screenshot 2025-12-12 142216" src="https://github.com/user-attachments/assets/d5ff2131-53fd-461c-9f4e-cc3fb8a3a6d5" />
<img width="935" height="410" alt="Screenshot 2025-12-12 142301" src="https://github.com/user-attachments/assets/e75392e0-d124-4820-83f9-db49bd63ef97" />




## Example Output

### Evaluation Metrics:

```
R-squared: 0.86
Mean Absolute Error: 3.25
Mean Squared Error: 12.80
Root Mean Squared Error: 3.58
```

### Plots:

* **Actual vs Predicted Closing Prices**: A plot comparing the actual vs predicted stock closing prices.

Conclusion

This project demonstrates the use of **Linear Regression** for stock price prediction, showing how to preprocess the data, build a regression model, and evaluate its performance. The evaluation metrics and visualizations help understand how well the model is predicting future stock prices.

