from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import talib
import plotly.graph_objs as go


app = Flask(__name__,static_folder='static')

def get_stock_prices():
    try:
        def fetch_ticker_data(ticker_symbol):
            data = yf.Ticker(ticker_symbol).history(period="5d")
            if len(data) >= 2:  # At least 2 rows exist
                close_price = data['Close'].iloc[-1]
                prev_close_price = data['Close'].iloc[-2]
                price_change = close_price - prev_close_price
                return round(close_price, 2), round(price_change, 2)
            elif len(data) == 1:  # Only 1 row of data
                close_price = data['Close'].iloc[-1]
                return round(close_price, 2), "No Change"
            else: 
                return "N/A", "N/A"

        nifty50_price, nifty50_change = fetch_ticker_data("^NSEI")
        niftybank_price, niftybank_change = fetch_ticker_data("^NSEBANK")
        sensex_price, sensex_change = fetch_ticker_data("^BSESN")

        
        print({
            "nifty50": {"price": nifty50_price, "change": nifty50_change},
            "niftybank": {"price": niftybank_price, "change": niftybank_change},
            "sensex": {"price": sensex_price, "change": sensex_change}
        })

        return {
            "nifty50": {"price": nifty50_price, "change": nifty50_change},
            "niftybank": {"price": niftybank_price, "change": niftybank_change},
            "sensex": {"price": sensex_price, "change": sensex_change}
        }
    except Exception as e:
        print(f"Error fetching stock prices: {e}")
        return {
            "nifty50": {"price": "N/A", "change": "N/A"},
            "niftybank": {"price": "N/A", "change": "N/A"},
            "sensex": {"price": "N/A", "change": "N/A"}
        }


def get_last_month_data(ticker):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=1826)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def get_live_price(ticker):
    ticker_data = yf.Ticker(ticker)
    live_price = ticker_data.history(period="1d")['Close'].iloc[-1]
    return live_price

# Feature engineering with TA-Lib indicators
def prepare_data(data):
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['SMA200'] = talib.SMA(data['Close'], timeperiod=200)
    data['SMA100'] = talib.SMA(data['Close'], timeperiod=100)
    data['SMA50'] = talib.SMA(data['Close'], timeperiod=50)
    data['EMA'] = talib.EMA(data['Close'], timeperiod=10)
    
    data['Return'] = data['Close'].pct_change()
    data['Trend'] = np.where(data['Return'] > 0, 1, 0)  # 1 for bullish, 0 for bearish
    data['Moving_Avg'] = data['Close'].rolling(window=5).mean()
    data['Volatility'] = data['Close'].rolling(window=5).std()

    data = data.dropna()
    return data

# Train Linear Regression model
def train_model(data, horizon):
    features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'ADX', 'SMA200', 'SMA100', 'SMA50', 'EMA', 'Moving_Avg', 'Volatility']
    X = data[features]
    y_price = data['Close'].shift(-horizon).dropna()
    X = X[:-horizon]  # Align features with target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_price, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler, features

# Predict the price for a specific horizon
def predict_future_price(model, scaler, features, last_row):
    last_row_df = pd.DataFrame([last_row], columns=features)
    last_row_scaled = scaler.transform(last_row_df)
    predicted_price = model.predict(last_row_scaled)[0]
    return predicted_price


@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].strip()
    data = get_last_month_data(ticker)
    
    if data.empty:
        return render_template('index.html', message="No data found for the ticker.")
    
    try:
        live_price = get_live_price(ticker)
        stock_info = get_stock_info(ticker)
    except Exception as e:
        live_price = None
        stock_info = {}

    if live_price is not None:
        last_row = {'Close': live_price}
        for col in ['Open', 'High', 'Low', 'Adj Close', 'Volume']:
            last_row[col] = np.nan
        data = pd.concat([data, pd.DataFrame([last_row], index=[datetime.date.today()])])

    data = prepare_data(data)
    
    if data.empty:
        return render_template('index.html', message="Not enough data after processing.")
    
    horizons = {
        "Tomorrow": 1,     
        "1 Week": 5,       
        "1 Month": 22,     
        "6 Months": 125,   
        "1 Year": 252      
    }

    last_close_price = data['Close'].iloc[-1]
    predictions = {}
    trends = {}

    for period, horizon in horizons.items():
        try:
            model, scaler, features = train_model(data, horizon)
            last_row = data.iloc[-1][features].values
            predicted_price = predict_future_price(model, scaler, features, last_row)
            predictions[period] = predicted_price

            if predicted_price > last_close_price:
                trends[period] = "Bullish"
            elif predicted_price < last_close_price:
                trends[period] = "Bearish"
            else:
                trends[period] = "Neutral"
        except Exception as e:
            predictions[period] = "Unavailable"
            trends[period] = "Unavailable"
    
   
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
    
    fig.update_layout(
        title=f"{ticker} Stock Price Over Last Year",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_dark"  
    )
    
    chart_html = fig.to_html(full_html=False)

    return render_template('prediction.html', predictions=predictions, trends=trends, live_price=live_price, ticker=ticker, chart_html=chart_html, stock_info=stock_info)

# Function to fetch additional stock info (PE ratio, market cap, etc.)
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    stock_info = {
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "Market Cap": info.get('marketCap', 'N/A'),
        "Beta": info.get('beta', 'N/A'),
        "Dividend Yield": info.get('dividendYield', 'N/A'),
        "50 Day Moving Average": info.get('fiftyDayAverage', 'N/A'),
        "200 Day Moving Average": info.get('twoHundredDayAverage', 'N/A'),
        "Previous Close": info.get('previousClose', 'N/A'),
        "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A')
    }
    return stock_info


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prices')
def prices():
    return jsonify(get_stock_prices())

@app.route('/about')
def about():
    return render_template('about.html') 

if __name__ == '__main__':
    app.run()
