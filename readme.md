# 📈 StockAI Predictor Pro

A sophisticated machine learning application for predicting stock prices using Random Forest algorithm, advanced feature engineering, and real-time market data visualization.

## 🎯 Model Architecture & Performance

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Implementation**: scikit-learn's RandomForestRegressor
- **Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42
  - n_jobs: -1 (parallel processing)

### Feature Engineering
1. **Price Features**:
   - Open, High, Low, Close prices
   - Daily Volume
   - Price Changes

2. **Technical Indicators**:
   - Moving Averages (MA_20, MA_50)
   - Relative Strength Index (RSI)
   - Volume Moving Average
   - Price Change Percentage

3. **Lag Features**:
   - Previous 1, 2, 3, and 5-day closing prices
   - Historical price patterns

### Model Performance Metrics
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Model's explanatory power
- **Feature Importance Analysis**: Identifies most influential factors

## 🚀 Features

- **Real-time Stock Data**: Fetch live data from Alpha Vantage
- **AI-Powered Predictions**: Advanced Random Forest ML algorithm with feature engineering
- **Multiple Markets**: Support for Indian (NSE) and US stocks
- **Technical Analysis**: Moving averages, volatility, and technical indicators
- **Multi-day Forecasting**: Predict up to 30 days ahead
- **Interactive Dashboard**: Beautiful visualizations with Plotly
- **Model Performance**: Detailed accuracy metrics and feature importance analysis
- **Data Export**: Download historical data as CSV

## 🏗️ Project Structure

```
market-forecast-app/
│
├── app.py                 # Main Streamlit application
├── y-test.py               # Test your alpha vantage APi key
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── models/               # Directory for saved models (auto-created)
│   ├── RELIANCE_model_*.pkl
│   └── RELIANCE_scaler_*.pkl
│
└── data/                 # Directory for data files (optional)
    └── *.csv
```

## 🛠️ Installation & Setup

### Local Development

1. **Clone or Download the Project**
   ```bash
   mkdir market-forecast-app
   cd market-forecast-app
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`

### 🚀 Deploy on Streamlit Cloud

1. **Fork the Repository**
   ```bash
   # Fork from GitHub
   https://github.com/techiepookie/StockAI-predictor-pro
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/StockAI-predictor-pro.git
   cd StockAI-predictor-pro
   ```

3. **Set Up Environment Variables**
   - Go to Streamlit Cloud Dashboard
   - Navigate to App Settings > Secrets
   - Add your Alpha Vantage API key:
     ```toml
     ALPHA_VANTAGE_API_KEY = "your_api_key_here"
     ```

4. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your forked repository
   - Configure deployment:
     - Main file path: `app.py`
     - Branch: `main`
     - Python version: 3.8+
   - Click "Deploy!"

5. **Access Your App**
   - URL format: `https://YOUR_USERNAME-stockai-predictor-pro.streamlit.app`
   - Share with your users!

## 📊 How to Use

### 1. Select Stock
- **Indian Stocks**: Choose from popular NSE stocks like RELIANCE.NS, TCS.NS
- **US Stocks**: Select from NASDAQ/NYSE stocks like AAPL, GOOGL
- **Custom Ticker**: Enter any valid stock ticker symbol

### 2. Configure Settings
- **Time Period**: Select data range (1 month to 5 years)
- **Prediction Days**: Choose how many days to forecast (1-30)
- **Model Settings**: Enable/disable model retraining

### 3. Analyze Results
- **Stock Analysis**: View current metrics and company information
- **Predictions**: See AI-generated price forecasts
- **Charts**: Interactive price and volume visualizations
- **Model Performance**: Detailed accuracy metrics
- **Data Table**: Historical data with download option

## 🤖 Machine Learning Details

### Algorithm
- **Random Forest Regressor**: Ensemble method for robust predictions
- **Feature Engineering**: Technical indicators, lag features, and temporal features
- **Cross-validation**: Time-series aware splitting to prevent data leakage

### Features Used
- **Price Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: Moving averages (5, 10, 20 days)
- **Lag Features**: Previous day values (1, 2, 3 days)
- **Volatility Measures**: Price changes and high-low percentages
- **Temporal Features**: Day of week, month

### Model Evaluation
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Feature Importance**: Most influential factors

## 📈 Supported Stock Examples

### Indian Stocks (NSE)
```
RELIANCE.NS    - Reliance Industries
TCS.NS         - Tata Consultancy Services
INFY.NS        - Infosys
HDFCBANK.NS    - HDFC Bank
ICICIBANK.NS   - ICICI Bank
HINDUNILVR.NS  - Hindustan Unilever
ITC.NS         - ITC Limited
KOTAKBANK.NS   - Kotak Mahindra Bank
LT.NS          - Larsen & Toubro
AXISBANK.NS    - Axis Bank
```

### US Stocks
```
AAPL    - Apple Inc.
GOOGL   - Alphabet Inc.
MSFT    - Microsoft Corporation
AMZN    - Amazon.com Inc.
TSLA    - Tesla Inc.
META    - Meta Platforms Inc.
NVDA    - NVIDIA Corporation
NFLX    - Netflix Inc.
AMD     - Advanced Micro Devices
INTC    - Intel Corporation
```

## � Model Evaluation Details

### Performance Metrics
1. **Root Mean Square Error (RMSE)**
   - Measures prediction accuracy
   - Lower values indicate better performance
   - Calculated as: sqrt(mean((actual - predicted)²))

2. **Mean Absolute Error (MAE)**
   - Average absolute difference between predictions and actual values
   - More robust to outliers than RMSE
   - Calculated as: mean(|actual - predicted|)

3. **R² Score (Coefficient of Determination)**
   - Indicates the proportion of variance explained by the model
   - Range: 0 to 1 (higher is better)
   - Calculated using scikit-learn's r2_score

### Feature Importance Analysis
- Random Forest provides built-in feature importance scores
- Higher scores indicate more influential features
- Top features typically include:
  1. Recent closing prices (Close_Lag_1)
  2. Volume indicators
  3. Technical indicators (RSI, MA)

### Cross-Validation Strategy
- Time series split to prevent data leakage
- 80-20 train-test split
- No shuffle to maintain temporal order
- Validation on most recent data

## �🔧 Technical Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 512MB RAM
- **Internet**: Required for fetching real-time stock data
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)
- **Storage**: ~100MB for application and dependencies

## 📦 Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **joblib**: Model serialization
- **plotly**: Interactive visualizations
- **matplotlib**: Additional plotting capabilities

## 🔍 Troubleshooting

### Common Issues

1. **"Invalid ticker" Error**
   - Ensure ticker symbol is correct
   - For Indian stocks, add `.NS` suffix (e.g., `RELIANCE.NS`)
   - For US stocks, use standard symbols (e.g., `AAPL`)

2. **"No data found" Error**
   - Check internet connection
   - Verify stock ticker exists
   - Try different time period

3. **Model Training Failed**
   - Ensure sufficient data (minimum 50 data points)
   - Try longer time period
   - Check for data quality issues

4. **Slow Performance**
   - Reduce prediction days for faster processing
   - Use shorter time periods for training
   - Ensure stable internet connection

### Performance Tips
- Start with 1-year data for balanced accuracy and speed
- Use 7-day predictions for optimal performance
- Retrain model weekly for best accuracy

## ⚠️ Important Disclaimers

1. **Educational Purpose**: This application is for educational and research purposes only
2. **Not Financial Advice**: Predictions should not be used as sole basis for investment decisions
3. **Market Risks**: Stock markets are inherently unpredictable and risky
4. **Consult Professionals**: Always consult with qualified financial advisors
5. **Do Your Research**: Conduct thorough analysis before making investment decisions

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b new-feature`
3. **Make your changes**
4. **Commit changes**: `git commit -am 'Add new feature'`
5. **Push to branch**: `git push origin new-feature`
6. **Submit pull request**

### Areas for Improvement
- Add more ML algorithms (LSTM, ARIMA, Prophet)
- Implement portfolio analysis
- Add real-time alerts and notifications
- Include fundamental analysis data
- Add backtesting capabilities

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

If you encounter any issues or have questions:

1. **Check the troubleshooting section** above
2. **Review the documentation** thoroughly
3. **Create an issue** on GitHub with detailed information
4. **Include error messages** and steps to reproduce

## 🎯 Future Enhancements

- **LSTM Neural Networks**: Deep learning for time series
- **Sentiment Analysis**: Social media and news sentiment
- **Portfolio Optimization**: Multi-stock portfolio analysis
- **Real-time Alerts**: Price target notifications
- **Mobile App**: React Native mobile version
- **API Integration**: RESTful API for developers
- **Backtesting**: Historical strategy performance testing

---

**Built with ❤️ using Streamlit, scikit-learn, and Alpha Vantage API**

*Last updated: August 2025*
