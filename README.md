# SP500-Prediction-
# 📊 **SP500**
### **Predicting S&P 500 Stocks with ML**

**SP500 is a machine learning-powered stock price prediction app specifically designed for S&P 500 companies. Built with Python and Streamlit, it leverages historical stock data to forecast future trends and empower investors with data-driven insights.**

---

## 🧬 **Project Structure**
```bash
SP500-PREDICTION- 
├── assets/         
│   ├── data/
│   │   └──  sp500_tickers.csv       
│   │ 
├── streamlit_app/
│   ├── modules/
│   │   └── helper.py
│   ├── pages/               
│   │   └── 01_📈_StockPredictor.py 
│   └── 00_ℹ️_Info.py     
├── LICENSE                 
├── README.md               
└── requirements.txt        
```

---

## 🛠️ **How It's Built**

SP500Prediction is built with the following core frameworks and tools:

- **Streamlit** - To create an intuitive web interface
- **Yahoo Finance API (YFinance)** - To fetch up-to-date financial data
- **Statsmodels** - To implement the AutoReg time-series forecasting model
- **Plotly** - To generate dynamic and interactive financial charts
- **Pandas** - To manipulate and process financial datasets

---

## 🧑‍💻 **How It Works**

1. The user selects a stock ticker from the S&P 500 list.
2. Historical stock data is retrieved using the Yahoo Finance API.
3. The AutoReg (Auto Regressive) model is trained on two years of historical data.
4. The model generates forecasts for the next 5–180 days.
5. Results are displayed with interactive charts and tables.

---

## ✨ **Key Features**

- **Real-time S&P 500 stock data** - Access accurate and up-to-date information.
- **Interactive charts** - View historical trends and future predictions visually.
- **Custom prediction ranges** - Forecast stock prices for 5 to 180 days.
- **Downloadable CSV** - Save prediction results for further analysis.
- **User-friendly interface** - Accessible for novice and experienced users alike.

---

## 🚀 **Getting Started**

### **Local Installation**

1. Navigate to the repository directory:
```bash
cd SP500
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
.venv\Scripts\Activate.ps1
```

5. Install requirements:
```bash
pip install -r requirements.txt
```

6. Navigate to the app directory:
```bash
cd streamlit_app
```

7.Make sure you download all the requirements also in the virtual environement (Any )
pip install -r requirements.txt 

7. Run the app:
```bash
streamlit run 00_ℹ️_Info.py
```

The app will be live at ```http://localhost:8501```

--
