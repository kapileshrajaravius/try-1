import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import time
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Configuration & Session Setup ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
DB_FILE = "portfolio.json"

if 'session' not in st.session_state:
    st.session_state.session = requests.Session()
    st.session_state.session.headers.update({"User-Agent": USER_AGENT})

# --- Data Persistence ---
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

# --- Pricing Engine ---
def get_google_finance_price(ticker):
    """Fallback scraper for Google Finance."""
    try:
        url = f"https://www.google.com/search?q=google+finance+{ticker}"
        response = st.session_state.session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Google Finance often uses a specific class for prices in search results
        price_element = soup.find("div", {"class": "YMlS1e"}) or soup.find("span", {"jsname": "vWLAgc"})
        if price_element:
            return float(price_element.text.replace(",", "").replace("$", "").replace("₹", "").strip())
    except Exception:
        return None
    return None

def get_live_price(ticker):
    """Primary data fetcher with fallback logic."""
    time.sleep(1.5)  # Anti-blocking delay
    try:
        data = yf.Ticker(ticker)
        # Attempt to get fast info
        price = data.fast_info.get('last_price')
        
        # If yfinance fails or returns None, use Google fallback
        if price is None or np.isnan(price):
            price = get_google_finance_price(ticker)
            
        return price
    except Exception:
        return get_google_finance_price(ticker)

def format_currency(value, ticker):
    symbol = "₹" if ticker.upper().endswith((".NS", ".BO")) else "$"
    return f"{symbol}{value:,.2f}"

# --- AI Logic ---
def analyze_stock(ticker, current_price):
    """Simple Linear Regression to predict short-term trend."""
    try:
        time.sleep(1.5)
        hist = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if len(hist) < 5:
            return "HOLD", 0.0, "Insufficient data for AI analysis"
        
        y = hist['Close'].values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        prediction = model.predict([[len(y) + 1]])[0]
        growth = (prediction - current_price) / current_price
        
        if growth > 0.01:
            return "BUY MORE", prediction - current_price, "Price is on a hot streak"
        elif growth < -0.02:
            return "SELL", prediction - current_price, "Technical indicators show downward momentum"
        else:
            return "HOLD", prediction - current_price, "Market consolidation expected"
    except:
        return "HOLD", 0.0, "Service Temporarily Unavailable"

# --- Streamlit UI ---
st.set_page_config(page_title="Institutional Portfolio Manager", layout="wide")

page = st.sidebar.selectbox("Navigation", ["Registration", "My Portfolio", "AI Analysis Report", "Global Opportunities"])

if page == "Registration":
    st.header("Stock Registration")
    with st.form("reg_form"):
        col1, col2, col3 = st.columns(3)
        t = col1.text_input("Ticker (e.g., AAPL or RELIANCE.NS)")
        u = col2.number_input("Units Owned", min_value=0.0, step=1.0)
        p = col3.number_input("Purchase Price", min_value=0.0)
        submit = st.form_submit_button("Add to Portfolio")
        
        if submit and t:
            current_portfolio = load_data()
            current_portfolio.append({"ticker": t.upper(), "units": u, "buy_price": p})
            save_data(current_portfolio)
            st.success(f"Added {t.upper()} to records.")

elif page == "My Portfolio":
    st.header("Portfolio Overview")
    portfolio = load_data()
    if portfolio:
        display_list = []
        total_portfolio_value = 0
        
        for item in portfolio:
            price = get_live_price(item['ticker'])
            if price:
                val = price * item['units']
                total_portfolio_value += val
                display_list.append({
                    "Ticker": item['ticker'],
                    "Units": item['units'],
                    "Live Price": format_currency(price, item['ticker']),
                    "Total Value": format_currency(val, item['ticker'])
                })
            else:
                display_list.append({
                    "Ticker": item['ticker'],
                    "Units": item['units'],
                    "Live Price": "Service Unavailable",
                    "Total Value": "N/A"
                })

        st.metric("Total Portfolio Value", f"Value calculated across multiple currencies")
        df = pd.DataFrame(display_list)
        df.index = df.index + 1
        st.table(df)
    else:
        st.info("No stocks registered yet.")

elif page == "AI Analysis Report":
    st.header("AI Analysis Report")
    portfolio = load_data()
    if portfolio:
        analysis_list = []
        unique_tickers = list(set([item['ticker'] for item in portfolio]))
        
        for ticker in unique_tickers:
            price = get_live_price(ticker)
            if price:
                action, gain, reason = analyze_stock(ticker, price)
                analysis_list.append({
                    "Stock": ticker,
                    "Action": action,
                    "Predicted Gain ($)": round(gain, 2),
                    "Simplified Reason": reason
                })
        
        df = pd.DataFrame(analysis_list)
        df.index = df.index + 1
        st.table(df)
    else:
        st.info("Add stocks to see AI insights.")

elif page == "Global Opportunities":
    st.header("Global Market Scanner")
    watch_list = ["TSLA", "NVDA", "AAPL", "MSFT"]
    picks = []
    exits = []
    
    for ticker in watch_list:
        price = get_live_price(ticker)
        if price:
            action, gain, _ = analyze_stock(ticker, price)
            entry = {"Ticker": ticker, "Price": format_currency(price, ticker), "Trend": f"{gain:+.2f}"}
            if action == "BUY MORE":
                picks.append(entry)
            elif action == "SELL":
                exits.append(entry)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strong Picks to Buy")
        if picks:
            df_p = pd.DataFrame(picks)
            df_p.index = df_p.index + 1
            st.table(df_p)
        else:
            st.write("No strong buy signals.")
            
    with col2:
        st.subheader("Exit Signals (Time to Sell)")
        if exits:
            df_e = pd.DataFrame(exits)
            df_e.index = df_e.index + 1
            st.table(df_e)
        else:
            st.write("No immediate sell signals.")
