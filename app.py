import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Algo Trading Platform",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("Algorithmic Trading Platform")

# Sidebar for user inputs
st.sidebar.header("Trading Parameters")

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical market data from Yahoo Finance
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the given ticker and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Sidebar inputs for data fetching
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Fetch and display data
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        data = fetch_data(ticker, start_date, end_date)
        if data is not None:
            st.session_state['data'] = data
            st.success(f"Successfully fetched data for {ticker}")
            
if 'data' in st.session_state:
    st.subheader(f"Historical Data for {ticker}")
    st.dataframe(st.session_state['data'].tail())

def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

if 'data' in st.session_state:
    plot_tab, stats_tab = st.tabs(["Chart", "Statistics"])
    
    with plot_tab:
        plot_type = st.radio("Chart Type", ["Candlestick", "Line"])
        if plot_type == "Candlestick":
            plot_candlestick(st.session_state['data'])
        else:
            st.line_chart(st.session_state['data']['Close'])
    
    with stats_tab:
        st.subheader("Data Statistics")
        st.write(st.session_state['data'].describe())

def calculate_indicators(data):
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

if 'data' in st.session_state:
    if st.button("Calculate Indicators"):
        with st.spinner("Calculating indicators..."):
            st.session_state['data'] = calculate_indicators(st.session_state['data'])
            st.success("Indicators calculated!")
            
    if 'SMA_20' in st.session_state.get('data', pd.DataFrame()):
        st.subheader("Technical Indicators")
        
        # Plot with indicators
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['data'].index, 
                                y=st.session_state['data']['Close'], 
                                name='Price'))
        fig.add_trace(go.Scatter(x=st.session_state['data'].index, 
                                y=st.session_state['data']['SMA_20'], 
                                name='20-day SMA'))
        fig.add_trace(go.Scatter(x=st.session_state['data'].index, 
                                y=st.session_state['data']['SMA_50'], 
                                name='50-day SMA'))
        
        fig.update_layout(height=600, title="Price with Moving Averages")
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI plot
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=st.session_state['data'].index, 
                                   y=st.session_state['data']['RSI'], 
                                   name='RSI'))
        rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
        rsi_fig.update_layout(height=400, title="RSI (14-day)")
        st.plotly_chart(rsi_fig, use_container_width=True)

def backtest_strategy(data, sma_short=20, sma_long=50):
    data = data.copy()
    
    # Calculate SMAs if not already present
    if 'SMA_20' not in data.columns:
        data['SMA_short'] = data['Close'].rolling(window=sma_short).mean()
        data['SMA_long'] = data['Close'].rolling(window=sma_long).mean()
    else:
        data['SMA_short'] = data['SMA_20']
        data['SMA_long'] = data['SMA_50']
    
    # Generate signals
    data['Signal'] = 0
    data['Signal'][sma_short:] = np.where(
        data['SMA_short'][sma_short:] > data['SMA_long'][sma_short:], 1, 0)
    data['Position'] = data['Signal'].diff()
    
    # Calculate returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']
    data['Cumulative_Market'] = (1 + data['Daily_Return']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    
    return data

if 'data' in st.session_state:
    st.subheader("Strategy Backtesting")
    
    col1, col2 = st.columns(2)
    with col1:
        sma_short = st.number_input("Short SMA Period", min_value=5, max_value=50, value=20)
    with col2:
        sma_long = st.number_input("Long SMA Period", min_value=20, max_value=200, value=50)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            backtest_data = backtest_strategy(st.session_state['data'].copy(), sma_short, sma_long)
            st.session_state['backtest_data'] = backtest_data
            st.success("Backtest completed!")
    
    if 'backtest_data' in st.session_state:
        # Plot strategy vs buy-and-hold
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state['backtest_data'].index,
            y=st.session_state['backtest_data']['Cumulative_Market'],
            name='Buy & Hold'
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state['backtest_data'].index,
            y=st.session_state['backtest_data']['Cumulative_Strategy'],
            name='Strategy'
        ))
        fig.update_layout(
            title="Strategy vs Buy & Hold",
            yaxis_title="Cumulative Returns",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        market_return = st.session_state['backtest_data']['Cumulative_Market'].iloc[-1] - 1
        strategy_return = st.session_state['backtest_data']['Cumulative_Strategy'].iloc[-1] - 1
        
        st.metric("Buy & Hold Return", f"{market_return:.2%}")
        st.metric("Strategy Return", f"{strategy_return:.2%}")
        st.metric("Outperformance", f"{(strategy_return - market_return):.2%}")

if 'backtest_data' in st.session_state:
    st.subheader("Risk Analysis")
    
    # Calculate drawdown
    cumulative_max = st.session_state['backtest_data']['Cumulative_Strategy'].cummax()
    drawdown = (st.session_state['backtest_data']['Cumulative_Strategy'] - cumulative_max) / cumulative_max
    
    # Plot drawdown
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state['backtest_data'].index,
        y=drawdown,
        fill='tozeroy',
        name='Drawdown'
    ))
    fig.update_layout(
        title="Strategy Drawdown",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics
    max_drawdown = drawdown.min()
    volatility = st.session_state['backtest_data']['Strategy_Return'].std() * np.sqrt(252)
    sharpe_ratio = st.session_state['backtest_data']['Strategy_Return'].mean() / st.session_state['backtest_data']['Strategy_Return'].std() * np.sqrt(252)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Drawdown", f"{max_drawdown:.2%}")
    col2.metric("Annualized Volatility", f"{volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

if 'data' in st.session_state:
    st.subheader("Paper Trading Simulation")
    
    initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000)
    commission = st.number_input("Commission per Trade", min_value=0.0, value=0.0, step=0.01)
    
    if st.button("Run Simulation"):
        if 'backtest_data' not in st.session_state:
            st.warning("Please run a backtest first")
        else:
            data = st.session_state['backtest_data'].copy()
            
            # Initialize account
            account = pd.DataFrame(index=data.index)
            account['Cash'] = initial_capital
            account['Shares'] = 0
            account['Total'] = initial_capital
            
            for i in range(1, len(data)):
                account.at[data.index[i], 'Cash'] = account.at[data.index[i-1], 'Cash']
                account.at[data.index[i], 'Shares'] = account.at[data.index[i-1], 'Shares']
                
                # Buy signal
                if data.at[data.index[i], 'Position'] == 1:
                    shares_to_buy = account.at[data.index[i], 'Cash'] // data.at[data.index[i], 'Open']
                    if shares_to_buy > 0:
                        account.at[data.index[i], 'Cash'] -= shares_to_buy * data.at[data.index[i], 'Open'] + commission
                        account.at[data.index[i], 'Shares'] += shares_to_buy
                
                # Sell signal
                elif data.at[data.index[i], 'Position'] == -1:
                    if account.at[data.index[i], 'Shares'] > 0:
                        account.at[data.index[i], 'Cash'] += account.at[data.index[i], 'Shares'] * data.at[data.index[i], 'Open'] - commission
                        account.at[data.index[i], 'Shares'] = 0
                
                # Update total value
                account.at[data.index[i], 'Total'] = (
                    account.at[data.index[i], 'Cash'] + 
                    account.at[data.index[i], 'Shares'] * data.at[data.index[i], 'Close']
                )
            
            st.session_state['account'] = account
            
            # Plot account value
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=account.index,
                y=account['Total'],
                name='Account Value'
            ))
            fig.update_layout(
                title="Paper Trading Account Value",
                yaxis_title="Value ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            final_value = account['Total'].iloc[-1]
            pnl = final_value - initial_capital
            roi = pnl / initial_capital
            
            col1, col2 = st.columns(2)
            col1.metric("Final Account Value", f"${final_value:,.2f}")
            col2.metric("Profit/Loss", f"${pnl:,.2f} ({roi:.2%})")
