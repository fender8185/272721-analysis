import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import matplotlib
from pathlib import Path
import matplotlib.dates as mdates
matplotlib.use('TkAgg')

# 讀取股票和債券價格資料
def get_data(stock_folder_path, bond_folder_path):
    stock_data = pd.DataFrame()
    bond_data = pd.DataFrame()

    # 讀取股票資料
    for file in stock_folder_path.glob("*.csv"):
        df = pd.read_csv(file)
        stock_data = pd.concat([stock_data, df], ignore_index=True)

    # 讀取債券資料
    for file in bond_folder_path.glob("*.csv"):
        df = pd.read_csv(file)
        bond_data = pd.concat([bond_data, df], ignore_index=True)

    return stock_data, bond_data


def calculate_price(bid_price, ask_price):
    # 將 0 值替換為 NaN，讓我們可以用插值來填充
    bid_price = bid_price.replace(0, np.nan)
    ask_price = ask_price.replace(0, np.nan)

    # 使用前後數值的平均來插值
    bid_price = bid_price.fillna((bid_price.ffill()+bid_price.bfill())/2)
    ask_price = ask_price.fillna((ask_price.ffill()+ask_price.bfill())/2)

    # 計算價格並返回
    price = (bid_price + ask_price) / 2
    return price

def calculate_implied_volatility(merged_data, risk_free_rate, strike_price):
    
    stock_price = calculate_price(merged_data["BidPrice0_x"], merged_data["AskPrice0_x"])
    bond_price = calculate_price(merged_data["BidPrice0_y"], merged_data["AskPrice0_y"])

    expiration_date = pd.Timestamp(2028, 2, 18)
    implied_volatility = []

    for i in range(len(bond_price)):
        s = stock_price[i]
        t = (expiration_date - pd.Timestamp(merged_data['Timestamp'][i])) / pd.Timedelta(days=365)
        r = risk_free_rate
        implied_volatility.append(calculate_iv(s, strike_price, t, r, bond_price[i]))
        
    merged_data['stock_data'] = stock_price
    merged_data['bond_data'] = bond_price
    merged_data['Implied Volatility'] = implied_volatility

    return merged_data


def calculate_iv(s, K, t, r, price):
    if np.isclose(price, 0.0) or np.isclose(s, price):
        return float('inf')

    implied_volatility = math.sqrt(2 * abs((math.log(s / K) + r * t) / t))
    return implied_volatility


# 使用 Black-Scholes 公式來計算選擇權價格
def black_scholes_call(s, K, t, r, sigma):
    d1 = (np.log(s / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

# 計算隱含波動率和選擇權理論價格
def calculate_option_price_and_implied_volatility(merged_data, risk_free_rate, strike_price):
    # 這裡插入你原來的程式碼，將所有的選項價格和隱含波動率都計算出來
    merged_data = calculate_implied_volatility(merged_data, risk_free_rate, strike_price)
    expiration_date = pd.Timestamp(2028, 2, 18)
    

    # 新增選擇權理論價格的計算
    option_prices = []
    for i in range(len(merged_data)):
        s = merged_data['stock_data'].iloc[i]
        t = (expiration_date - pd.Timestamp(merged_data['Timestamp'][i])) / pd.Timedelta(days=365)
        r = risk_free_rate
        sigma = merged_data['Implied Volatility'].iloc[i]
        option_prices.append(black_scholes_call(s, strike_price, t, r, sigma))

    merged_data['Option Price'] = option_prices
    return merged_data

#計算選擇權和27271實際價格之間的差值
def calculate_price_difference(merged_data):
    option_prices = merged_data['Option Price']
    bond_prices = merged_data['bond_data']

    price_difference = option_prices - bond_prices
    average_difference = price_difference.mean()

    return average_difference

# 路徑設置
stock_folder_path = Path("20230501_20230531_2727")
bond_folder_path = Path("20230501_20230531_27271")

# 讀取股票和債券價格資料
stock_data, bond_data = get_data(stock_folder_path, bond_folder_path)
# 合併股票資料和債券資料
merged_data = pd.merge(stock_data, bond_data, on=["Timestamp"], how="outer")

merged_data.sort_values(by="Timestamp", inplace=True)

# 設定其他參數
risk_free_rate = 0.035
strike_price = 295

# 計算隱含波動率和選擇權理論價格，並將結果添加到 `merged_data` 中
merged_data = calculate_option_price_and_implied_volatility(merged_data, risk_free_rate, strike_price)


print(calculate_price_difference(merged_data))

# 繪製隱含波動率圖表

#竟然4 ns o.o+
merged_data['Timestamp'] = pd.to_datetime(merged_data['Timestamp'], unit='ns')

# 繪製圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 子圖1: 隱含波動率
ax1.set_ylabel('Implied Volatility')
ax1.plot(np.arange(len(merged_data)), merged_data['Implied Volatility'], color='tab:blue', drawstyle='steps-post')

# 子圖2: 選擇權理論價格和債券價格
ax2.set_xlabel('Date')
ax2.set_ylabel('Option Price', color='tab:red')
ax2.plot(np.arange(len(merged_data)), merged_data['Option Price'], color='tab:red', drawstyle='steps-post')

ax3 = ax2.twinx()
ax3.set_ylabel('Real Bond Price', color='tab:green')
ax3.plot(np.arange(len(merged_data)), merged_data['bond_data'], color='tab:green', drawstyle='steps-post')

# 調整刻度和間距
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax3.tick_params(axis='y', labelcolor='tab:green')

# 設定 x 軸刻度和標籤
x_ticks = np.arange(0, len(merged_data), 12000)
x_labels = [dt.strftime('%Y-%m-%d') for dt in merged_data['Timestamp'].iloc[x_ticks]]
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels, rotation=45, ha='right')

plt.tight_layout()
plt.show()

