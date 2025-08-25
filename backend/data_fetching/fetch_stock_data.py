import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import os

def get_historical_closing(symbol, count=30):
    url = f'https://www.financialnotices.com/stock-nepse.php?symbol={symbol}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    history = []
    rows = soup.find_all('tr')
    for row in rows:
        cols = [c.text.strip() for c in row.find_all('td')]
        if len(cols) >= 6:
            date_str = cols[0]
            close_price = cols[1]
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                price = float(close_price.replace(',', ''))
                history.append({'Date': date_obj, 'Close': price})
            except ValueError:
                continue
            if len(history) >= count:
                break
    return history[::-1]  # oldest to newest

# 15 commercial bank symbols
symbols = ["ADBL", "EBL", "CZBIL", "GBIME", "HBL", "KBL", "MBL", "NABIL", "NBL", "NICA", 
           "NIMB", "NMB", "PRVU", "SANIMA", "SBI"]

# Folder to store individual CSVs
os.makedirs("../fetchStockData", exist_ok=True)

# Initialize empty DataFrame for combined data
combined_df = pd.DataFrame()

# Collect and combine data
for symbol in symbols:
    data = get_historical_closing(symbol)
    if data:
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)

        # Save individual CSV (Date, Close)
        df.to_csv(f"../fetchStockData/{symbol}.csv")
        print(f"Saved: ../fetchStockData/{symbol}.csv")


        # Add to combined DataFrame
        df_symbol = df.rename(columns={'Close': symbol})
        combined_df = combined_df.join(df_symbol, how='outer') if not combined_df.empty else df_symbol

combined_df.dropna(inplace=True)

# Sort by date
combined_df.sort_index(inplace=True)

combined_df.to_csv("15_commercial_bank.csv")
