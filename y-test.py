from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd

def main():
    # Prefer env var for security; fallback to provided key for quick test
    api_key = (os.getenv("ALPHA_VANTAGE_API_KEY") or "U6C4TOUUYCXNM53B").strip()

    # Create Alpha Vantage TimeSeries client
    ts = TimeSeries(key=api_key, output_format='pandas')

    symbol = 'RELIANCE.BSE'  # Example Indian BSE symbol; for US, use 'AAPL', 'MSFT', etc.

    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
        print("\n📊 Last 5 Rows:\n")
        print(data.head())
        print("\nℹ️ Meta Data:\n")
        print(meta_data)
    except Exception as e:
        print("\n❌ Error fetching data:")
        print(e)

if __name__ == "__main__":
    main()
