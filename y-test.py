from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Tumhara API key yahan daalo
api_key = "RSMSPEMM4AQHT6QL"

# Alpha Vantage object create karo
ts = TimeSeries(key=api_key, output_format='pandas')

try:
    # Example: 'RELIANCE.BSE' => BSE (Indian) format
    data, meta_data = ts.get_daily(symbol='RELIANCE.BSE', outputsize='compact')

    # Latest 5 days ka data print karo
    print("\n📊 Last 5 Days Data:\n")
    print(data.head())

except Exception as e:
    print("\n❌ Error fetching data:")
    print(e)
