import datetime
import pandas as pd
timestamp = 1683239419051505854
milliseconds = int(timestamp / 1000000000)

# 將毫秒轉換為日期和時間
dt = datetime.datetime.fromtimestamp(milliseconds)
expiration_date=pd.Timestamp(2028, 2, 18)
ans=((expiration_date-dt)/pd.Timedelta(days=365))
print(type(ans))