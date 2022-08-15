import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import datetime

df = pd.read_csv("Bitcoin_Geçmiş_Veriler.csv")
df = df[["Tarih", "Şimdi"]]
df = df.rename({"Tarih":"ds", "Şimdi":"y"}, axis= "columns")

df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"].apply(lambda x: float(x.split()[0].replace(",","")))

model = Prophet(yearly_seasonality = True)
model.fit(df)

future = model.make_future_dataframe(periods=14)
forecast = model.predict(future)
model.plot(forecast)

plt.show()
