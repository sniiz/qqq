import pandas as pd

# i feel like the model is being messed up by enormous scores for checkmates
data = pd.read_csv("bigdata.csv")
data = data[data["score"] <= 90000]
data.to_csv("bigdata.csv", index=False)
