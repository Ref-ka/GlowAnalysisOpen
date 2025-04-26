import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"train/predictor_loss.csv")

sns.lineplot(x=data["epoch"], y=data["loss"])
plt.show()
