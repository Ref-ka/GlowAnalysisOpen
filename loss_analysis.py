import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Скрипт для визуализации значения функции потерь во время обучения моделей

data = pd.read_csv(r"train/detector_loss.csv")  # Также может быть predictor_loss.csv

sns.lineplot(x=data["epoch"], y=data["loss"])
plt.show()
