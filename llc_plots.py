import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv("llc-bs64-steps400/llc_layer0_attention_query_key_value_weight.csv")
df.plot()
plt.show()
