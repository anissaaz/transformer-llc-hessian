import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

files = Path('llc-bs64-steps400') / 'llc_layer0_attention_query_key_value_weight'
df = pd.read_csv(files)
df.plot()
plt.show()
