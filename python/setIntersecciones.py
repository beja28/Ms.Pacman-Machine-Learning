import pandas as pd

df = pd.read_csv('01_gameStatesData.csv')

valores_unicos = sorted(set(df['pacmanCurrentNodeIndex'].unique()))

print(valores_unicos)
