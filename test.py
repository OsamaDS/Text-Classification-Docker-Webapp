from engine import training
import pandas as pd

df = pd.read_csv('input/IMDB Dataset.csv')
coloms = df.columns
df['text'] = df[coloms[0]]
df['label'] = df[coloms[1]]

trainer = training(df)

trainer.train_model()