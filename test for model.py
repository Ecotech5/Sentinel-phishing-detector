# src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/final_merged_dataset.csv")
df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')

# stratified split
train, temp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
valid, test  = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

train.to_csv("data/train.csv", index=False)
valid.to_csv("data/valid.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("Saved train/valid/test")
