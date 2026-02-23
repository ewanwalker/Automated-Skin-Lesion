import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

percentageSplit = 0.2
df = pd.read_csv("data/ISIC-images/metadata.csv")

df["label"] = df["diagnosis_1"].apply(
    lambda x: "malignant" if x == "Malignant" else "benign"
)

train_df, val_df = train_test_split(df, test_size=percentageSplit, stratify=df["label"], random_state=42)

print("Segmenting data into training and validation sets...")
for split, split_df in [("train", train_df), ("val", val_df)]:
    for _, row in split_df.iterrows():
        src = f"data/ISIC-images/{row['isic_id']}.jpg"
        dst = f"dataset/{split}/{row['label']}/{row['isic_id']}.jpg"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

print("Done!")