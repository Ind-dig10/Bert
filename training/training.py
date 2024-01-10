import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('toxic_ru.csv', delimiter=',')
df.replace('pos', '1', inplace=True)
df.replace('neg', '-1', inplace=True)
df.replace('neg_2', '-2', inplace=True)
df1 = df[df['Label']==1]
df2 = df[df['Label']==0]
df = pd.concat([df1,df2])
df.reset_index(inplace=True)
print(df)