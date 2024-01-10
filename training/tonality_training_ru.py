
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
file_path = 'dataset.txt'
data = {'Label': [], 'Text': []}

# Откройте файл и считайте строки
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Используйте регулярное выражение для извлечения метки и текста
        match = re.match(r'(__label__\w+)\s(.+)', line)
        if match:
            label, text = match.groups()
            if (label == "__label__INSULT"):
                label = '-1'
            elif (label == "__label__NORMAL"):
                label = '1'
            elif (label == "__label__OBSCENITY"):
                label = '-2'
            data['Label'].append(label)
            data['Text'].append(text)

df = pd.DataFrame(data)

# Выведите полученный датафрейм
excel_file_path = 'toxic_ru.csv'

# Сохраните датафрейм в Excel
df.to_csv(excel_file_path, index=False)