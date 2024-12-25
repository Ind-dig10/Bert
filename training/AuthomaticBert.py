import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Шаг 1. Загрузка и подготовка данных
# Замените это на ваш датасет
data = {
    "text": [
        "Это научный текст с формальным стилем.",
        "Привет, как дела? Это пример разговорного стиля.",
        "Литературный стиль обычно богаче и изящнее.",
        "Деловая переписка требует строгой структуры.",
        "Эй, послушай, это совсем неформально!"
    ],
    "label": [0, 1, 2, 0, 1]  # Метки стилей: 0 - формальный, 1 - разговорный, 2 - литературный
}

df = pd.DataFrame(data)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Шаг 2. Токенизация
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts, labels):
    tokens = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
    return tokens['input_ids'], tokens['attention_mask'], torch.tensor(labels)

train_inputs, train_masks, train_labels = tokenize_data(train_texts, train_labels)
val_inputs, val_masks, val_labels = tokenize_data(val_texts, val_labels)

# Шаг 3. Создание Dataset
class TextDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

train_dataset = TextDataset(train_inputs, train_masks, train_labels)
val_dataset = TextDataset(val_inputs, val_masks, val_labels)

# Шаг 4. Создание модели
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Шаг 5. Обучение
from torch.optim import AdamW

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

optimizer = AdamW(model.parameters(), lr=1e-5)

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = (
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['labels'].to(device),
        )
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = (
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['labels'].to(device),
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

for epoch in range(3):
    train_loss = train_epoch(model, train_loader)
    val_accuracy = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

# Шаг 6. Предсказание
def predict(text):
    tokens = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids, attention_mask = tokens['input_ids'].to(device), tokens['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    return torch.argmax(outputs.logits, dim=-1).item()

sample_text = "Этот текст выглядит довольно официально."
predicted_label = predict(sample_text)
print(f"Predicted Style: {predicted_label}")
