from transformers import BertTokenizer, BertForSequenceClassification, get_cosine_schedule_with_warmup, AdamW
import torch

"""Классификатор BERT"""
class BertClassifier:
    def __init__(self, model_path, tokenizer_path, data, n_classes=13, epochs=5):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.data = data
        self.device = torch.device('cuda')
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes).cuda()
        self.model = self.model.cuda()

    def preparation(self):
        self.df_train, self.df_val, self.df_test = np.split(self.data.sample(frac=1, random_state=42),
                                                            [int(.85 * len(self.data)), int(.95 * len(self.data))])

        self.train, self.val = CustomDataset(self.df_train, self.tokenizer, phase='train'), CustomDataset(self.df_val,
                                                                                                          self.tokenizer,
                                                                                                          phase='train')
        self.train_dataloader = torch.utils.data.DataLoader(self.train, batch_size=4, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val, batch_size=4)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def fit(self):
        self.model = self.model.train()

        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0
            for train_input, train_label in tqdm(self.train_dataloader):
                train_label = train_label.cuda()
                mask = train_input['attention_mask'].cuda()
                input_id = train_input['input_ids'].squeeze(1).cuda()
                output = self.model(input_id.cuda(), mask.cuda())

                batch_loss = self.loss_fn(output[0], train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output[0].argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            total_acc_val, total_loss_val = self.eval()

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.df_train): .3f} \
            | Train Accuracy: {total_acc_train / len(self.df_train): .3f} \
            | Val Loss: {total_loss_val / len(self.df_val): .3f} \
            | Val Accuracy: {total_acc_val / len(self.df_val): .3f}')

            os.makedirs('checkpoint', exist_ok=True)
            torch.save(self.model, f'checkpoint/BertClassifier{epoch_num}.pt')

        return total_acc_train, total_loss_train

    def eval(self):
        self.model = self.model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in tqdm(self.val_dataloader):
                val_label = val_label.cuda()
                mask = val_input['attention_mask'].cuda()
                input_id = val_input['input_ids'].squeeze(1).cuda()

                output = self.model(input_id.to('cuda'), mask.to('cuda'))

                batch_loss = self.loss_fn(output[0], val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output[0].argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        return total_acc_val, total_loss_val