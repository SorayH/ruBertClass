# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import pymorphy2
from db import res_db

device = torch.device("cpu")
MAX_LEN = 256
VALID_BATCH_SIZE = 16
path = res_db('path')
tokenizer = BertTokenizer.from_pretrained(path[0])
my_model = path[1]

# BERT
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(tokenizer)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title = dataframe['text']
        self.targets = self.data.target_list
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class bert_func0:

    def __init__(self):

        self.model = torch.load(my_model, map_location="cpu")

    def predict(self, text):

        test_targets = []
        test_outputs = []

        def test_model(test_loader, model):
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader, 0):
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.float)
                    outputs = model(ids, mask, token_type_ids)

                    test_targets.extend(targets.cpu().detach().numpy().tolist())
                    test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        test_params = {'batch_size': VALID_BATCH_SIZE,
                       'shuffle': False,
                       'num_workers': 0
                       }
        categ = ["Животноводство", "агрономия", "Растениеводство", "Помощь животновода", "сельхозтехника",
                 "агротехника"]

        df_raw = pd.DataFrame([[0, 1, 1, 1, 1, 1, 1, text],],
                               columns=['id', 'Животноводство', 'агрономия', 'Растениеводство',
                                        'Помощь животновода', 'сельхозтехника', 'агротехника', 'text'])
        df_raw['target_list'] = df_raw[
            ['Животноводство', 'агрономия', 'Растениеводство', 'Помощь животновода', 'сельхозтехника',
             'агротехника']].values.tolist()
        df_raw = df_raw[df_raw['target_list'].map(lambda x: 1 in x)]
        df_raw['WORD_COUNT'] = df_raw['text'].apply(lambda x: len(str(x).split()))

        df2 = df_raw[['text', 'target_list']].copy()
        df2 = df2.sample(frac=1, random_state=200)
        df2 = df2.reset_index(drop=True)
        df2 = CustomDataset(df2, tokenizer, MAX_LEN)
        df2 = DataLoader(df2, **test_params)

        test_model(df2, self.model)

        res = []

        for i in test_outputs:
            for pred in i:
                if pred > 0.5:
                    pred = 1
                else:
                    pred = 0
                res.append(pred)

        res = dict(zip(categ, res))

        return res

class bert_func1:

    def __init__(self):

        x = 'bert_func1'

    def predict(self, text):

        text = 'bert_func1'

        return text

# ru_GPT
class ru_GPT:

    def __init__(self):

        x = 'ru_GPT'

    def predict(self, text):

        text = 'ru_GPT'

        return text
