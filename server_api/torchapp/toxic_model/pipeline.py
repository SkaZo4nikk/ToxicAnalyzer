import torch
from transformers import BertModel

from torchapp.toxic_model.train_params import pre_trained_model_ckpt

class BertClass(torch.nn.Module):
    def __init__(self, n_classes):
        super(BertClass, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_ckpt,return_dict=False)
        self.drop = torch.nn.Dropout(p = 0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask= attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)