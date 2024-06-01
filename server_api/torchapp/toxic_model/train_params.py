import torch
from transformers import BertTokenizer, get_linear_schedule_with_warmup

pre_trained_model_ckpt = 'cointegrated/rubert-tiny2'
tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)

epochs = 5

def write_params(model, train_data_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-5)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 16, 
        num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    return optimizer, scheduler, loss_fn