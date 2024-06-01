import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from torchapp.toxic_model.pipeline import BertClass
from torchapp.toxic_model.scoring import score_model
from torchapp.toxic_model.seeds import seed_all
from torchapp.toxic_model.dataloader import create_data_loader
from torchapp.toxic_model.trainer import trainer
from torchapp.toxic_model.train_params import tokenizer, epochs, write_params

def run_training() -> None:
    seed_all(42)
    RANDOM_SEED = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['positive', 'negative']

    df = pd.read_csv('torchapp/toxic_model/dataset/labeled.csv')
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = RANDOM_SEED)

    train_data_loader = create_data_loader(df_train, tokenizer, batch_size = 8)
    val_data_loader = create_data_loader(df_val, tokenizer, include_raw_text=False)
    test_data_loader = create_data_loader(df_test, tokenizer, include_raw_text=False)

    model = BertClass(len(class_names))
    model = model.to(device)

    optimizer, scheduler, loss_fn = write_params(model, train_data_loader, device)
    trainer('torchapp/toxic_model/trained_models', epochs, model, train_data_loader, val_data_loader, 
            loss_fn, optimizer, device, scheduler, df_train, df_val)    
    
    score_model(model, test_data_loader, class_names, device)