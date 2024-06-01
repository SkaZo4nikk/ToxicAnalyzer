import torch
import logging
from collections import defaultdict

from torchapp.toxic_model.train import train_model, eval_model

def trainer(path, epochs, model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, df_train, df_val) -> None: 
    history = defaultdict(list)
    best_accuracy = 0

    log_file = 'torchapp/toxic_model/logs/training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1}/{epochs}')
        
        train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        logger.info(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
        
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
        logger.info(f'Val loss: {val_loss}, Val accuracy: {val_acc}')
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
    
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), f'{path}/best_model_state.bin')
            best_accuracy = val_acc
            logger.info('New best model saved with accuracy: {:.4f}'.format(best_accuracy))