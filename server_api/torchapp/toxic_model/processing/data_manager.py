import kaggle
    
def load_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('blackmoon/russian-language-toxic-comments', path='./torchapp/toxic_model/dataset', unzip=True)