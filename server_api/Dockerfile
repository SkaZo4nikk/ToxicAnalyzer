FROM dockerhub.timeweb.cloud/python:3.10.7

SHELL [ "/bin/bash", "-c" ]

COPY . .

COPY ./torchapp/config/kaggle_api.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

RUN export KAGGLE_USERNAME=$(jq -r .username /root/.kaggle/kaggle.json) \
    && export KAGGLE_KEY=$(jq -r .key /root/.kaggle/kaggle.json)

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p torchapp/toxic_model/trained_models
RUN wget "https://huggingface.co/Izlaster/ToxicMetr/resolve/main/server_api/torchapp/model/best_model_state.bin?download=true" -O torchapp/toxic_model/trained_models/best_model_state.bin

RUN touch torchapp/toxic_model/logs/training.log

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
