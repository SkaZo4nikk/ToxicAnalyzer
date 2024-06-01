from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

import torch
import json

from torchapp.toxic_model.pipeline import BertClass
from torchapp.toxic_model.train_params import tokenizer
from torchapp.toxic_model.seeds import seed_all

@csrf_exempt
@api_view(['POST'])
def classify_text(request):
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            text = data.get('text', '')
            seed_all(42)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            class_names = ['positive', 'negative']

            myModel = BertClass(len(class_names))
            myModel.load_state_dict(torch.load('torchapp/toxic_model/trained_models/best_model_state.bin', map_location=device))
            myModel = myModel.to(device)

            encoded_review = tokenizer.encode_plus(text, max_length=512, add_special_tokens=True, return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True, truncation=True, return_tensors='pt')
            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)
            output = myModel(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)

            # Формирование ответа
            response_data = {
                'text': text,
                'probabilities': prediction.tolist()
            }

            return JsonResponse(response_data)
        else:
            return JsonResponse({'error': 'Invalid request method'})
    except:
        return JsonResponse({'error': 'Please wait, model is study'})

