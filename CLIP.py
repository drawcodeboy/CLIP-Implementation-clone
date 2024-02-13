import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import os
import cv2
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import albumentations as A

class CLIPDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):

        self.image_filenames = image_filenames
        self.captions = list(captions)
        # Dataset 선언을 통해 Tokenizer Object를 input으로 받는다.
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=200
        )
        self.transforms = transforms
        
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        image = cv2.imread(f'F:/Doby/CLIP/Flickr8k/Images/{self.image_filenames[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item
        
    def __len__(self):
        return len(self.captions)

def get_transforms(mode='train'):
    # Train이나 Test나 전처리 같음
    # 따로 Augmentation은 하지 않음
    if mode == 'train':
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )



class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.flatten = nn.Flatten() # Image output Vector size = 2,048
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        return x

from transformers import DistilBertModel, DistilBertConfig

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in self.model.parameters():
            param.requires_grad = False

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, 256)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(256)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, temperature, image_embedding, text_embedding):
        super().__init__()
        
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Get Image, Text features through ResNet50, DistilBERT
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Same DIMENSION EMBEDDING
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax(
            (images_similarity + texts_similarity) / 2.0 * self.temperature, dim=-1
        )

        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()