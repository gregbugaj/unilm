import os


# from funsd_dataset.funsd_dataset import FunsdLikeDataset 
import numpy as np
import torch
from torch.nn import DataParallel
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import DataLoader
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2Processor, AdamW
from tqdm import tqdm

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

from datasets import load_dataset, load_metric
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

import warnings 
warnings.filterwarnings('ignore')


# https://towardsdatascience.com/fine-tuning-transformer-model-for-invoice-recognition-1e55869336d4
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb#scrollTo=VMYoOQuyp4NT
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb#scrollTo=NxahVHZ0NKq7

use_cuda = torch.cuda.is_available()
device= torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
device_ids = [0]

# cache_dir, data_dir
dataset = load_dataset("nielsr/funsd")
# dataset = load_dataset("funsd_dataset/funsd_dataset.py")
 
# print(dataset['train'].features)
print(dataset['train'].features['bboxes'])

labels = dataset['train'].features['ner_tags'].feature.names

print('NER-Labels -> ')
print(labels)

# os.exit()

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

print("ID2Label : ")
print(id2label)
print(label2id)

##Next, let's use `LayoutLMv2Processor` to prepare the data for the model.

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
})

def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  words = examples['words']
  boxes = examples['bboxes']
  word_labels = examples['ner_tags']
  
  encoded_inputs = processor(images, words, boxes=boxes, word_labels=word_labels, padding="max_length", truncation=True)

  return encoded_inputs


# We should us this to train on all words
# only_label_first_subword

train_dataset = dataset['train'].map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names,
                                      features=features)
test_dataset = dataset['test'].map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names,
                                      features=features)


print("Data size: ")
print(f"Data train: {len(train_dataset)}")
print(f"Data test: {len(test_dataset)}")

##Finally, let's set the format to PyTorch, and place everything on the GPU:

train_dataset.set_format(type="torch", device=device)
test_dataset.set_format(type="torch", device=device)

print(f'keys : {train_dataset.features.keys()}')

decoded = processor.tokenizer.decode(train_dataset['input_ids'][0])
print(decoded)

print('Train labels **************')
print(train_dataset['labels'][0])


##Next, we create corresponding dataloaders.

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)

##Let's verify a batch:

batch = next(iter(train_dataloader))

print("Batch verification : ")
for k,v in batch.items():
  print(k, v.shape)

# os.exit()
os.makedirs(f"./checkpoints", exist_ok = True)

##Train the model
##Here we train the model in native PyTorch. We use the AdamW optimizer.

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', num_labels=len(labels))
                                                         
# if use_cuda:
#     model = DataParallel(model,device_ids=device_ids)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 2
t_total = len(train_dataloader) * num_train_epochs # total number of training steps 

# put the model in training mode
model.train() 
for epoch in range(num_train_epochs):  
   print("Epoch:", epoch)
   for batch in tqdm(train_dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch) 
        loss = outputs.loss
        
        # print loss every 100 steps
        if global_step % 100 == 0:
          print(f"Loss after {global_step} steps: {loss.item()}")


        loss.backward()
        optimizer.step()
        global_step += 1
    
   print(f"Saving epoch {epoch}  loss : {loss.item()}")
  #  torch.save(model, f"./tuned/layoutlmv2-finetuned-funsd-torch_epoch_{epoch}.pth")   
   model.save_pretrained(f"./checkpoints")

# # torch.save(model.state_dict(), "./tuned/layoutlmv2-finetuned-funsd-torch.pth")
# torch.save(model, "./tuned/layoutlmv2-finetuned-funsd-torch.pth")

## Evaluation

#Next, let's evaluate the model on the test set.

metric = load_metric("seqeval")

# put model in evaluation mode
model.eval()
index = 0
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():        
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        print("labels::")
        print(labels)
        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, labels=labels)
        
        # predictions
        predictions = outputs.logits.argmax(dim=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]


        metric.add_batch(predictions=true_predictions, references=true_labels)

        print('Eval info : ***********************')
        print(true_predictions)
        print(true_labels)
        print(image.shape)

        img = Image.fromarray((image[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0))
        img.save(f'/tmp/tensors/{index}.png')
        
        if False:
          draw = ImageDraw.Draw(img)
          font = ImageFont.load_default()
          bb = bbox[0].cpu().numpy()

          print('BBOXES')
          print(bb)
          for i, box in enumerate(bb):
              print(box)
              if box[3] == 0:
                continue
              # box = item["box"]
              # text = item["text"]
              text = 'Label'
              # draw.rectangle(box, outline="red")
              draw.text((box[0], box[1]), text=text, fill="blue", font=font)

          img.save(f'/tmp/tensors/bbox_{index}.png')


        index=index+1

final_score = metric.compute()
print(final_score)
