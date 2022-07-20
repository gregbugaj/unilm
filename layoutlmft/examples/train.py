import os

import numpy as np
import torch
from torch.nn import DataParallel
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import DataLoader
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2Processor, AdamW

from transformers import LayoutLMv2Processor, LayoutLMv2FeatureExtractor, LayoutLMv2ForTokenClassification, \
    LayoutLMv2TokenizerFast


from tqdm import tqdm

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

from datasets import load_dataset, load_metric
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler


import warnings 
warnings.filterwarnings('ignore')

# Spliting into multiple words/tokens
# https://github.com/NielsRogge/Transformers-Tutorials/issues/41

# https://towardsdatascience.com/fine-tuning-transformer-model-for-invoice-recognition-1e55869336d4
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb#scrollTo=VMYoOQuyp4NT
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb#scrollTo=NxahVHZ0NKq7


# Complete CORD
# https://github.com/katanaml/sparrow/blob/f18591947f2b1e26fbac045ba430e73af2c68f0c/research/app/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD_using_HuggingFace_Trainer_ipynb.ipynb


# Sequence classificatin
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb


# {'ANSWER': {'precision': 0.6618106139438086, 'recall': 0.7861557478368356, 'f1': 0.7186440677966102, 'number': 809}, 'HEADER': {'precision': 0.4625, 'recall': 0.31092436974789917, 'f1': 0.3718592964824121, 'number': 119}, 'QUESTION': {'precision': 0.7798085291557877, 'recall': 0.8413145539906103, 'f1': 0.8093947606142727, 'number': 1065}, 'overall_precision': 0.7164383561643836, 'overall_recall': 0.787255393878575, 'overall_f1': 0.7501792971551519, 'overall_accuracy': 0.679780527667522
# {'ANSWER': {'precision': 0.6998904709748083, 'recall': 0.7898640296662547, 'f1': 0.7421602787456447, 'number': 809}, 'HEADER': {'precision': 0.4375, 'recall': 0.17647058823529413, 'f1': 0.25149700598802394, 'number': 119}, 'QUESTION': {'precision': 0.8159448818897638, 'recall': 0.7784037558685446, 'f1': 0.7967323402210476, 'number': 1065}, 'overall_precision': 0.7531613555892767, 'overall_recall': 0.7471149021575514, 'overall_f1': 0.7501259445843829, 'overall_accuracy': 0.7541442913845435}

use_cuda = torch.cuda.is_available()
device= torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
device_ids = [0]

# cache_dir, data_dir
dataset = load_dataset("funsd_dataset/funsd_dataset.py", cache_dir="/mnt/data/cache/")
 
# print(dataset['train'].features)
print(dataset['train'].features['bboxes'])

labels = dataset['train'].features['ner_tags'].feature.names

print('NER-Labels -> ')
print(labels)

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

print("ID2Label : ")
print(id2label)
print(label2id) 


feature_size = 224 * 2 # 224
batch_size   = 8 #  28
 
##Next, let's use `LayoutLMv2Processor` to prepare the data for the model.
# 115003 / 627003

feature_extractor = LayoutLMv2FeatureExtractor(size = feature_size, apply_ocr=False)
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-large-uncased")
processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


#processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, feature_size, feature_size)), # 224
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
})

def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  # images = [image for image in examples['image']]
  words = examples['words']
  boxes = examples['bboxes']
  word_labels = examples['ner_tags']
  
  encoded_inputs = processor(images, words, boxes=boxes, word_labels=word_labels, padding="max_length", truncation=True)

  return encoded_inputs


# We should us this to train on all words
# only_label_first_subword

train_dataset = dataset['train'].map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names,
                                      features=features, num_proc = 4)
test_dataset = dataset['test'].map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names,
                                      features=features, num_proc = 4)

# num_proc = int(mp.cpu_count() // 8)
print("Data size: ")
print(f"Data train: {len(train_dataset)}")
print(f"Data test: {len(test_dataset)}")


##Finally, let's set the format to PyTorch, and place everything on the GPU:

train_dataset.set_format(type="torch", device=device)
test_dataset.set_format(type="torch", device=device)

print(f'keys : {train_dataset.features.keys()}')

decoded = processor.tokenizer.decode(train_dataset['input_ids'][0])

print('Train labels **************')
print(train_dataset['labels'][0])

##Next, we create corresponding dataloaders.

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)

##Let's verify a batch:

batch = next(iter(train_dataloader))

print("Batch verification : ")
for k,v in batch.items():
  print(k, v.shape)

# os.exit()
os.makedirs(f"./checkpoints", exist_ok = True)
# model = LayoutLMv2ForTokenClassification.from_pretrained("./checkpoints")

## Train the model

if True:
  # model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', num_labels=len(labels))
  model = LayoutLMv2ForTokenClassification.from_pretrained("/mnt/data/models/layoutlmv2-large-finetuned-funsd", num_labels=len(labels))

  # Set id2label and label2id
  model.config.id2label = id2label
  model.config.label2id = label2id

  # if use_cuda:
  #     model = DataParallel(model,device_ids=device_ids)
  model.to(device)


def train():
  # define a summary writer that logs data and flushes to the file every 5 seconds
  sw = SummaryWriter(log_dir='./logs', flush_secs=5)
  optimizer = AdamW(model.parameters(), lr=5e-5)

  best_epoch = -1
  best_acc = 0.0
  global_step = 0

  num_training_epochs = 100
  num_training_steps = len(train_dataloader) * num_training_epochs # total number of training steps 

  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps,
  )

  print(f"num_training_steps = {num_training_steps}")

  # progress_bar = tqdm(range(num_training_steps))

  # put the model in training mode
  model.train() 
  best_loss = 1

  for epoch in range(num_training_epochs):  
    print(f"Epoch: {epoch}")

    total_loss = 0
    for batch in tqdm(train_dataloader):
    # for batch in train_dataloader:
          # forward + backward + optimize
          outputs = model(**batch) 
          loss = outputs.loss
          loss.backward()
          train_loss = loss.item()

          # print loss every 100 steps
          if global_step % 100 == 0:
            print(f"Loss after {global_step} steps: {loss.item()}")

            # Save if the current loss is better then previous
            if train_loss < best_loss:
              print(f"Saving @step {global_step}  loss : {loss.item()}")
              best_loss = train_loss
              out_dir = f"/mnt/data/models/layoutlmv2-large-finetuned-funsd_{global_step}_{best_loss}"
              os.mkdir(out_dir)
              model.save_pretrained(out_dir)

          total_loss += train_loss

          optimizer.step()
          lr_scheduler.step()          
          optimizer.zero_grad()# zero the parameter gradients

          sw.add_scalar('train_loss', train_loss, global_step=global_step)
          global_step += 1
          # progress_bar.update(1)
      
    sw.add_scalar("Loss", total_loss, epoch)
    #  torch.save(model, f"./tuned/layoutlmv2-finetuned-funsd-torch_epoch_{epoch}.pth")   
    if train_loss < best_loss:
      print(f"Saving epoch {epoch}  loss : {loss.item()}")
      model.save_pretrained(f"/mnt/data/models/layoutlmv2-large-finetuned-funsd")

    evaluate(model)

  # # torch.save(model.state_dict(), "./tuned/layoutlmv2-finetuned-funsd-torch.pth")


## Evaluation
#Next, let's evaluate the model on the test set.
def evaluate(model):
  print('Evaluating model')

  metric = load_metric("seqeval")
  # put model in evaluation mode
  # model.eval()

  index = 0
  for batch in tqdm(test_dataloader, desc="Evaluating"):
      with torch.no_grad():        
          input_ids = batch['input_ids'].to(device)
          bbox = batch['bbox'].to(device)
          image = batch['image'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          token_type_ids = batch['token_type_ids'].to(device)
          labels = batch['labels'].to(device)

          # print("labels::")
          # print(labels)
          # forward pass
          outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                          token_type_ids=token_type_ids, labels=labels)
          
          # predictions
          predictions = outputs.logits.argmax(dim=2)

          # print(f'predictions shape : {predictions.shape}')
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

          # print('Eval info : ***********************')
          # print(true_labels)
          # print(true_predictions)
          # print(image.shape)

          if False:
            img = Image.fromarray((image[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0))
            img.save(f'/tmp/tensors/{index}.png')
            
          if False:
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            bb = bbox[0].cpu().numpy()

            print('BBOXES')
            for i, box in enumerate(bb):
                print(box)
                print(box.sum())
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
  return final_score


# 'DOS': {'precision': 1.0, 'recall': 0.9953703703703703, 'f1': 0.9976798143851509, 'number': 648}, 'DOS_ANSWER': {'precision': 1.0, 'recall': 0.9953703703703703, 'f1': 0.9976798143851509, 'number': 648}, 'MEMBER_NAME': {'precision': 0.9830508474576272, 'recall': 0.9666666666666667, 'f1': 0.9747899159663865, 'number': 720}, 'MEMBER_NAME_ANSWER': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 720}, 'MEMBER_NUMBER': {'precision': 0.9873417721518988, 'recall': 0.9873417721518988, 'f1': 0.9873417721518988, 'number': 948}, 'MEMBER_NUMBER_ANSWER': {'precision': 0.997872340425532, 'recall': 0.989451476793249, 'f1': 0.9936440677966102, 'number': 948}, 'PAN': {'precision': 0.95, 'recall': 1.0, 'f1': 0.9743589743589743, 'number': 228}, 'PAN_ANSWER': {'precision': 0.9537815126050421, 'recall': 0.9956140350877193, 'f1': 0.9742489270386266, 'number': 228}, 'PATIENT_NAME': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 708}, 'PATIENT_NAME_ANSWER': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 708}, 'overall_precision': 0.9924615384615385, 'overall_recall': 0.9918511685116851, 'overall_f1': 0.9921562596124269, 'overall_accuracy': 0.9997116988710906}

# Big dataset
#{'DOS': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 4751}, 'DOS_ANSWER': {'precision': 0.9997895179962113, 'recall': 0.9997895179962113, 'f1': 0.9997895179962113, 'number': 4751}, 'MEMBER_NAME': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 3000}, 'MEMBER_NAME_ANSWER': {'precision': 0.9976720984369803, 'recall': 1.0, 'f1': 0.9988346928583319, 'number': 3000}, 'MEMBER_NUMBER': {'precision': 0.8487770041064095, 'recall': 0.8266388454181881, 'f1': 0.8375616631430585, 'number': 5751}, 'MEMBER_NUMBER_ANSWER': {'precision': 0.9941605839416059, 'recall': 0.8288993218570684, 'f1': 0.9040394462355397, 'number': 5751}, 'PAN': {'precision': 0.6640106241699867, 'recall': 1.0, 'f1': 0.7980845969672785, 'number': 2000}, 'PAN_ANSWER': {'precision': 0.666098807495741, 'recall': 0.9775, 'f1': 0.7922998986828773, 'number': 2000}, 'PATIENT_NAME': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1751}, 'PATIENT_NAME_ANSWER': {'precision': 0.9982886480319453, 'recall': 0.9994288977727013, 'f1': 0.9988584474885844, 'number': 1751}, 'overall_precision': 0.9185993890711619, 'overall_recall': 0.9412276125891149, 'overall_f1': 0.929775843806361, 'overall_accuracy': 0.9984652360490871}

if __name__ == "__main__":
  
  os.environ['TRANSFORMERS_CACHE'] = '/mnt/data/cache/'
  train()
  
  # # model = LayoutLMv2ForTokenClassification.from_pretrained("./checkpoints", num_labels=len(labels))
  # model = LayoutLMv2ForTokenClassification.from_pretrained('/home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints/checkpoint-8500/', num_labels=len(labels))
  # model = LayoutLMv2ForTokenClassification.from_pretrained('/home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints-tuned/checkpoint-700', num_labels=len(labels))
  # model = LayoutLMv2ForTokenClassification.from_pretrained('/home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints-tuned-noaccelerate/checkpoint-13300', num_labels=len(labels))

  # model.to(device)
  # evaluate(model)

# Generating train split: 604942 examples [14:11, 1582.31 examples/s]Time elapsed[all]: 850.1650393009186
# 32 bit
# {'ADDRESS': {'precision': 0.7618739903069467, 'recall': 0.84888, 'f1': 0.8030271497493142, 'number': 25000}, 'ANSWER': {'precision': 0.8337747081629626, 'recall': 0.7568269230769231, 'f1': 0.7934395822622756, 'number': 104000}, 'DOCUMENT_CONTROL': {'precision': 0.6951983298538622, 'recall': 0.5449090909090909, 'f1': 0.610946896340842, 'number': 5500}, 'DOS': {'precision': 0.8662547130289066, 'recall': 0.7693953488372093, 'f1': 0.8149571386343482, 'number': 10750}, 'DOS_ANSWER': {'precision': 0.926337359792925, 'recall': 0.8379512195121951, 'f1': 0.8799303350066592, 'number': 10250}, 'GREETING': {'precision': 0.8458681522748375, 'recall': 0.8574117647058823, 'f1': 0.8516008413180649, 'number': 17000}, 'HEADER': {'precision': 0.6443639394716594, 'recall': 0.6232248062015504, 'f1': 0.6336181078780618, 'number': 32250}, 'LETTER_DATE': {'precision': 0.9152302243211334, 'recall': 0.9492244897959183, 'f1': 0.9319174514125426, 'number': 12250}, 'MEMBER_NAME': {'precision': 0.9892458293120088, 'recall': 0.9566666666666667, 'f1': 0.9726835219955263, 'number': 7500}, 'MEMBER_NAME_ANSWER': {'precision': 0.9183295890048897, 'recall': 0.868625, 'f1': 0.8927860217125971, 'number': 8000}, 'MEMBER_NUMBER': {'precision': 0.9928425357873211, 'recall': 0.971, 'f1': 0.9817997977755308, 'number': 11000}, 'MEMBER_NUMBER_ANSWER': {'precision': 0.9996277338296883, 'recall': 0.9764545454545455, 'f1': 0.9879052655782938, 'number': 11000}, 'PAN': {'precision': 0.6745886654478976, 'recall': 0.984, 'f1': 0.8004338394793927, 'number': 1500}, 'PAN_ANSWER': {'precision': 0.6123218776194468, 'recall': 0.974, 'f1': 0.7519300051466804, 'number': 1500}, 'PARAGRAPH': {'precision': 0.5954225187812048, 'recall': 0.6874343434343434, 'f1': 0.638128695913086, 'number': 74250}, 'PATIENT_NAME': {'precision': 0.8690516368545393, 'recall': 0.8956521739130435, 'f1': 0.8821514217197671, 'number': 5750}, 'PATIENT_NAME_ANSWER': {'precision': 0.9284476784476784, 'recall': 0.8931666666666667, 'f1': 0.9104655113829425, 'number': 6000}, 'PHONE': {'precision': 1.0, 'recall': 0.5813333333333334, 'f1': 0.7352445193929175, 'number': 750}, 'QUESTION': {'precision': 0.9005717653993516, 'recall': 0.9203734939759036, 'f1': 0.9103649635036497, 'number': 83000}, 'URL': {'precision': 0.8691543882126842, 'recall': 0.7752857142857142, 'f1': 0.819540924192087, 'number': 7000}, 'overall_precision': 0.7984122509688524, 'overall_recall': 0.8022567645365573, 'overall_f1': 0.8003298908442655, 'overall_accuracy': 0.9156411921141673}
# {'ADDRESS': {'precision': 0.7865506205604559, 'recall': 0.80612, 'f1': 0.7962150843506776, 'number': 25000}, 'ANSWER': {'precision': 0.8501443153040574, 'recall': 0.7391923076923077, 'f1': 0.7907955170835327, 'number': 104000}, 'DOCUMENT_CONTROL': {'precision': 0.7127614169867691, 'recall': 0.6072727272727273, 'f1': 0.6558020812880424, 'number': 5500}, 'DOS': {'precision': 0.890881913303438, 'recall': 0.7761860465116279, 'f1': 0.8295883873533506, 'number': 10750}, 'DOS_ANSWER': {'precision': 0.924542600411389, 'recall': 0.833170731707317, 'f1': 0.8764817570688151, 'number': 10250}, 'GREETING': {'precision': 0.7720373446360044, 'recall': 0.7928823529411765, 'f1': 0.7823210191822165, 'number': 17000}, 'HEADER': {'precision': 0.6990067990934542, 'recall': 0.6503255813953488, 'f1': 0.6737880296848394, 'number': 32250}, 'LETTER_DATE': {'precision': 0.930727969348659, 'recall': 0.9915102040816327, 'f1': 0.9601581027667985, 'number': 12250}, 'MEMBER_NAME': {'precision': 0.999723909442297, 'recall': 0.9656, 'f1': 0.9823657080846445, 'number': 7500}, 'MEMBER_NAME_ANSWER': {'precision': 0.9087003222341569, 'recall': 0.846, 'f1': 0.8762299326773693, 'number': 8000}, 'MEMBER_NUMBER': {'precision': 0.9803609838211914, 'recall': 0.953, 'f1': 0.9664868851703314, 'number': 11000}, 'MEMBER_NUMBER_ANSWER': {'precision': 0.9909876428505063, 'recall': 0.9696363636363636, 'f1': 0.9801957450719111, 'number': 11000}, 'PAN': {'precision': 0.6496887379739672, 'recall': 0.7653333333333333, 'f1': 0.7027854300581573, 'number': 1500}, 'PAN_ANSWER': {'precision': 0.7185226636821489, 'recall': 0.856, 'f1': 0.7812595071493763, 'number': 1500}, 'PARAGRAPH': {'precision': 0.6005356480983297, 'recall': 0.6764579124579124, 'f1': 0.6362398424189452, 'number': 74250}, 'PATIENT_NAME': {'precision': 0.8970454957404387, 'recall': 0.8606956521739131, 'f1': 0.8784947190911513, 'number': 5750}, 'PATIENT_NAME_ANSWER': {'precision': 0.9018823921372647, 'recall': 0.9023333333333333, 'f1': 0.902107806381738, 'number': 6000}, 'PHONE': {'precision': 1.0, 'recall': 0.48133333333333334, 'f1': 0.6498649864986499, 'number': 750}, 'QUESTION': {'precision': 0.8901784167677118, 'recall': 0.9287349397590361, 'f1': 0.9090480261800171, 'number': 83000}, 'URL': {'precision': 0.8729788298049675, 'recall': 0.7481428571428571, 'f1': 0.8057542887914455, 'number': 7000}, 'overall_precision': 0.8060397741222686, 'overall_recall': 0.7938169257340242, 'overall_f1': 0.7998816588642731, 'overall_accuracy': 0.9166312911240683}
# {'ADDRESS': {'precision': 0.7297903054247075, 'recall': 0.76844, 'f1': 0.748616631595355, 'number': 25000}, 'ANSWER': {'precision': 0.8280905196659599, 'recall': 0.7360769230769231, 'f1': 0.779377328907984, 'number': 104000}, 'DOCUMENT_CONTROL': {'precision': 0.6814767117407169, 'recall': 0.5772727272727273, 'f1': 0.6250615218033272, 'number': 5500}, 'DOS': {'precision': 0.8558310376492194, 'recall': 0.7802790697674419, 'f1': 0.8163106418179163, 'number': 10750}, 'DOS_ANSWER': {'precision': 0.9251197213757074, 'recall': 0.8292682926829268, 'f1': 0.874575573618685, 'number': 10250}, 'GREETING': {'precision': 0.7612435691763014, 'recall': 0.8094705882352942, 'f1': 0.7846167004019728, 'number': 17000}, 'HEADER': {'precision': 0.7083931007941784, 'recall': 0.6278449612403101, 'f1': 0.6656913188565418, 'number': 32250}, 'LETTER_DATE': {'precision': 0.9919354838709677, 'recall': 0.9639183673469388, 'f1': 0.977726256520659, 'number': 12250}, 'MEMBER_NAME': {'precision': 0.9968345719790807, 'recall': 0.9657333333333333, 'f1': 0.9810375186238658, 'number': 7500}, 'MEMBER_NAME_ANSWER': {'precision': 0.9155778894472362, 'recall': 0.911, 'f1': 0.9132832080200501, 'number': 8000}, 'MEMBER_NUMBER': {'precision': 0.982533150434385, 'recall': 0.9767272727272728, 'f1': 0.9796216093002053, 'number': 11000}, 'MEMBER_NUMBER_ANSWER': {'precision': 0.989867354458364, 'recall': 0.976909090909091, 'f1': 0.9833455344070279, 'number': 11000}, 'PAN': {'precision': 0.6691331923890064, 'recall': 0.844, 'f1': 0.7464622641509434, 'number': 1500}, 'PAN_ANSWER': {'precision': 0.6144444444444445, 'recall': 0.7373333333333333, 'f1': 0.6703030303030303, 'number': 1500}, 'PARAGRAPH': {'precision': 0.5724906230756311, 'recall': 0.6886464646464646, 'f1': 0.6252193317641304, 'number': 74250}, 'PATIENT_NAME': {'precision': 0.8954307386671483, 'recall': 0.8622608695652174, 'f1': 0.8785328253743245, 'number': 5750}, 'PATIENT_NAME_ANSWER': {'precision': 0.9855734112490869, 'recall': 0.8995, 'f1': 0.9405716277448588, 'number': 6000}, 'PHONE': {'precision': 0.9822134387351779, 'recall': 0.6626666666666666, 'f1': 0.7914012738853503, 'number': 750}, 'QUESTION': {'precision': 0.8929518935036274, 'recall': 0.9164698795180722, 'f1': 0.9045580488268942, 'number': 83000}, 'URL': {'precision': 0.8985313751668892, 'recall': 0.7691428571428571, 'f1': 0.8288177339901478, 'number': 7000}, 'overall_precision': 0.791953442579104, 'overall_recall': 0.7909522164651699, 'overall_f1': 0.7914525128722705, 'overall_accuracy': 0.9114579301561935}
