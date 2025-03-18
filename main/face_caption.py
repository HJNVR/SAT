import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from PIL import Image
from tqdm import tqdm
import json

model_path = "/data02/jing/lora-master/backbone/blip-image-caption-large/"

processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)#.to("cuda")


# translate model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tran_backbone_path = "/data02/jing/lora-master/backbone/pus-mt-en-zh"
print("loading translate backbone: ", tran_backbone_path)
tran_tokenizer = AutoTokenizer.from_pretrained(tran_backbone_path)
tran_model = AutoModelForSeq2SeqLM.from_pretrained(tran_backbone_path)


def translate(text):
    global tran_tokenizer, tran_model
    inputs = tran_tokenizer(text, return_tensors="pt")
    output = tran_model.generate(**inputs)
    output = tran_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    output = output.lower()
    if output[-1] == '.':
      output = output[:-1]
      
    return output


root = "/data02/jing/lora/face/"
count = 0
with open('face_metadata.jsonl', 'w') as outfile:
  for img_name in tqdm(os.listdir("/data02/jing/lora/face/")):
    if img_name.endswith('png'):
      img = Image.open(root + img_name).convert('RGB')
      img_size = img.size
      if (img_size[0] >= 512 and img_size[1] >= 512) or (img_size[0] / img_size[1] <=1.5) or (img_size[1] / img_size[0] <= 1.5):
        inputs = processor(img, return_tensors="pt")#.to("cuda")
        out = model.generate(**inputs)# no_repeat_ngram_size=1  no glas glas
        caption = processor.decode(out[0], skip_special_tokens=True)
        #caption = processor.decode(out[0])
        caption = ' '.join(dict.fromkeys(caption.split()))
        print(caption)
        caption = translate(caption)
        print(caption)
        #print(processor.decode(out[0], skip_special_tokens=True))
        record = {'file_name': img_name, 'text': caption}
        json.dump(record, outfile)
        outfile.write('\n')
        count += 1 
        if count == 10:
          break
      else:
        count += 1
        
print("Number of bad images: ", count)
      
    
    