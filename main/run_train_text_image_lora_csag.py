import os
import moxing as mox

os.system(f'pip install datasets')
#os.system(f'pip show diffusers')
import datasets

mox.file.shift('os', 'mox')

root_path = "/cache/"

print("Loading program ...")
mox.file.copy_parallel(os.path.join('s3://bucket-852/jing/SD_lora_face'), os.path.join(root_path, 'SD_lora_face'))

os.system(f'pip install /cache/SD_lora_face/diffusers-0.16.1-py3-none-any.whl')
os.system(f'pip show diffusers')

print("Loading backbone ...")
mox.file.copy_parallel(os.path.join('s3://bucket-852/jing/backbone/stable-diffusion-v1-5'), os.path.join(root_path, 'stable-diffusion-v1-5'))
mox.file.copy_parallel(os.path.join('s3://bucket-852/jing/backbone/sd-vae-ft-mse'), os.path.join(root_path, 'sd-vae-ft-mse'))
#mox.file.copy_parallel(os.path.join('s3://bucket-852/jing/backbone/taiyi-v1_5'), os.path.join(root_path, 'taiyi-v1_5'))

print('Loading data ... ')
mox.file.copy_parallel('s3://bucket-852/jing/SD_lora_face/data', os.path.join(root_path, 'data'))

zip_name = "images_crop_cag_noun.zip"
os.system(f'cd /cache/data && unzip {zip_name}')
print(os.listdir('/cache/data'))

PRETRAINED_MODEL = os.path.join(root_path, 'stable-diffusion-v1-5')
DATASET_NAME = '/cache/data/images_crop_cag_noun'
ATTENTION_GUIDE_TYPE = 'sag' #'cag'#'org'
ATTENTION_GUIDE_RATIO = 0.01 #0.01 #0.1
BATCH_SIZE = 2 #16 # 4 #2 
OUTPUT_DIR = os.path.join(root_path, 'bb_output_lora') 
STEPS = 10000 #10000 #5000 #2000 #500 #4000 # steps = total number of images / batch size = 17022/4

print(ATTENTION_GUIDE_TYPE, ATTENTION_GUIDE_RATIO)

os.system(

  f'accelerate launch --multi_gpu --gpu_ids="0,1,2,3,4" --num_processes=4 --mixed_precision="fp16" /cache/SD_lora_face/train_text_to_image_lora_csag.py '
  f'--pretrained_model_name_or_path="{PRETRAINED_MODEL}" '
  f'--train_data_dir="{DATASET_NAME}" '
  f'--dataloader_num_workers=8 '
  f'--resolution=512 --center_crop --random_flip '
  f'--train_batch_size={BATCH_SIZE} '
  f'--gradient_accumulation_steps=1 '
  f'--max_train_steps={STEPS} '
  f'--learning_rate=1e-04 '
  f'--max_grad_norm=1 '
  f'--lr_scheduler="cosine" '
  f'--lr_warmup_steps=1000 '
  f'--attention_guide_type={ATTENTION_GUIDE_TYPE} '
  f'--attention_guide_ratio={ATTENTION_GUIDE_RATIO} '
  f'--output_dir="{OUTPUT_DIR}" '
  f'--seed=2023 '
  )

# random flip
#f'accelerate launch --multi_gpu --gpu_ids="0,1,2,3,4,5,6,7" --num_processes=8 --mixed_precision="fp16" /cache/SD_lora_face/train_text_to_image_lora_csag.py '
# f'--snr_gamma=5 '
# f'--resume_from_checkpoint="latest" '
# f'--checkpointing_steps=500 '

#mox.file.copy_parallel(OUTPUT_DIR, 's3://bucket-852/jing/SD_lora_face/csag_original_lora')
#mox.file.copy_parallel(OUTPUT_DIR, 's3://bucket-852/jing/SD_lora_face/csag_cag_0_01_lora')
#mox.file.copy_parallel(OUTPUT_DIR, 's3://bucket-852/jing/SD_lora_face/csag_cag_1_lora')
#mox.file.copy_parallel(OUTPUT_DIR, 's3://bucket-852/jing/SD_lora_face/csag_sag_1_lora')
#mox.file.copy_parallel(OUTPUT_DIR, 's3://bucket-852/jing/SD_lora_face/csag_sag_0_1_lora')
mox.file.copy_parallel(OUTPUT_DIR, 's3://bucket-852/jing/SD_lora_face/csag_sag_0_01_lora')
print("Lora training completed.")
#f'--noise_offset=0.5 '
#f'--resume_from_checkpoint="latest" '








