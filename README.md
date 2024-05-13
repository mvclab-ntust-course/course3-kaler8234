LoRA Training
===

## 使用說明：
先裝 diffuser
```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

再進到裡面裝 text_to_image 的套件
```
cd examples/text_to_image
pip install -r requirements.txt
```

輸入下方指令後會有選單可以選，根據使用設備輸入
```
accelerate config
```

輸入下方指令即可訓練(可能要根據不同的路徑改 --dataset_name 以及 --output_dir)
```
accelerate launch --mixed_precision="bf16"  train_text_to_image_lora.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" --dataset_name="C:/Users/kaler/Desktop/anaconda/LoRA/diffusers/examples/text_to_image/dataset" --dataloader_num_workers=4 --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=15000 --learning_rate=1e-04 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir="C:/Users/kaler/Desktop/anaconda/LoRA/diffusers/examples/text_to_image/models" --report_to=wandb --checkpointing_steps=500 --validation_prompt="fhirou, sitting, confusing" --seed=1337 --resume_from_checkpoint="latest"
```

### 輸出圖片
* **prompt = fhirou**
![output](https://cdn.discordapp.com/attachments/871761664503074868/1239603804647981116/media_images_validation_4032_375445981f32b0bfcd82.png?ex=66438681&is=66423501&hm=7220bc59c32b36bd6e92aa8dc1a6c1bd8d412d68e0f624633fe6f0b1a3173518& x200)
