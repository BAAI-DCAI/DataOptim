# DataOptim
We launch DataOptim, an MLLM benchmark and competition where we aim to find the optimal training data for training Multimodal Large Language Models (MLLMs).

- Project page 🏠: http://dataoptim.org
- HuggingFace 🤗: https://huggingface.co/datasets/BAAI/DataOptim

## Datasets
Currently, the visual instruction tuning data used in the challenge contain 14 public datasets.
More datasets are coming in the future! 🔥

|Category|Dataset|Images|QAs|Split|
|:-:|:-:|:-:|:-:|:-:|
|Image captioning|COCO|82783|414113|train|
|Image captioning|Flickr30K|29000|145000|Karpathy train split|
|Image captioning|TextCaps|21953|109765|train|
|Visual question answering|VQAv2|82783|443757|train|
|Visual question answering|OKVQA|8998|9009|train|
|Visual question answering|OCRVQA|166041|801673|train|
|Visual question answering|GQA|72140|943000|train|
|Visual question answering|TextVQA|21953|34602|train|
|Visual question answering|A-OKVQA|16540|17056|train|
|Grounding|RefCOCO/RefCOCO+/RefCOCOg|24407|287604|train|
|Grounding|Shikra-RD|883|5922|train|
|GPT-4 generated|LLaVA-Instruct-150K|81479|157712|-|
|GPT-4 generated|SVIT|108076|2992799|-|

We use different strategies to collect the prompts for different tasks.
- **Image captioning.** We carefully collect 5 manually written instructions and randomly sample one as the prompt for each caption.
- **Open-ended VQA.** As the answers in VQA datasets are generally short, we add an instruction after the question to ask the model to provide answers of appropriate length.
- **Multiple-choice VQA.** We add an instruction before the question to ask the model to provide answers with correct options.
- **Grounding.** We use the templates designed in [Shikra](https://github.com/shikras/shikra) and randomly sample one to format the prompts.
- **GPT-4 generated datasets.** We keep the prompts unchanged.

|Category|Data|Prompts|
|:-:|:-:|:-:|
|Image captioning|COCO, Flickr30K, TextCaps|Describe the image as simply as possible with a sentence or phrase.<br />Give a brief summary of what you see.<br />Provide a short description of the image.<br />Write a short description for the image.<br />Briefly describe the content of the image.|
|Open-ended VQA|VQAv2, OKVQA, OCRVQA, GQA, TextVQA|*question* Answer the question directly with a short sentence or phrase.|
|Multiple-choice VQA|A-OKVQA|Choose the correct option for the following question: *question*|

## Prerequisites

### Models
For now we use LLaVA as the fixed model.
You can prepare the environment according to the [instructions](https://github.com/haotian-liu/LLaVA).

### Images
You can download the images from our HuggingFace repository or the original websites.
If you already have the images, you can skip the above process as the image IDs and file names remain unchanged.

Then unzip and organize the images in following structure:
```
|- images
  |- coco
  |- filckr30k
  |- ocrvqa
  |- open_images
  |- visual_genome
```

After that, you can use this diretory as the `--image-folder` in LLaVA's training script.

### Visual instruction tuning data
In our HuggingFace repository, all of the visual instruction tuning data mentioned above are already converted to the input format of LLaVA.
You can download them directly from [data.zip](https://huggingface.co/datasets/BAAI/DataOptim/blob/main/data/data.zip).

## How to participate
To participate the challenge, visit the [project page](http://dataoptim.org) for more details.

The example submission file for the standard setting and BYOD setting can be found [here](./example/example-standard.txt) and [here](./example/example-byod.json) respectively.

We will finetune the model with your selected data and respond with the evaluation results.

## Training & evaluation details
In the training stage, we first sample the data in the submitted file.
Then based on the [pretrained weights](https://huggingface.co/liuhaotian/llava-pretrain-llama-2-7b-chat) of LLaVA-LLaMA-2-7B, we perform the finetuning stage with selected data.

The training script is shown as follows:
```
################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /path/to/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /path/to/your-selected-data \
    --image_folder /path/to/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /path/to/llava-pretrain-llama-2-7b-chat/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

In the evaluation stage, we prompt the model with questions in the benchmark and calculate the final scores.

Currently, we use LLaVA at commit@744cb3e.
More models may be supported in future.
