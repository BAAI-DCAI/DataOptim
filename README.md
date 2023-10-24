# DataOptim
We launch DataOptim, an MLLM benchmark and competition where we aim to find the optimal training data for training Multimodal Large Language Models (MLLMs).

- Project page 🏠: http://dataoptim.org
- HuggingFace 🤗: https://huggingface.co/datasets/BAAI/DataOptim

## News
- [2023/10/17] ScienceQA is now available in the training datasets.

## Training datasets
Currently, the visual instruction tuning data used in the challenge contain 14 public datasets.
More datasets are coming in the future! 🔥🔥🔥

|Category|Dataset|Images|Samples|Split|
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
|Visual question answering|ScienceQA|6218|6218|train|
|Grounding|RefCOCO/RefCOCO+/RefCOCOg|24407|287604|train|
|Grounding|Shikra-RD|883|5922|train|
|GPT-4 generated|LLaVA-Instruct-150K|81479|157712|-|
|GPT-4 generated|SVIT|108076|2992799|-|
|Total||414K|6.37M|

We use different strategies to collect the prompts for different tasks.
- **Image captioning.** We carefully collect 5 manually written instructions and randomly sample one as the prompt for each caption. The fourth and fifth instructions are from [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md).
- **Open-ended VQA.** As the answers in VQA datasets are generally short, we add an instruction after the question to ask the model to provide answers with a short sentence or phrase.
- **Multiple-choice VQA.** For OK-VQA, we add an instruction before the question to ask the model to provide answers with correct options. For ScienceQA, we use the instructions and templates designed by [M3IT](https://m3-it.github.io/) and randomly sample one to format the prompt. Only data with image context are involved.
- **Grounding.** We use the templates designed by [Shikra](https://github.com/shikras/shikra) and randomly sample one to format the prompt.
- **GPT-4 generated datasets.** We keep the prompts unchanged.

|Category|Data|Prompts|
|:-:|:-:|:-:|
|Image captioning|COCO, Flickr30K, TextCaps|Describe the image as simply as possible with a sentence or phrase.<br />Give a brief summary of what you see.<br />Provide a short description of the image.<br />Write a short description for the image.<br />Briefly describe the content of the image.|
|Open-ended VQA|VQAv2, OKVQA, OCRVQA, GQA, TextVQA|*question* Answer the question directly with a short sentence or phrase.|
|Multiple-choice VQA|A-OKVQA|Choose the correct option for the following question: *question*|

## Prerequisites

### Models
For now we use LLaVA-LLaMA-2-7B as the fixed model.
In this release, we use LLaVA at commit@744cb3e.

To start training, you need to apply for and download the LLaMA-2-7B-chat-hf checkpoints [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and download the [LLaVA pretrained weights](https://huggingface.co/liuhaotian/llava-pretrain-llama-2-7b-chat).

Then you can prepare the environment of LLaVA according to the [instructions](https://github.com/haotian-liu/LLaVA#install).

### Datasets
For training images, you can download the images from our [HuggingFace repository](https://huggingface.co/datasets/BAAI/DataOptim/tree/main/images) or the original websites.
If you already have the images, you can skip this process as the image IDs and file names are not changed.

Then unzip and organize the images in following structure.
For all datasets except ScienceQA, the images should be placed directly under the dataset's folder.
For ScienceQA, the image is placed in a subfolder with the image's ID as the folder's name.
This is the original structure of ScienceQA and we do not make any changes to it to keep consistency.

```
|- images
  |- coco
    |- COCO_train2014_000000000009.jpg
    |- COCO_train2014_000000000025.jpg
    |- ...
  |- filckr30k
    |- 36979.jpg
    |- 65567.jpg
    |- ...
  |- ocrvqa
    |- 13714.jpg
    |- ...
  |- open_images
    |- 0a0bc91825468c45.jpg
    |- ...
  |- visual_genome
    |- 1.jpg
    |- ...
  |- scienceqa
    |- 1
      |- image.png
    |- 2
      |- image.png
    |- ...
```

After that, you can use this diretory as the `--image_folder` in LLaVA's training script.

For the visual instruction tuning QAs, all of the data mentioned above are already converted to the training format of LLaVA in our HuggingFace repository.
You can download them directly from [data.zip](https://huggingface.co/datasets/BAAI/DataOptim/blob/main/data/data.zip).

## How to participate
To participate the challenge, visit the [project page](http://dataoptim.org) for more details.
Basically, the target is to find a subset of data that can best enhance the model's abilities.
You can design your own method to find this subset.

For standard setting, you only need to submit the data **IDs**.
We will sample the data according to the IDs.
For the BYOD setting, you need to update the **full dataset** you select.
The example submission file for both settings can be found in [example-standard.txt](./example/example-standard.txt) and [example-byod.json](./example/example-byod.json), respectively.
In the example, we sample 17 data from the dataset just to show the format.
You can sample a specific amount of data according to the competition you participate.

Before submitting the your selected data, we recommend training and evaluating the selected data locally to validate their effectiveness.

### Training
We provided a [script](./tools/get_subset_from_ids.py) to help convert the data IDs to the training format of LLaVA.
You can use this file as the `--data_path` in LLaVA's training script.

The training script used in the competition is shown as follows, which is modified based on LLaVA's finetuning [script](https://github.com/haotian-liu/LLaVA/blob/main/scripts/finetune.sh).
For 100K data, it takes around one hour on 8xA100 40G.
If the training resources are limited, you can train and evaluate the model with the support of [LoRA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/finetune_lora.sh) or [QLoRA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/finetune_qlora.sh) locally.

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

### Evaluation
For evaluation, we release a validation set of our benchmark to help local evaluation. More information about the benchmark will be released soon.

We also provide a [script](./evaluation/run_llava_eval.py) to prompt the model with questions in the benchmark.
To run the script, you need to 
1. Download and unzip the validation set [images](https://huggingface.co/datasets/BAAI/DataOptim/blob/main/evaluation/val.zip)
2. Download the validation set [QAs](https://huggingface.co/datasets/BAAI/DataOptim/blob/main/evaluation/dataoptim_val.json)
3. Copy the [script](./evaluation//run_llava_eval.py) to LLaVA's code base with location `LLaVA/llava/eval/run_llava_eval.py`.

Then run the script with following command:

```
python -m llava.eval.run_llava_eval \
    --model-path /path/to/your/model \
    --image-dir /path/to/val \
    --input-file /path/to/dataoptim_val.json \
    --result-dir /path/to/output \
    --conv-mode llava_llama_2
```

You can find the results in the `result_dir`.

## Contact
If you have any questions, you can open an issue in the GitHub repository or contact zhaobo@baai.ac.cn for more information.
