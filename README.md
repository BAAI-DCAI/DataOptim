# DataOptim
DataOptim is a data repository designed to offer an optimized solution for utilizing training data in Multimodal Large Language Models (MLLMs) efficiently.

- HuggingFace 🤗: https://huggingface.co/datasets/BAAI/DataOptim

## News
- [2024.03.14] Data of TextOCR-GPT4V is now available!
- [2023.12.15] Data of ShareGPT4V is now available!
- [2023.11.06] Data of LLaVA-v1.5 is now available!
- [2023.10.26] VGQA, DocVQA and DVQA are now available!
- [2023.10.17] ScienceQA is now available!

## Introduction
Currently, the visual instruction tuning data contain 20 public datasets.
More datasets are coming in the future! 🔥🔥🔥

|Category|Dataset|Images|Samples|Split|
|:-:|:-:|:-:|:-:|:-:|
|Image captioning|[COCO](https://cocodataset.org/#home)|82783|414113|train|
|Image captioning|[Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)|29000|145000|Karpathy train split|
|Image captioning|[TextCaps](https://textvqa.org/textcaps/)|21953|109765|train|
|Image captioning|[TextOCR-GPT4V](https://huggingface.co/datasets/jimmycarter/textocr-gpt4v)|25114|25114|train|
|Visual question answering|[VQAv2](https://visualqa.org/)|82783|443757|train|
|Visual question answering|[OKVQA](https://okvqa.allenai.org/)|8998|9009|train|
|Visual question answering|[OCRVQA](https://ocr-vqa.github.io/)|166041|801673|train|
|Visual question answering|[GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html)|72140|943000|train|
|Visual question answering|[TextVQA](https://textvqa.org/)|21953|34602|train|
|Visual question answering|[A-OKVQA](https://allenai.org/project/a-okvqa/home)|16540|17056|train|
|Visual question answering|[ScienceQA](https://scienceqa.github.io/)|6218|6218|train|
|Visual question answering|[Visual Genome QA (VGQA)](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)|99280|1445322|-|
|Visual question answering|[DocVQA](https://www.docvqa.org/)|10194|39463|train|
|Visual question answering|[DVQA](https://github.com/kushalkafle/DVQA_dataset)|200000|2325316|train|
|Grounding|[RefCOCO/RefCOCO+/RefCOCOg](https://github.com/lichengunc/refer)|24407|287604|train|
|Grounding|[Shikra-RD](https://github.com/shikras/shikra)|883|5922|train|
|GPT-4 generated|[LLaVA-Instruct-150K](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)|81479|157712|-|
|GPT-4 generated|[SVIT](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning)|108076|2992799|-|
|GPT-4V generated|[ShareGPT-4V](https://sharegpt4v.github.io/)|87296|102025|-|
|Mixed|[LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main#visual-instruction-tuning)<sup>1</sup>|291684|665298|-|
|Total||974K<sup>2</sup>|11.2M|

<sup>1</sup> The bounding boxes in LLaVA-v1.5 are based on the padded image. You can find the discussion [here](https://github.com/haotian-liu/LLaVA/issues/606).

<sup>2</sup> The number of images are counted based on image IDs.
There might be duplicate images across different image sources.

We use different strategies to collect the prompts for different tasks.
- **Image captioning.** We carefully collect 5 manually written instructions and randomly sample one as the prompt for each caption. The fourth and fifth instructions are from [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md).
- **Open-ended VQA.** As the answers in VQA datasets are generally short, we add an instruction after the question to ask the model to provide answers with a short sentence or phrase.
- **Multiple-choice VQA.** For A-OKVQA, we add an instruction before the question to ask the model to provide answers with correct options. For ScienceQA, we use the instructions and templates designed by [M3IT](https://m3-it.github.io/) and randomly sample one to format the prompt. Only data with image context are involved.
- **Grounding.** For RefCOCO/RefCOCO+/RefCOCOg, we use the data and templates in [Shikra](https://github.com/shikras/shikra) and randomly sample one to format the prompt.
- **GPT-4/GPT-4V generated & mixed datasets.** We keep the prompts unchanged.

|Category|Data|Prompts|
|:-:|:-:|:-:|
|Image captioning|COCO, Flickr30K, TextCaps, TextOCR-GPT4V|Describe the image as simply as possible with a sentence or phrase.<br />Give a brief summary of what you see.<br />Provide a short description of the image.<br />Write a short description for the image.<br />Briefly describe the content of the image.|
|Open-ended VQA|VQAv2, OKVQA, OCRVQA, GQA, TextVQA, VGQA, DocVQA, DVQA|*question* Answer the question directly with a short sentence or phrase.|
|Multiple-choice VQA|A-OKVQA|Choose the correct option for the following question: *question*|

## Quickstart

For the images, you can download the images from our [HuggingFace repository](https://huggingface.co/datasets/BAAI/DataOptim/tree/main/images) or the original websites.
If you already have the images, you can skip this process as the image IDs and file names are not changed.

Then unzip and organize the images in following structure.

```
|- images
  |- coco
    |- COCO_train2014_000000000009.jpg
    |- ...
  |- coco_2017
    |- 000000274591.jpg
    |- ...
  |- docvqa
    |- ffbf0023_4.png
    |- ...
  |- dvqa
    |- ...
  |- filckr30k
    |- 36979.jpg
    |- ...
  |- llava
    |- llava_pretrain
      |- images
  |- ocrvqa
    |- 13714.jpg
    |- ...
  |- open_images
    |- 0a0bc91825468c45.jpg
    |- ...
  |- sam
    |- images
  |- scienceqa
    |- 1
      |- image.png
    |- 2
      |- image.png
    |- ...
  |- share_textvqa
    |- images
  |- visual_genome
    |- 1.jpg
    |- ...
  |- web-celebrity
    |- images
  |- web-landmark
    |- images
  |- wikiart
    |- images
```

After that, you can use this diretory as the `--image_folder` in LLaVA's training script.

For the visual instruction tuning QAs, all of the data mentioned above are already converted to the training format of LLaVA in our HuggingFace repository.
You can download them directly from [HuggingFace](https://huggingface.co/datasets/BAAI/DataOptim/tree/main/data).

For referring QAs, the bounding box is in the form of [x1, y1, x2, y2], corresponding to the top left x, top left y, bottom right x and bottom right y. The values are float numbers normalized to [0, 1], based on the size of **original images**, except LLaVA-v1.5, which is based on the **padded image** (see more discussion [here](https://github.com/haotian-liu/LLaVA/issues/606)). We provide a script [here](./tools/expand_to_square.py) to expand the bounding boxes to square.

## Contact
If you have any questions, you can open an issue in the GitHub repository or contact zhaobo@baai.ac.cn for more information.
