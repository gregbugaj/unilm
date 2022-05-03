# layoutlmft
**Multimodal (text + layout/format + image) fine-tuning toolkit for document understanding**

## Introduction

## Supported Models
Popular Language Models: BERT, UniLM(v2), RoBERTa, InfoXLM

LayoutLM Family: LayoutLM, LayoutLMv2, LayoutXLM

## Installation

~~~bash
conda create -n layoutlmft python=3.7
conda activate layoutlmft
git clone https://github.com/microsoft/unilm.git
cd unilm
cd layoutlmft
pip install -r requirements.txt
pip install -e .
~~~

## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using layoutlmft, please submit a GitHub issue.

For other communications related to layoutlmft, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).


```

python ./run_funsd.py --do_eval=True --do_predict=True --do_train=True --evaluation_strategy=epoch --fp16=True --load_best_model_at_end=True --max_train_samples=1000 --model_name_or_path=microsoft/layoutlmv2-base-uncased --num_train_epochs=30 --output_dir=/tmp/test-ner --overwrite_output_dir=True --report_to=wandb --save_strategy=epoch --save_total_limit=1 --warmup_ratio=0.1


python ./run_funsd.py --do_eval=True --do_predict=True --do_train=True --evaluation_strategy=epoch --fp16=True --load_best_model_at_end=True --max_train_samples=1000 --model_name_or_path=microsoft/layoutlmv2-base-uncased --num_train_epochs=30 --output_dir=/tmp/test-ner --overwrite_output_dir=True --report_to=wandb --save_strategy=epoch --save_total_limit=1 --warmup_ratio=0.1

```