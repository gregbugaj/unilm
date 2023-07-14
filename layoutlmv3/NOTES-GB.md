layoutlmv3) gbugaj@asp-gpu001:~/dev/unilm/layoutlmv3/examples$ python train.py 
cuda:0
4.21.0.dev0
Downloading and preparing dataset funsd_dataset/funsd to /data/cache/funsd_dataset/funsd/2.0.0/0ce7794b21add0bd05cc8705432ceb5b37431709550c60c347c60ffd77545c73...
Generating train split: 0 examples [00:00, ? examples/s]Total files: 166000
Generating test split: 0 examples [00:00, ? examples/s]Total files: 41499
Dataset funsd_dataset downloaded and prepared to /data/cache/funsd_dataset/funsd/2.0.0/0ce7794b21add0bd05cc8705432ceb5b37431709550c60c347c60ffd77545c73. Subsequent calls will reuse this data.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 38.15it/s]
Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None)
NER-Labels -> 
['O', 'B-MEMBER_NAME', 'I-MEMBER_NAME', 'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER', 'B-PAN', 'I-PAN', 'B-PATIENT_NAME', 'I-PATIENT_NAME', 'B-DOS', 'I-DOS', 'B-DOS_ANSWER', 'I-DOS_ANSWER', 'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER', 'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER', 'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER', 'B-PAN_ANSWER', 'I-PAN_ANSWER', 'B-ADDRESS', 'I-ADDRESS', 'B-GREETING', 'I-GREETING', 'B-HEADER', 'I-HEADER', 'B-LETTER_DATE', 'I-LETTER_DATE', 'B-PARAGRAPH', 'I-PARAGRAPH', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER', 'B-DOCUMENT_CONTROL', 'I-DOCUMENT_CONTROL', 'B-PHONE', 'I-PHONE', 'B-URL', 'I-URL', 'B-CLAIM_NUMBER', 'I-CLAIM_NUMBER', 'B-CLAIM_NUMBER_ANSWER', 'I-CLAIM_NUMBER_ANSWER', 'B-BIRTHDATE', 'I-BIRTHDATE', 'B-BIRTHDATE_ANSWER', 'I-BIRTHDATE_ANSWER', 'B-BILLED_AMT', 'I-BILLED_AMT', 'B-BILLED_AMT_ANSWER', 'I-BILLED_AMT_ANSWER', 'B-PAID_AMT', 'I-PAID_AMT', 'B-PAID_AMT_ANSWER', 'I-PAID_AMT_ANSWER', 'B-CHECK_AMT', 'I-CHECK_AMT', 'B-CHECK_AMT_ANSWER', 'I-CHECK_AMT_ANSWER', 'B-CHECK_NUMBER', 'I-CHECK_NUMBER', 'B-CHECK_NUMBER_ANSWER', 'I-CHECK_NUMBER_ANSWER', 'B-LIST', 'I-LIST', 'B-FOOTER', 'I-FOOTER', 'B-DATE', 'I-DATE', 'B-IDENTIFIER', 'I-IDENTIFIER', 'B-PROC_CODE', 'I-PROC_CODE', 'B-PROC_CODE_ANSWER', 'I-PROC_CODE_ANSWER', 'B-PROVIDER', 'I-PROVIDER', 'B-PROVIDER_ANSWER', 'I-PROVIDER_ANSWER', 'B-MONEY', 'I-MONEY', 'B-COMPANY', 'I-COMPANY', 'B-STAMP', 'I-STAMP']
ID2Label : 
{0: 'O', 1: 'B-MEMBER_NAME', 2: 'I-MEMBER_NAME', 3: 'B-MEMBER_NUMBER', 4: 'I-MEMBER_NUMBER', 5: 'B-PAN', 6: 'I-PAN', 7: 'B-PATIENT_NAME', 8: 'I-PATIENT_NAME', 9: 'B-DOS', 10: 'I-DOS', 11: 'B-DOS_ANSWER', 12: 'I-DOS_ANSWER', 13: 'B-PATIENT_NAME_ANSWER', 14: 'I-PATIENT_NAME_ANSWER', 15: 'B-MEMBER_NAME_ANSWER', 16: 'I-MEMBER_NAME_ANSWER', 17: 'B-MEMBER_NUMBER_ANSWER', 18: 'I-MEMBER_NUMBER_ANSWER', 19: 'B-PAN_ANSWER', 20: 'I-PAN_ANSWER', 21: 'B-ADDRESS', 22: 'I-ADDRESS', 23: 'B-GREETING', 24: 'I-GREETING', 25: 'B-HEADER', 26: 'I-HEADER', 27: 'B-LETTER_DATE', 28: 'I-LETTER_DATE', 29: 'B-PARAGRAPH', 30: 'I-PARAGRAPH', 31: 'B-QUESTION', 32: 'I-QUESTION', 33: 'B-ANSWER', 34: 'I-ANSWER', 35: 'B-DOCUMENT_CONTROL', 36: 'I-DOCUMENT_CONTROL', 37: 'B-PHONE', 38: 'I-PHONE', 39: 'B-URL', 40: 'I-URL', 41: 'B-CLAIM_NUMBER', 42: 'I-CLAIM_NUMBER', 43: 'B-CLAIM_NUMBER_ANSWER', 44: 'I-CLAIM_NUMBER_ANSWER', 45: 'B-BIRTHDATE', 46: 'I-BIRTHDATE', 47: 'B-BIRTHDATE_ANSWER', 48: 'I-BIRTHDATE_ANSWER', 49: 'B-BILLED_AMT', 50: 'I-BILLED_AMT', 51: 'B-BILLED_AMT_ANSWER', 52: 'I-BILLED_AMT_ANSWER', 53: 'B-PAID_AMT', 54: 'I-PAID_AMT', 55: 'B-PAID_AMT_ANSWER', 56: 'I-PAID_AMT_ANSWER', 57: 'B-CHECK_AMT', 58: 'I-CHECK_AMT', 59: 'B-CHECK_AMT_ANSWER', 60: 'I-CHECK_AMT_ANSWER', 61: 'B-CHECK_NUMBER', 62: 'I-CHECK_NUMBER', 63: 'B-CHECK_NUMBER_ANSWER', 64: 'I-CHECK_NUMBER_ANSWER', 65: 'B-LIST', 66: 'I-LIST', 67: 'B-FOOTER', 68: 'I-FOOTER', 69: 'B-DATE', 70: 'I-DATE', 71: 'B-IDENTIFIER', 72: 'I-IDENTIFIER', 73: 'B-PROC_CODE', 74: 'I-PROC_CODE', 75: 'B-PROC_CODE_ANSWER', 76: 'I-PROC_CODE_ANSWER', 77: 'B-PROVIDER', 78: 'I-PROVIDER', 79: 'B-PROVIDER_ANSWER', 80: 'I-PROVIDER_ANSWER', 81: 'B-MONEY', 82: 'I-MONEY', 83: 'B-COMPANY', 84: 'I-COMPANY', 85: 'B-STAMP', 86: 'I-STAMP'}
{'O': 0, 'B-MEMBER_NAME': 1, 'I-MEMBER_NAME': 2, 'B-MEMBER_NUMBER': 3, 'I-MEMBER_NUMBER': 4, 'B-PAN': 5, 'I-PAN': 6, 'B-PATIENT_NAME': 7, 'I-PATIENT_NAME': 8, 'B-DOS': 9, 'I-DOS': 10, 'B-DOS_ANSWER': 11, 'I-DOS_ANSWER': 12, 'B-PATIENT_NAME_ANSWER': 13, 'I-PATIENT_NAME_ANSWER': 14, 'B-MEMBER_NAME_ANSWER': 15, 'I-MEMBER_NAME_ANSWER': 16, 'B-MEMBER_NUMBER_ANSWER': 17, 'I-MEMBER_NUMBER_ANSWER': 18, 'B-PAN_ANSWER': 19, 'I-PAN_ANSWER': 20, 'B-ADDRESS': 21, 'I-ADDRESS': 22, 'B-GREETING': 23, 'I-GREETING': 24, 'B-HEADER': 25, 'I-HEADER': 26, 'B-LETTER_DATE': 27, 'I-LETTER_DATE': 28, 'B-PARAGRAPH': 29, 'I-PARAGRAPH': 30, 'B-QUESTION': 31, 'I-QUESTION': 32, 'B-ANSWER': 33, 'I-ANSWER': 34, 'B-DOCUMENT_CONTROL': 35, 'I-DOCUMENT_CONTROL': 36, 'B-PHONE': 37, 'I-PHONE': 38, 'B-URL': 39, 'I-URL': 40, 'B-CLAIM_NUMBER': 41, 'I-CLAIM_NUMBER': 42, 'B-CLAIM_NUMBER_ANSWER': 43, 'I-CLAIM_NUMBER_ANSWER': 44, 'B-BIRTHDATE': 45, 'I-BIRTHDATE': 46, 'B-BIRTHDATE_ANSWER': 47, 'I-BIRTHDATE_ANSWER': 48, 'B-BILLED_AMT': 49, 'I-BILLED_AMT': 50, 'B-BILLED_AMT_ANSWER': 51, 'I-BILLED_AMT_ANSWER': 52, 'B-PAID_AMT': 53, 'I-PAID_AMT': 54, 'B-PAID_AMT_ANSWER': 55, 'I-PAID_AMT_ANSWER': 56, 'B-CHECK_AMT': 57, 'I-CHECK_AMT': 58, 'B-CHECK_AMT_ANSWER': 59, 'I-CHECK_AMT_ANSWER': 60, 'B-CHECK_NUMBER': 61, 'I-CHECK_NUMBER': 62, 'B-CHECK_NUMBER_ANSWER': 63, 'I-CHECK_NUMBER_ANSWER': 64, 'B-LIST': 65, 'I-LIST': 66, 'B-FOOTER': 67, 'I-FOOTER': 68, 'B-DATE': 69, 'I-DATE': 70, 'B-IDENTIFIER': 71, 'I-IDENTIFIER': 72, 'B-PROC_CODE': 73, 'I-PROC_CODE': 74, 'B-PROC_CODE_ANSWER': 75, 'I-PROC_CODE_ANSWER': 76, 'B-PROVIDER': 77, 'I-PROVIDER': 78, 'B-PROVIDER_ANSWER': 79, 'I-PROVIDER_ANSWER': 80, 'B-MONEY': 81, 'I-MONEY': 82, 'B-COMPANY': 83, 'I-COMPANY': 84, 'B-STAMP': 85, 'I-STAMP': 86}
#2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [17:05<00:00, 24.43s/ba]
#1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [17:06<00:00, 24.43s/ba]
#0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [17:35<00:00, 25.14s/ba]
#3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [18:07<00:00, 25.88s/ba]
#1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [04:20<00:00, 23.65s/ba]
#3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [04:20<00:00, 23.71s/ba]
#0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [04:21<00:00, 23.73s/ba]
#2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [04:42<00:00, 25.67s/ba]
pixel_values torch.Size([3, 224, 224])█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [04:20<00:00, 20.12s/ba]
input_ids torch.Size([512])████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [04:42<00:00, 21.50s/ba]
attention_mask torch.Size([512])
bbox torch.Size([512, 4])
labels torch.Size([512])
Some weights of LayoutLMv3ForTokenClassification were not initialized from the model checkpoint at microsoft/layoutlmv3-large and are newly initialized: ['classifier.out_proj.bias', 'classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
training_args *************************
TrainingArguments(
_n_gpu=2,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_pin_memory=True,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=False,
do_train=True,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=1000,
evaluation_strategy=steps,
fp16=True,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
greater_is_better=True,
group_by_length=False,
half_precision_backend=auto,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
load_best_model_at_end=True,
local_rank=-1,
log_level=-1,
log_level_replica=-1,
log_on_each_node=True,
logging_dir=/data/models/layoutlmv3-large-20230711-stride128/runs/Jul12_13-19-37_asp-gpu001,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=steps,
lr_scheduler_type=linear,
max_grad_norm=1.0,
max_steps=25000,
metric_for_best_model=eval_overall_f1,
mp_parameters=,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_hf,
output_dir=/data/models/layoutlmv3-large-20230711-stride128,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=6,
per_device_train_batch_size=10,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['tensorboard'],
resume_from_checkpoint=None,
run_name=/data/models/layoutlmv3-large-20230711-stride128,
save_on_each_node=False,
save_steps=1000,
save_strategy=steps,
save_total_limit=None,
seed=42,
sharded_ddp=[],
skip_memory_metrics=True,
tf32=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_ipex=False,
use_legacy_prediction_loop=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
xpu_backend=None,
)
max_steps is given, it will override any value given in num_train_epochs
Using cuda_amp half precision backend
***** Running training *****
  Num examples = 262894
  Num Epochs = 2
  Instantaneous batch size per device = 10
  Total train batch size (w. parallel, distributed & accumulation) = 20
  Gradient Accumulation steps = 1
  Total optimization steps = 25000
{'loss': 0.3014, 'learning_rate': 4.9002e-05, 'epoch': 0.04}                                                                                                                                                       
{'loss': 0.0555, 'learning_rate': 4.8008000000000005e-05, 'epoch': 0.08}                                                                                                                                           
  4%|██████▋                                                                                                                                                                | 1000/25000 [22:37<8:59:14,  1.35s/it]***** Running Evaluation *****
  Num examples = 65965
  Batch size = 12
{'eval_loss': 0.011304712854325771, 'eval_precision': 0.9891959117452876, 'eval_recall': 0.9915531398054118, 'eval_f1': 0.9903731231433194, 'eval_accuracy': 0.9967911060110995, 'eval_runtime': 3372.7748, 'eval_samples_per_second': 19.558, 'eval_steps_per_second': 1.63, 'epoch': 0.08}                                                                                                                                          
  4%|██████▌                                                                                                                                                              | 1000/25000 [1:18:50<8:59:14,  1.35s/itSaving model checkpoint to /data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000                                                                                                                         
Configuration saved in /data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000/config.json
Model weights saved in /data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000/pytorch_model.bin
tokenizer config file saved in /data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000/tokenizer_config.json
Special tokens file saved in /data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000/special_tokens_map.json
Traceback (most recent call last):
  File "/home/gbugaj/dev/unilm/layoutlmv3/examples/train.py", line 273, in <module>
    train_result = trainer.train()
  File "/home/gbugaj/dev/transformers/src/transformers/trainer.py", line 1474, in train
    return inner_training_loop(
  File "/home/gbugaj/dev/transformers/src/transformers/trainer.py", line 1793, in _inner_training_loop
    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
  File "/home/gbugaj/dev/transformers/src/transformers/trainer.py", line 1981, in _maybe_log_save_evaluate
    self._save_checkpoint(model, trial, metrics=metrics)
  File "/home/gbugaj/dev/transformers/src/transformers/trainer.py", line 2099, in _save_checkpoint
    metric_value = metrics[metric_to_check]
KeyError: 'eval_overall_f1'
