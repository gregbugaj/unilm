# Issues

##  

No huggingface_hub attribute hf_api

```
pip install -U datasets huggingface_hub
```

```
pip install -U tensorboard
```


## Training 
Launch traing

```
python -m torch.distributed.launch --nproc_per_node=1 ./train.py
``` 



```
tensorboard --logdir=./logs
```



LARGE

***** eval metrics *****
  epoch                               =       7.38
  eval_DOS_ANSWER_f1                  =        1.0
  eval_DOS_ANSWER_number              =         20
  eval_DOS_ANSWER_precision           =        1.0
  eval_DOS_ANSWER_recall              =        1.0
  eval_DOS_f1                         =        1.0
  eval_DOS_number                     =         20
  eval_DOS_precision                  =        1.0
  eval_DOS_recall                     =        1.0
  eval_MEMBER_NAME_ANSWER_f1          =     0.9143
  eval_MEMBER_NAME_ANSWER_number      =         18
  eval_MEMBER_NAME_ANSWER_precision   =     0.9412
  eval_MEMBER_NAME_ANSWER_recall      =     0.8889
  eval_MEMBER_NAME_f1                 =     0.9091
  eval_MEMBER_NAME_number             =         18
  eval_MEMBER_NAME_precision          =        1.0
  eval_MEMBER_NAME_recall             =     0.8333
  eval_MEMBER_NUMBER_ANSWER_f1        =     0.9667
  eval_MEMBER_NUMBER_ANSWER_number    =         30
  eval_MEMBER_NUMBER_ANSWER_precision =     0.9667
  eval_MEMBER_NUMBER_ANSWER_recall    =     0.9667
  eval_MEMBER_NUMBER_f1               =     0.9508
  eval_MEMBER_NUMBER_number           =         30
  eval_MEMBER_NUMBER_precision        =     0.9355
  eval_MEMBER_NUMBER_recall           =     0.9667
  eval_PAN_ANSWER_f1                  =     0.9231
  eval_PAN_ANSWER_number              =          6
  eval_PAN_ANSWER_precision           =     0.8571
  eval_PAN_ANSWER_recall              =        1.0
  eval_PAN_f1                         =     0.9231
  eval_PAN_number                     =          6
  eval_PAN_precision                  =     0.8571
  eval_PAN_recall                     =        1.0
  eval_PATIENT_NAME_ANSWER_f1         =     0.9524
  eval_PATIENT_NAME_ANSWER_number     =         20
  eval_PATIENT_NAME_ANSWER_precision  =     0.9091
  eval_PATIENT_NAME_ANSWER_recall     =        1.0
  eval_PATIENT_NAME_f1                =     0.9756
  eval_PATIENT_NAME_number            =         20
  eval_PATIENT_NAME_precision         =     0.9524
  eval_PATIENT_NAME_recall            =        1.0
  eval_loss                           =     0.0086
  eval_overall_accuracy               =     0.9986
  eval_overall_f1                     =     0.9577
  eval_overall_precision              =     0.9526
  eval_overall_recall                 =     0.9628
  eval_runtime                        = 0:00:02.85
  eval_samples                        =         60
  eval_samples_per_second             =     21.046
  eval_steps_per_second               =      2.806



8  conda activate layoutlmv3
 1969  pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
 1970  pip uninstall torch
 1971  pip uninstall torchvision
 1972  clear
 1973  pip install -r requirements.txt
 1974  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
 1975  python -c "import torch; print(torch.__version__)"
 1976  python -c "import torch; print(torch.__version__)"pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
 1977  pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
 1978  python -c "import torch; print(torch.__version__)"
 1979  conda install pytorch torchvision -c pytorch
 1980  python -c "import torch; print(torch.__version__)"
 1981  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
 1982  pip install -e .
 1983  python -m pip install --upgrade pip
 1984  pip install -e .python -m pip install --upgrade pip
 1985  python -m pip install --upgrade pip
 1986  pip install -e .
 1987  ls
 1988  cat train.sh 
 1989  ls
 1990  cat train.sh 
 1991  ./train.sh 
 1992  pip install datasets
 1993  ./train.sh 
 1994  conda deactivate
 1995  rm -rf ~/anaconda3/
 1996  ls /tmp/Anaconda3-2022.05-Linux-x86_64.sh 
 1997  /tmp/Anaconda3-2022.05-Linux-x86_64.sh 
 1998  conda create --name layoutlmv3 python=3.7
 1999  conda activate layoutlmv3
 2000  pip install -r requirements.txt




100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 386/386 [00:48<00:00,  7.92it/s]
***** eval metrics *****
  epoch                               =      11.48
  eval_ADDRESS_f1                     =     0.9496
  eval_ADDRESS_number                 =       7502
  eval_ADDRESS_precision              =      0.938
  eval_ADDRESS_recall                 =     0.9615
  eval_ANSWER_f1                      =     0.8801
  eval_ANSWER_number                  =      19776
  eval_ANSWER_precision               =     0.8518
  eval_ANSWER_recall                  =     0.9103
  eval_BILLED_AMT_ANSWER_f1           =     0.7644
  eval_BILLED_AMT_ANSWER_number       =        500
  eval_BILLED_AMT_ANSWER_precision    =     0.6944
  eval_BILLED_AMT_ANSWER_recall       =       0.85
  eval_BILLED_AMT_f1                  =     0.6778
  eval_BILLED_AMT_number              =        400
  eval_BILLED_AMT_precision           =     0.6506
  eval_BILLED_AMT_recall              =     0.7075
  eval_BIRTHDATE_ANSWER_f1            =     0.9862
  eval_BIRTHDATE_ANSWER_number        =       1650
  eval_BIRTHDATE_ANSWER_precision     =     0.9994
  eval_BIRTHDATE_ANSWER_recall        =     0.9733
  eval_BIRTHDATE_f1                   =     0.9994
  eval_BIRTHDATE_number               =       1650
  eval_BIRTHDATE_precision            =     0.9988
  eval_BIRTHDATE_recall               =        1.0
  eval_CLAIM_NUMBER_ANSWER_f1         =     0.9513
  eval_CLAIM_NUMBER_ANSWER_number     =        600
  eval_CLAIM_NUMBER_ANSWER_precision  =     0.9945
  eval_CLAIM_NUMBER_ANSWER_recall     =     0.9117
  eval_CLAIM_NUMBER_f1                =     0.9991
  eval_CLAIM_NUMBER_number            =        550
  eval_CLAIM_NUMBER_precision         =     0.9982
  eval_CLAIM_NUMBER_recall            =        1.0
  eval_DOCUMENT_CONTROL_f1            =     0.6623
  eval_DOCUMENT_CONTROL_number        =       3878
  eval_DOCUMENT_CONTROL_precision     =     0.6324
  eval_DOCUMENT_CONTROL_recall        =     0.6952
  eval_DOS_ANSWER_f1                  =     0.9603
  eval_DOS_ANSWER_number              =       3100
  eval_DOS_ANSWER_precision           =     0.9431
  eval_DOS_ANSWER_recall              =     0.9781
  eval_DOS_f1                         =      0.949
  eval_DOS_number                     =       3050
  eval_DOS_precision                  =     0.9095
  eval_DOS_recall                     =     0.9921
  eval_GREETING_f1                    =     0.9067
  eval_GREETING_number                =       4965
  eval_GREETING_precision             =     0.9156
  eval_GREETING_recall                =     0.8979
  eval_HEADER_f1                      =     0.8262
  eval_HEADER_number                  =       8920
  eval_HEADER_precision               =     0.8006
  eval_HEADER_recall                  =     0.8535
  eval_LETTER_DATE_f1                 =     0.9872
  eval_LETTER_DATE_number             =       3750
  eval_LETTER_DATE_precision          =     0.9805
  eval_LETTER_DATE_recall             =     0.9939
  eval_MEMBER_NAME_ANSWER_f1          =     0.9363
  eval_MEMBER_NAME_ANSWER_number      =       2100
  eval_MEMBER_NAME_ANSWER_precision   =     0.9086
  eval_MEMBER_NAME_ANSWER_recall      =     0.9657
  eval_MEMBER_NAME_f1                 =     0.9661
  eval_MEMBER_NAME_number             =       2100
  eval_MEMBER_NAME_precision          =     0.9837
  eval_MEMBER_NAME_recall             =      0.949
  eval_MEMBER_NUMBER_ANSWER_f1        =     0.9658
  eval_MEMBER_NUMBER_ANSWER_number    =       3050
  eval_MEMBER_NUMBER_ANSWER_precision =     0.9513
  eval_MEMBER_NUMBER_ANSWER_recall    =     0.9807
  eval_MEMBER_NUMBER_f1               =     0.9667
  eval_MEMBER_NUMBER_number           =       3050
  eval_MEMBER_NUMBER_precision        =     0.9503
  eval_MEMBER_NUMBER_recall           =     0.9836
  eval_PAID_AMT_ANSWER_f1             =        0.0
  eval_PAID_AMT_ANSWER_number         =          0
  eval_PAID_AMT_ANSWER_precision      =        0.0
  eval_PAID_AMT_ANSWER_recall         =        0.0
  eval_PAID_AMT_f1                    =        0.0
  eval_PAID_AMT_number                =          0
  eval_PAID_AMT_precision             =        0.0
  eval_PAID_AMT_recall                =        0.0
  eval_PAN_ANSWER_f1                  =     0.8759
  eval_PAN_ANSWER_number              =       2650
  eval_PAN_ANSWER_precision           =      0.839
  eval_PAN_ANSWER_recall              =     0.9162
  eval_PAN_f1                         =     0.8291
  eval_PAN_number                     =       2550
  eval_PAN_precision                  =     0.7892
  eval_PAN_recall                     =     0.8733
  eval_PARAGRAPH_f1                   =     0.8474
  eval_PARAGRAPH_number               =      23912
  eval_PARAGRAPH_precision            =     0.8219
  eval_PARAGRAPH_recall               =     0.8745
  eval_PATIENT_NAME_ANSWER_f1         =     0.9426
  eval_PATIENT_NAME_ANSWER_number     =       2300
  eval_PATIENT_NAME_ANSWER_precision  =     0.9327
  eval_PATIENT_NAME_ANSWER_recall     =     0.9526
  eval_PATIENT_NAME_f1                =     0.9738
  eval_PATIENT_NAME_number            =       2100
  eval_PATIENT_NAME_precision         =     0.9489
  eval_PATIENT_NAME_recall            =        1.0
  eval_PHONE_f1                       =      0.742
  eval_PHONE_number                   =        650
  eval_PHONE_precision                =     0.5898
  eval_PHONE_recall                   =        1.0
  eval_QUESTION_f1                    =     0.8559
  eval_QUESTION_number                =      16860
  eval_QUESTION_precision             =     0.8285
  eval_QUESTION_recall                =     0.8852
  eval_URL_f1                         =     0.9992
  eval_URL_number                     =       1950
  eval_URL_precision                  =     0.9985
  eval_URL_recall                     =        1.0
  eval_loss                           =     0.7763
  eval_overall_accuracy               =     0.9337
  eval_overall_f1                     =     0.8827
  eval_overall_precision              =     0.8585
  eval_overall_recall                 =     0.9082
  eval_runtime                        = 0:00:49.97
  eval_samples                        =       6163
  eval_samples_per_second             =    123.326
  eval_steps_per_second               =      7.724



# Reference 

[Max_seq_length LayoutLMV3 - How to implement it ? #942](https://github.com/microsoft/unilm/issues/942)
 
[LayoutLMV3 Training with Morethan 512 tokens. #19190](https://github.com/huggingface/transformers/issues/19190#issuecomment-1441883471)