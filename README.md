# Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank (2023 CVPR)
Shirui Huang*, Keyan Wang*+, Huan Liu, Jun Chen, Yunsong Li

*Equal Contributions
+Corresponding Author

Xidian University, McMaster University
## Introduction
This is the official repository for our recent paper, "Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank", where more implementation details are presented.


### Dependencies

- Ubuntu==18.04
- Pytorch==1.8.1
- CUDA==11.1

Other dependencies are listed in `requirements.txt`

### Prepare Data

Run `data_split.py` to randomly split your paired datasets into training, validation and testing set.

Run `estimate_illumination.py` to get illumination map of the corresponding image.

Finally, the structure of  `data`  are aligned as follows:

```
data
├── labeled
│   ├── input
│   └── GT
│   └── LA
├── unlabeled
│   ├── input
│   └── LA
│   └── candidate
└── val
    ├── input
    └── GT
    └── LA
└── test
    ├── benchmarkA
        ├── input
        └── LA
```

You can download the training set and test sets for our paper [here](). 

### Test

Put your test benchmark under `data/test` folder, run `estimate_illumination.py` to get its illumination map.

Run `test.py` and you can find results from folder `result`.

### Train

To train the framework, run `create_candiate.py` to initialize reliable bank. Hyper-parameters can be modified in `trainer.py`.

Run `train.py` to start training.

### Citation


### Contact

If you have any problem with the released code, please do not hesitate to contact us by email (shiruihh@gmail.com).
