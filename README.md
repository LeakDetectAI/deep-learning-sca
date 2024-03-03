# A Python library to perform Automated Side-Channel Attacks using Black-Box Neural Architecture Search



Package
------------

The deepscapy is a python package for automatically finding optimal architectures when performing a side channel attack.
Its Ranking Loss (RKL) implementation is based on https://github.com/gabzai/Ranking-Loss-SCA.


Installation
------------
You can install deepsca using::

	python setup.py install


Dependencies
------------
deepscapy depends on NumPy, SciPy, matplotlib, scikit-learn, joblib and tqdm, tensorflow, tensorflow_addons, keras_tuner, keras, autokeras.
For data processing and generation you will also need and pandas.


License
--------
[Apache License, Version 2.0](LICENSE)


Datasets
--------
The datasets used in the paper "Automated Side-Channel Attacks using Black-Box Neural Architecture Search" can be found at https://drive.google.com/drive/folders/1GcWQvwwEdbj2L0c1hd2YpLpbS-gIFJJ5. The following table shows their properties:

| Dataset name       | # Features   | # Profiling traces   | # Attack traces   | Attack byte | URL                                                                                                                              |
|----------------------|----------------|------------------------|---------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------|
| ASCAD\_f           | 700          | 50000                | 10000             | 2           | ASCAD.h5 from https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key                      |
| ASCAD\_f desync50  | 700          | 50000                | 10000             | 2           | ASCAD_desync50.h5 from "                                                                                    |
| ASCAD\_f desync100 | 700          | 50000                | 10000             | 2           | ASCAD_desync100.h5 from "                                                                                    |
| ASCAD\_r           | 1400         | 200000               | 100000            | 2           | ASCAD.h5 from https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key/                  |
| ASCAD\_r desync50  | 1400         | 200000               | 100000            | 2           | ASCAD_desync50.h5 from "                                                                                     |
| ASCAD\_r desync100 | 1400         | 200000               | 100000            | 2           | ASCAD_desync100.h5 from "                                                                                    |
| CHES CTF           | 2200         | 45000                | 5000              | 2           | http://aisylabdatasets.ewi.tudelft.nl/ches_ctf.h5                                                                            |
| AES\_HD            | 1250         | 50000                | 25000             | 0           | https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/blob/master/AES_HD/AES_HD_dataset.zip           |
| AES\_RD            | 3500         | 25000                | 25000             | 0           | https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/tree/master/AES_RD/AES_RD_dataset               |
| DPAv4              | 4000         | 4500                 | 500               | 0           | https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/blob/master/DPA-contest%20v4/DPAv4_dataset.zip |

More datasets of hardware side-channel attacks can be found at https://github.com/ITSC-Group/sca-datasets.


Search Space
------------
The search space used in the paper "Automated Side-Channel Attacks using Black-Box Neural Architecture Search" is already defined in this project.
These are the relevant parameter ranges:

| Hyperparameter Type                         | Hyperparameter                  | Possible Options                                                                                                                                                                       |
|---------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Whole Network | Optimizer                       | \{'adam' , 'adam_with_weight_decay'\}                                                                                                                                               |
|                                             | Learning rate                   | {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5} |
| Every Layer   | Dropout                         | \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5\}                                                                                                                                                    |
|                                             | Use Batch Normalization         | \{True, False\}                                                                                                                                                                        |
|                                             | Activation Function             | \{'relu', 'selu', 'elu', 'tanh'\}                                                                                                                                                      |
| Convolutional Block                             | # of Blocks | {1, 2, 3, 4, 5}
|                                             | Convolutional Kernel Size       | \{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}                                                                                                                                         |
|                                             | Convolutional Filters           | \{2, 8, 16, 32, 64, 128, 256\}                                                                                                                                                         |
|                                             | Pooling Type                    | \{'max' , 'average'\}                                                                                                                                                                  |
|                                             | Pooling Strides 1D CNN  | \{2, 3, 4, 5, 6, 7, 8, 9, 10\}                                                                                                                                                         |
|                                             | Pooling Poolsize 1D CNN | \{2, 3, 4, 5\}                                                                                                                                                                         |
|                                             | Pooling Strides 2D CNN  | \{2, 4\}                                                                                                                                                                               |
|                                             | Pooling Poolsize 2D CNN | Convolutional Kernel Size-1                                                                                                                                                            |
| Dense Block   | # of Blocks                      | \{1, 2, 3\}                                                                                                                                                                            |
|                                             | Hidden Units                    | \{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024\}                                                                                                                                           |

