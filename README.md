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
## Approach
![Approach](/images/approach.png)

Our approach uses black-box neural architecture search (NAS) to automatically identify optimal deep learning models for side-channel attacks.

Starting from a profiling dataset, the method splits data into training and validation sets. Different neural architectures from a predefined search space are iteratively trained and evaluated using a search strategy. Each candidate model is assessed based on its validation performance (e.g., accuracy or ranking loss).

The best-performing architecture is then selected and retrained on the full profiling dataset to obtain the final model.

Finally, the trained model is applied to the attack dataset, where performance is measured using **Guessing Entropy (GE)**, resulting in a **vulnerability score (VS)** that quantifies the leakage of the target system.

## Per-Dataset Performance

![Per-Dataset Performance](/images/performance.png)

This figure compares vulnerability scores across multiple standard side-channel datasets under two settings:

- **All parameter combinations** (left): exhaustive evaluation over the defined search space  
- **1D CNNs with Random Search** (right): architectures automatically selected via neural architecture search  

### Key Observations

- **Higher vulnerability scores** indicate more successful key recovery (i.e., stronger side-channel leakage exploitation)
- NAS-based models (right) consistently achieve **higher performance across datasets**
- Even under **desynchronization (ASCAD desync50 / desync100)**, the models remain effective
- Systems with **no countermeasures** show near-perfect vulnerability (close to 100%)
- Systems with **countermeasures** exhibit reduced but still exploitable leakage

### Interpretation

This demonstrates that automated architecture search:
- Adapts effectively to different leakage characteristics  
- Outperforms manual or exhaustive configurations  
- Provides a scalable method for **hardware side-channel evaluation**

**4 systems include countermeasures, 6 systems do not.**


## 💬 Cite Us

If you use this toolkit in your research, please cite our work:

- 📄 **Conference Paper (ARES 2023)**  
- 📘 **PhD Dissertation (2025)** for a more comprehensive understanding  

### BibTeX

```bibtex
@inproceedings{gupta2023hwsca,
  author    = {Pritha Gupta and Jan Peter Drees and Eyke H{\"u}llermeier},
  title     = {Automated Side-Channel Attacks using Black-Box Neural Architecture Search},
  year      = {2023},
  isbn      = {9798400707728},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3600160.3600161},
  doi       = {10.1145/3600160.3600161},
  booktitle = {Proceedings of the 18th International Conference on Availability, Reliability and Security},
  articleno = {5},
  numpages  = {11},
  location  = {Benevento, Italy},
  series    = {ARES '23}
}
```
## 📧 Contact Information
For any questions or feedback, please contact Pritha Gupta at prithagupta.nsit@icloud.com.
