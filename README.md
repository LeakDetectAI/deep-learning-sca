# A Python library to perform Automated Side-Channel Attacks using Black-Box Neural Architecture Search



deepscapy
------------

deepscapy is a python package for automatically finding optimal architectures when performing a side channel attack.
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
The datasets used in the paper Automated Side-Channel Attacks using Black-Box Neural Architecture Search can be found at https://drive.google.com/drive/folders/1GcWQvwwEdbj2L0c1hd2YpLpbS-gIFJJ5. The following table shows their properties:



Search Space
------------
