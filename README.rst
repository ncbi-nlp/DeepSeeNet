.. image:: https://github.com/ncbi-nlp/DeepSeeNet/blob/master/images/deepseenet.png?raw=true
   :target: https://github.com/ncbi-nlp/DeepSeeNet/blob/master/images/deepseenet.png?raw=true
   :alt: DeepSeeNet

.. role:: raw-html(raw)
   :format: html

-----------------------

DeepSeeNet is a high-performance deep learning framework for grading of color fundus photographs using the AREDS simplified severity scale. For more details, please see `<https://ncbi-nlp.github.io/DeepSeeNet/>`_.

Getting Started with DeepSeeNet
===============================

These instructions will get you a copy of the project up and run on your local machine for development and testing purposes.
The package should successfully install on Linux.

Installing
----------

Prerequisites
~~~~~~~~~~~~~

*  python =3.6
*  tensorflow >=1.6.0
*  keras =2.2.4
*  Linux

Tensorflow can be downloaded from `https://www.tensorflow.org <https://www.tensorflow.org/>`_.


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

1. Download the source code from GitHub: ``git clone https://github.com/ncbi-nlp/DeepSeeNet.git``
2. Change to the directory of ``DeepSeeNet``
3. Install required packages: ``pip install -r requirements.txt``
4. Add the code directory to ``PYTHONPATH``: ``export PYTHONPATH=.:$PYTHONPATH``


Using DeepSeeNet for grading simplified scores
----------------------------------------------

The easiest way is to run the following command

.. code-block:: bash

   $ python examples/predict_simplified_score.py data/left_eye.jpg data/right_eye.jpg
   ...
   Downloading data from https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/drusen_model.h5
   INFO:root:Loading the model: /tmp/.keras/datasets/drusen_model.h5
   Downloading data from https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/pigment_model.h5
   INFO:root:Loading the model: /tmp/.keras/datasets/pigment_model.h5
   Downloading data from https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/advanced_amd_model.h5
   INFO:root:Loading the model: /tmp/.keras/datasets/advanced_amd_model.h5
   ...
   INFO:root:Processing: data/left_eye.jpg
   INFO:root:Processing: data/right_eye.jpg
   ...
   INFO:root:Risk factors: {'pigment': (0, 0), 'advanced_amd': (0, 0), 'drusen': (2, 2)}
   The simplified score: 2

The script will

1. Download the models from the ``DeepSeeNet`` repository
2. Predict the simplified score based on the sample left and right eyes

More options (e.g., setting the models) can be obtained by running

.. code-block:: bash

   $ python examples/predict_simplified_score.py --help


Pre-trained DeepSeeNet models
-----------------------------

Besides grading the simplified score, we also provide individual risk factor models. For example

.. code-block:: bash

   $ python examples/predict_drusen.py data/left_eye.jpg
   ...
   INFO:root:Loading the model: /tmp/.keras/datasets/drusen_model.h5
   ...
   INFO:root:Processing: data/left_eye.jpg
   ...
   The drusen score: [[0.21020733 0.2953384  0.49445423]]
   The drusen size: large

Here, we provide the following pre-trained models:

*  `drusen size <https://github.com/ncbi-nlp/DeepSeeNet/releases/tag/0.1>`_: non/small, intermediate, large
*  `pigmentary abnormalities <https://github.com/ncbi-nlp/DeepSeeNet/releases/tag/0.1>`_: no, yes
*  `late AMD <https://github.com/ncbi-nlp/DeepSeeNet/releases/tag/0.1>`_: no, yes
*  `geographic atrophy (GA) <https://github.com/ncbi-nlp/DeepSeeNet/releases/tag/0.2>`_: no, yes
*  `central GA <https://github.com/ncbi-nlp/DeepSeeNet/releases/tag/0.2>`_: no, yes


Training DeepSeeNet model
-------------------------

You can train the individual risk factor model too. For example

.. code-block:: bash

   $ python examples/train.py data/pigment_label_sample.csv data/pigment_best_model.h5
   ...
   Epoch 1/100
   2/2 [==============================] - 27s 14s/step - loss: 1.0103 - acc: 0.5148...
   ...
   early stopping


The program will read images and labels from a CSV file, train the model, and save the latest best model according to the ``val_acc``.


Acknowledgments
===============

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine and National Eye Institute.


Citing DeepSeeNet
=================

If you're running the DeepSeeNet framework, please cite:

*  Peng Y*, Dharssi S*, Chen Q, Keenan T, Agron E, Wong W, Chew E, Lu Z. DeepSeeNet: A deep learning model for automated classification of patientbased age-related macular degeneration severity from color fundus photographs. Ophthalmology. 2019. 126(4), 565-575.

*  Keenan T*, Dharssi S*, Peng Y*, Chen Q, Agron E, Wong W, Lu Z, Chew E. A deep learning approach for automated detection of geographic atrophy from color fundus photographs. Ophthalmology. 2019 (Accepted).


NCBI's Disclaimer
=================

This tool shows the results of research conducted in the `Computational Biology Branch <https://www.ncbi.nlm.nih.gov/research/>`_, `NCBI <https://www.ncbi.nlm.nih.gov/home/about>`_. 

The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. 

More information about `NCBI's disclaimer policy <https://www.ncbi.nlm.nih.gov/home/about/policies.shtml>`_ is available.

About `text mining group <https://www.ncbi.nlm.nih.gov/research/bionlp/>`_.

For Research Use Only
=====================

The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only. 


