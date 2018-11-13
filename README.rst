.. image:: https://github.com/ncbi-nlp/DeepSeeNet/blob/master/images/deepseenet.png?raw=true
   :target: https://github.com/ncbi-nlp/DeepSeeNet/blob/master/images/deepseenet.png?raw=true
   :alt: DeepSeeNet


-----------------------

DeepSeeNet is a high-performance deep learning framework for grading of color fundus photographs using the AREDS simplified severity scale. For more details, please see `<https://ncbi-nlp.github.io/DeepSeeNet/>`_.


Getting Started with DeepSeeNet
============================

These instructions will get you a copy of the project up and run on your local machine for development and testing purposes.
The package should successfully install on Linux.

Installing
----------

Prerequisites
~~~~~~~~~~~~~

*  python =3.6
*  tensorflow >=1.6.0
*  keras =2.1.5
*  Linux

Tensorflow can be downloaded from `https://www.tensorflow.org <https://www.tensorflow.org/>`_.


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

1. Download the source code from GitHub: ``git clone https://github.com/ncbi-nlp/DeepSeeNet.git``
2. Change to the directory of ``DeepSeeNet``
3. Install required packages: ``pip install -r requirements.txt``
4. Add the code directory to ``PYTHONPATH``: ``export PYTHONPATH=.:$PYTHONPATH``


Using DeepSeeNet for grading simplified scores
-------------------------------------------

The easiest way is to run the following command

.. code-block:: bash

   $ python examples/predict_simplified_score.py data/left_eye.jpg data/right_eye.jpg
   ...
   INFO:root:Loading the model: drusen
   INFO:root:Loading the model: advanced_amd
   INFO:root:Loading the model: pigment
   ...
   INFO:root:Processing: data/left_eye.jpg
   INFO:root:Processing: data/right_eye.jpg
   ...
   The simplified score: 2

The script will

1. Download the models from the ``DeepSeeNet`` repository
2. Predict the simplified score based on the sample left and right eyes

More options (e.g., setting the models) can be obtained by running

.. code-block:: bash

   $ python examples/predict_simplified_score.py --help


Pretrained DeepSeeNet models
----------------------------

Besides grading the simplified score, we also provide individual risk factor models. For example

.. code-block:: bash

   $ python examples/predict_drusen.py data/left_eye.jpg
   ...
   INFO:root:Loading the model: drusen
   ...
   INFO:root:Processing: data/left_eye.jpg
   ...
   Drusen size: large


All models can be found at ``deepseenet``.

The pretrained models can be found at: 


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

*  Peng Y, Dharssi S, Chen Q, Keenan T, Agron E, Wong W, Chew E, Lu Z. DeepSeeNet: A deep learning model for automated classification of patientbased age-related macular degeneration severity from color fundus photographs. Ophthalmology. 2018 (Accepted).


Disclaimer
==========

This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.



