Data
=====

Download
------------

The data can be downloaded from the following link: 
<https://www.research-collection.ethz.ch/handle/20.500.11850/505938>. 
And the data is unzipped in the folder 'dataset/'. 
If there is no such folder, please create one and unzip the data in it.

Create the folder:

.. code-block:: console

   $ mkdir dataset

Dataset
-------

To get the dataset for model,
you can use the ``BaseDataset`` class.

.. autoclass:: source.BaseDataset.BaseDataset
    :members:
    :undoc-members:
    :show-inheritance:


Preprocessing
-------------

To get the preprocess for dataset,
you can use the ``Preprocessor`` class.

.. autoclass:: preprocess.Preprocessor.Preprocessor
    :members:
    :undoc-members:
    :show-inheritance:


Transform 
---------

To get the transform for dataset,
you can use the ``Transform`` class.

.. autoclass:: transform.Transforms.Transforms
    :members:
    :undoc-members:
    :show-inheritance: