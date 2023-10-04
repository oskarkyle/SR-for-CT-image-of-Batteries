Usage
=====

Here is a simple guide to use this project.

Installation and depandencies
-----------------------------

This project is based on Python 3.11.
To install the depandencies, you can use the following command:

.. code-block:: console

   $ pip install -r requirements.txt


Start
-----

To start running the code, you can use the following command:

.. code-block:: console

    $ python main.py


Configuration
-------------
In this project, a configuration file is used to store the parameters of the code. 
The configuration file is a '.yaml' file, and it is located in the 'configs' folder.
The configuration file is divided into different section, and each section contains the parameters of a specific part of code.

.. autoyaml:: ../configs/config.yaml

You can change the parameters in the configuration file to change the behavior of the code.
Here is an example to change the train epoch.

.. code-block:: console
    
    $ python main.py ++train.epochs=50

This will change the train epochs from 10 to 50.