Model
=====

This module contains the implementation of the U-Net model.

ConvBlock
---------

In this subsection, class ``ConvBlock`` is a building block of the U-Net model.

.. autoclass:: model.unet.blocks.ConvBlock 
    :members:
    :undoc-members:
    :show-inheritance:


SkipConnection 
--------------

In this subsection, class ``SkipConnection`` builds a skip connection in the U-Net model.

.. autoclass:: model.unet.blocks.SkipConnection 
    :members:
    :undoc-members:
    :show-inheritance:


CombineConnection
-----------------

In this subsection, class ``CombineConnection`` combines skip connection in the U-Net model.

.. autoclass:: model.unet.blocks.CombineConnection 
    :members:
    :undoc-members:
    :show-inheritance:


ConvUNet
--------

In this subsection, class ``ConvUNet`` builds a U-Net model.

.. autoclass:: model.unet.ConvUNet.ConvUNet
    :members:
    :undoc-members:
    :show-inheritance: