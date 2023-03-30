# Learning generative image manipulations

Master thesis project @ [Örebro University](oru.se)

Using 
- [x] python and 
- [x] pytorch

## Files:
### gimli.py 
contain the models; a generator and a discriminator. The generator is a CAE+miniBERT+RN and takes an image (3x180x120) and language encoding through a convolutional encoder, relational network and a convolutional decoder. The discriminator classifies images as real, fake or wrong. Trained in GAN settings the generator tries to convince the discriminator that its images are real.

### datasets.py 
loads the CSS3D dataset from a local directory.

### train.py 
trains the generator model.

### GAN_dataset.py 
loads the CSS3D dataset from a local directory, but also adds a 'wrong_img' on top of the 'source_img' and 'target_img', this is to match the labels for the discriminator.

### trainDCGAN.py 
trains the deep convolutional generative adversarial network comprised of the generator and discriminator models.


## --- Work in progress ---
