# Learning generative image manipulations

Master thesis project @ [Örebro University](https://www.oru.se)

Using 
- [x] python and 
- [x] pytorch

## Instructions
**Dowload dataset** [External website to CSS3D dataset](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view) 

Make sure the python files are in the same directory, preferably in a subdirectory to the dataset (otherwise, specify path to dataset in command line);
  - working directory (dir)
    - scripts (dir)
        - '...'.py
        - '...'.py
        - '...'.py
    - images (dir)
        - css_train_******.png
        - css_test_******.png
    - css-actions.dup.npy (file)

**Train generator**

Run the training loop with arguments
  `python train.py`
to train gimli_v2 with default arguments.

Available arguments:
- `--batch-size=`, default is 32
- `--epochs=`, default is 350
- `--data-dir=`, default is '../'
- `--resume=`, default is '../' (type the complete filename of the stored model)
- `--model=`, default is 'original-fp' (can choose between original, gimli_v2, or gimli_with_attention)

Use

  `nohup python train.py --batch-size.... > ./output.log 2>&1&`
  
to run as background process


**Train in GAN setting**

Run the training loop with arguments
  `python trainDCGAN.py`
to train gimli_v2 with default arguments.

Available arguments (same as train.py):
- `--batch-size=`, default is 32
- `--epochs=`, default is 350
- `--data-dir=`, default is '../'

Use

  `nohup python trainDCGAN.py --batch-size.... > ./output.log 2>&1&`

to run as background process



## Files:
### gimli.py 
contain the models; a generator and a discriminator. The generator is a CAE+miniBERT+RN and takes an image (3x180x120) and language encoding through a convolutional encoder, relational network and a convolutional decoder. The discriminator classifies images as real, fake or wrong. Trained in GAN settings the generator tries to convince the discriminator that its images are real.

### gimli_v2.py
same modules as in 'gimli.py' but with an extra convolutional layer creating a total of 1024 feature maps.

### gimli_with_attention.py
replaces the relational network (RN) module with a multi-head attention (MHA) module. Otherwise, same as original gimli.

### datasets.py 
loads the CSS3D dataset from a local directory.

### train.py 
trains the generator model.

### GAN_dataset.py 
loads the CSS3D dataset from a local directory, but also adds a 'wrong_img' on top of the 'source_img' and 'target_img', this is to match the labels for the discriminator.

### trainDCGAN.py 
trains the deep convolutional generative adversarial network comprised of the generator and discriminator models.


## --- Work in progress ---
