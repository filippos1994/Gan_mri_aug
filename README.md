### Repository of the paper:  
## *Generative Adversarial Networks as an advanced data augmentation technique for MRI data* 
### by Filippos Konidaris, Thanos Tagaris, Maria Sdraka and Andreas Stafylopatis

Abstract:

> This paper presents a new methodology for data augmentation through the use of Generative Adversarial Networks.  Traditional augmentation strategies are severely limited, especially in tasks where the images follow strict standards, as is the case in medical datasets.  Experiments conducted on the ADNI dataset prove that augmentation through GANs greatly outperforms traditional methods, based both on the validation accuracy and the models’ generalization capability on a holdout test set. Although traditional data augmentation did not seem to aid the classification process in any way, by adding GAN-based augmentation an increase of 11.68% in accuracy was achieved. Furthermore, even higher scores can be reached by combining the two schemes.

You can download the full paper from [here](http://insticc.org/node/TechnicalProgram/visigrapp/presentationDetails/73639).

## Repository structure:

1. **Preprocessing**
    - Python scripts that divide the subjects into training-validation-test sets, convert volumetric nifty images to 2D .png, resize them and drop their first and last frames (i.e. frames containing background and useless parts of the brain).   
2. **GAN models**
    - GAN architecture and training script.
    - Weights.
3. **Classification**
    - Script that combines GAN-generated images with real ones, at different ratios, to create new augmented images for the subsequent experiments.  
    - ResNet18 architecture and training scripts.
4. **Logs**
    - GAN training losses.
    - Classification experiments.
      - TensorBoard logs.
      - Runtime metrics.
5. **Results**
    - GAN evaluation.
      - Sample AD and NC images generated from GANs.
      - A notebook for empirically evaluating the generated images through a k-NN model.
    - Classification experiments.
      - Scripts for evaluating the ResNet-50 models
      - The final results in csv form
      - A notebook for creating the figures used in the paper. 

## Requirements

All scripts were either run with a python3 interpreter or on a Jupyter notebook with a python3 kernel.

- numpy (1.13.3)
- pandas (0.20.3)
- matplotlib (2.0.2)
- seaborn (0.8.1)
- tensorflow (1.4.1)
- keras (2.1.2).
- scikit-learn (0.19.0) -> Required for the empirical evaluation of GAN-generated images through a k-NN model.
- nibabel (2.2.1) -> Used for reading the volumetric nifty images (*.nii*). 

## 1. Preprocessing

To download the dataset:

- Visit https://ida.loni.usc.edu/login.jsp?project=ADNI&page=HOME# and click on SEARCH (you will have to sign up)
- Click on Advanced Image Search (beta)
- On the Search Options menu, select Pre-processed on the IMAGE TYPES sub-menu
- On the Search Criteria menu, select MRI on the IMAGE sub-menu. Then, on the IMAGING PROTOCOL sub-menu, select T1 on Weighting
- CAUTION! Even though the experiments were performed on axial MRI data, do not select AXIAL on the Aquisition Plane, since most of the original data are volumetric .nii files)
- To download the AD (Alzheimer's Disease) subjects, on the Search Options menu, select Subject on the Search Sections sub-menu, and on the Search Criteria menu, select AD on the subject sub-menu. The same holds for NC (Normal Control) subjects.
- Of all the downloaded data, select the "MPR; GradWarp; B1 Correction; N3; Scaled" ones.
- For compatibility, store images downloaded for AD subjects to a directory named 'ad' and images from NC subjects to a directory called 'nor'.

After the dataset is downloaded, run:

- `1_ttv_split.py`, in order to randomly split the patients on training, validation and test sets
- `2_subjects_lists.py` (optional), to log which subject is on which set
- `3_nii_to_png_all.py`, to transform the *.nii* volumetric data to axial *.png* MRI images
- `4_check_shapes.ipynb` (optional) to find the number of different slices and dimensions of each visit's corresponding images
- `5_select_sequences_ttv.py`, to throw away the images corresponding to the irrelevant parts of a subject's head, as far as AD diagnosis is concerned
- `6_resize_images.py`, to resize all the images to a single size (192,160)

## 2. GAN Models

Two identical GANs were trained, one per class, following the Wasserstein GAN framework.

### Generator

An architecture with 11 layers and more than 15 million trainable parameters was selected as the generator of the network.

The input of the generator is a vector of 128 random values in the \[0, 1) range, sampled from a uniform distribution. Following the input is a Fully Connected (FC) layer with  6 · 5 · 512 = 15360 neurons. The output of the FC layer is then transformed into a 3D volume, which can be thought of as a 6 × 5 image with 512 channels. The subsequent layers are regular 2D convolutions and 2D transposed convolutions. A 5 × 5 kernel and zero padding were used for both types of layers, while a stride of 2 was used for the transposed convolutions. This results in the doubling of the spatial dimensions of its input. All layers apart from the last are activated by the Leaky ReLU function. The final layer has a hyperbolic tangent (tanh) activation function, as its output needs to be bounded in order to output an image.

Finally, after five alternations of convolution and transposed convolution layers (each of which doubles the size of its input), an image with a resolution of 6 · 2^5 × 5 · 2^5 = 192 × 160 and 1 channel is produced.

### Discriminator

The discriminator is a regular CNN architecture aimed towards binary classification. The one used in the present study consists of 11 layers and around 9.5 million trainable parameters.

The input of the discriminator is a single-channel 192 × 160 image, which is then passed five times through alternating convolutional layers with a stride of 1 and 2 respectively; the latter are used for sub-sampling, instead of pooling layers. The final two layers are FC ones. All layers are activated by a Leaky ReLU function, besides the last one, which has no activation function.

### Usage

- The GAN training script is `wgan_bs_32_lat_128_eps_600.py`. Because of limitations that Keras put on model creation, the training data (the ones produced at the end of Chapter 1) must be stored on a single Numpy array.

- The optimal weights for the generators of both models (i.e. the one trained on the AD subset and the other on the NC subset) are stored at `2_gan_models/weights/`.

- Both models are capable of generating high-quality realistic images of AD/NC subjects respectively. A few sample images can be seen below.

Model trained on the AD subset:

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/gan_evaluation/sample_images/gan_images_ad.png)

Model trained on the NC subset:

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/gan_evaluation/sample_images/gan_images_nor.png)

- This can be further validated through `knn_gan.ipynb`, which selects a generated image at random and searches the original training set to find its closest through a k-NN algorithm. This can confirm that the model can produce high-quality realistic images and does not memorize the training set.

## 3. Classification

After the GANs have been trained, we use 8 different fake to real ratios to train a ResNet-18 model.

- Use `generate_imgs.py` from Chapter 2 to generate the fake dataset. 
- Use `1_permutations.py` to randomly select which images are used in the 8 different training datasets.
- Use `2_create_gan_datasets.py` to actually copy the images on the corresponding datasets.
- Use `3_get_train_gan_stats.py` to only calculate once each dataset's statistical features.
- Run `4_train_resnet18.py` to train the ResNet.

## 4. Losses and Runtime Metrics

### GAN training loss

GAN trained on the AD subset

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/gan_plots/figures/ad_loss.png)

GAN trained on the NC subset

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/gan_plots/figures/nc_loss.png)

### ResNet TensorBoard logs

The TensorBoard training logs for the ResNet models stored under [4_logs_plots/resnet_logs/tensorboard_logs](https://github.com/filippos1994/Gan_mri_aug/tree/master/4_logs_plots/resnet_logs/tensorboard_logs). The identifier *dr_x* signifies the probability of dropout (e.g. *dr_25* refers to a dropout probability of 25%). The identifier *x_f2r* refers to the ratio of fake to real images, in experiments containing GAN-augmented data (e.g. the logs for an experiment with a 75% fake-to-real image ratio would have the identifier *075_f2r*).

### ResNet runtime metrics

During training, at the end of each epoch, the models were evaluated on a hold-out validation set. The figures below depict the validation accuracy of each of the models, during training.

#### I. Baseline

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/baseline.png)

#### II. Traditional augmentation

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/augment.png)

#### III. GAN augmentation

Models trained on a GAN-augmented dataset with 8 different fake/real image ratios. All 8 experiments are run with **no dropout**. 

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/gan_noaug_0dr.png)

Same 8 runs, all trained with a **25% dropout probability**.

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/gan_noaug_25dr.png)

Comparison of the two cases above (i.e. with and without dropout). The error bands span from the best to the worst model in each of the two cases, while their mean is represented by a thicker line.

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/gan_noaug.png)

#### IV. Both forms of augmentation

Models trained on a GAN-augmented dataset with the addition of traditional augmentation techniques. The same 8 fake/real iamge ratios were examined, the *augmentation strength* is set to 25% and **no dropout** is used for any of the runs. 

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/gan_aug_0dr.png)

Same 8 runs, all trained with a **25% dropout probability**.

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/gan_aug_25dr.png)

Comparison of the two cases above (i.e. with and without dropout). The error bands span from the best to the worst model in each of the two cases, while their mean is represented by a thicker line.

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/drop_comparison_4.png)

#### Comparison of the best cases in experiments III and IV

The best case for experiment III was the one that didn't use dropout, while the best for IV had a 25% dropout probability. The eight models from these two cases can be seen below. The error bands span from the best to the worst model in each of the two cases, while their mean is represented by a thicker line.

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/resnet_logs/runtime_metrics/figures/best_iii_iv.png)

## 5. Results: ResNet generalization

These results involved evaluating the best models from the validation set on a hold-out test set.

#### I. Baseline

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/resnet_evaluation/figures/i.png)

#### II. Traditional augmentation

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/resnet_evaluation/figures/ii.png)

#### III. GAN augmentation

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/resnet_evaluation/figures/iii.png)

#### IV. Both forms of augmentation

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/resnet_evaluation/figures/iv.png)

#### Comparison of the best models from all 4 experiments

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/5_results/resnet_evaluation/figures/comparative.png)
