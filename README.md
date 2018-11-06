### Repository of the paper:  
## *Generative Adversarial Networks as an advanced data augmentation technique for MRI data* 
### by Filippos Konidaris, Thanos Tagaris, Maria Sdraka and Andreas Stafylopatis

Abstract:

> This paper presents a new methodology for data augmentation through the use of Generative Adversarial Networks.  Traditional augmentation strategies are severely limited, especially in tasks where the images follow strict standards, as is the case in medical datasets.  Experiments conducted on the ADNI dataset prove that augmentation through GANs greatly outperforms traditional methods, based both on the validation accuracy and the models’ generalization capability on a holdout test set. Although traditional data augmentation did not seem to aid the classification process in any way, by adding GAN-based augmentation an increase of 11.68% in accuracy was achieved. Furthermore, even higher scores can be reached by combining the two schemes.

## Repository structure:

- Preprocessing
- GAN models
  - GAN architecture
  - Weights
- Classification
  - ResNet18 architecture
- Logs
  - GAN training losses
  - Classification experiments
    - TensorBoard logs
    - Runtime metrics
- Results
  - GAN evaluation
  - Classification experiments

## 5. Results

### GAN training loss

GAN trained on the AD subset

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/gan_plots/figures/ad_loss.png)

GAN trained on the NC subset

![](https://github.com/filippos1994/Gan_mri_aug/blob/master/4_logs_plots/gan_plots/figures/nc_loss.png)
