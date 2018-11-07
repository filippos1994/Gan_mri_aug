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

### ResNet generalization

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
