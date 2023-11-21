# ConGeLe

Neuroimaging Feature Extraction for Major Depressive Disorders Diagnosis using Contrastive Variational Autoencoder (CVAE) via generative and contrastive learning.

## Repository Contents

This repository contains:

* Source code for the CVAE models and training scripts
* ~~Sample datasets and~~ preprocessing scripts
* Results and analysis scripts
* Visualization tools for model performance
* Additional resources and reference materials

## Running the Code

In order to run the code, make sure you have all the dependencies installed. Instructions for set-up and execution are available in the attached files with this document.

## Abstract

The objective of the dissertation was to tackle the challenge of diagnosing Major Depressive Disorder (MDD) using neuroimaging data. The study focuses on developing and evaluating a Convolutional Variational Autoencoder (CVAE) for feature extraction from brain scans, with an emphasis on identifying neuroimaging biomarkers associated with MDD.

## Introduction

Major Depressive Disorder (MDD) is a common psychiatric ailment that can severely impact a person's quality of life. Current diagnostic methods largely rely on self-reported symptoms and clinical assessments, showing the need for objective and reliable diagnostic tools. Neuroimaging, particularly Magnetic Resonance Imaging (MRI), has the potential to aid in the diagnostic process by revealing patterns that are indicative of the disorder. This research project aims to advance the use of MRI in diagnosing MDD through deep learning approaches that can extract relevant features from complex brain imaging data.

## Methodology

The dissertation employs Convolutional Variational Autoencoders (CVAEs) as the main deep learning tool due to their efficacy in extracting latent features from high-dimensional data. The project involves training CVAEs on a dataset comprising MRI scans to learn efficient representations of the data, with the following key phases:

* Data Collection and Preprocessing
* Model Design and Implementation
* Training and Optimization
* Evaluation and Analysis

### Experimental Setup

* The hyperparameters used for grid search in fine-tuning the CVAE network include:
  * Learning Rate (CVAE/VAE)
  * Learning Rate (Discriminator in CVAE)
  * Beta, Gamma, Alpha (reconstruction loss)
  * Batch Size
  * Discriminator Size
* The training processes are documented and visualized to ensure proper optimization and adjustment where necessary.
* The evaluation of model performance is conducted through various statistical measures such as the Silhouette Score and Normalized Mutual Information (NMI).

## Results

Results include quantitative assessments of the extracted features' quality, with comparisons against baselines and existing models. The performance metrics indicate a CVAE model's capacity to discern relevant features from MRI scans, which could be instrumental in diagnosing Major Depressive Disorder.

## Conclusion

This dissertation contributes valuable insights into the feasibility and effectiveness of using deep learning for neuroimaging analysis in psychiatric diagnosis. Future work can expand on these findings to further refine the models and possibly integrate them into clinical workflows.


## References

The in-depth bibliography is available in the dissertation PDF document file.

## Acknowledgments

The author would like to extend gratitude to the advisors, peers, and participants who contributed to the success of this research study.
