# VQVAE (Vector-Quantized Variational Autoencoder) for 2D Pelvic MRI Image Reconstruction from the HiPMRI Prostate Cancer Dataset

## Project Description

This project introduces a VQ-VAE model that aims to reconstruct 2D pelvic MRI slices from the CSIRO HipMRI dataset. It relies on a trainable codebook of embeddings to learn discrete latent representations, thus enabling medical image reconstructions that are both interpretable and of high quality.

The primary objective is to generate reconstructions that maintain the anatomical structure and the integrity of soft tissue. This is necessary for most downstream applications of medical imaging, such as radiotherapy treatment planning.

The goal is to achieve Structural Similarity Index Measure (SSIM) above 0.6 for image reconstruction to get reasonably clear image and be accepted clinically.

## VQVAE Model Overview

In contrast to traditional autoencoders such as VAE, with continuous latent spaces, VQVAE is a generative model which introduces a vector quantization which replaces continuous latent space to discrete codebook of learned embeddings, which quantizes outputs from encoder to the closest embedding vector.

As a result, VQVAE can maintain these advantages; more stable latent representations, reduced posterior collapse, better structural consistency in reconstructions, latent feature interpretability, which is critical during medical analysis.

## Implementation of VQVAE

**Architecture Components are as follows:**

**Encoder:** The encoder compresses the 2D MRI slices with three convolutional layers, two strid for 4x down-sampling and one more for feature refinement into a residual stack layer that preserve structural detail.

**Vector Quantizer (EMA):** Vector quantizer EMAs are used to discretize the latent variables by maintaining a moving-average-updated codebook. Each encoded vector is substituted with its closest embedding, enforcing discrete independent labels and balancing the codebook usage.

**Decoder:** It has a standard convolution for mixing features and a residual black after that to refine the latent representation. Then, two transposed convolutional layers, these layers are used to do up-sampling the quantized embeddings by a 4x factor. These helps for MRI images to be reconstructed to its original spatial dimensions.

**Residual Block:** It consists of the 3×3 and 1×1 convolution combined with skip connections, making sure the convergence with stable gradients and retaining anatomical precision.

## Environment Setup

There is requirements.txt in this repository which allows reproducibility of the environment required for run files for this project. I ran all files on my own PC with the environment being shared with one more project I am currently working on.

To run training and prediction scripts, I ran following commands:

Goto the folder where these files are located, activate the environment, and run the scripts as below.

```
python train.py --data ./Data/keras_slices_data --outdir runs/hipmri_vqvae --epochs 20 --batch-size 16 --workers 4
```
```
python predict.py --data ./Data/keras_slices_data --ckpt ./runs/hipmri_vqvae/checkpoint_best_ssim_0.924.pt --outdir Output/predict --num-
samples 12 --viz-codebook
```

## Optimization and Loss functions

The VQ-VAE model minimizes two main loss terms while updating the codebook using an EMA strategy.

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 6px;">Loss Term</th>
    <th style="border: 1px solid black; padding: 6px;">Formula</th>
    <th style="border: 1px solid black; padding: 6px;">Purpose</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><b>Reconstruction Loss</b></td>
    <td style="border: 1px solid black; padding: 6px;">L<sub>recon</sub> = ‖x − x̂‖²</td>
    <td style="border: 1px solid black; padding: 6px;">Calculates the pixel-level difference between the input and reconstructed MRI slices. It helps push the decoder to generate anatomically viable results.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><b>Commitment Loss</b></td>
    <td style="border: 1px solid black; padding: 6px;">L<sub>commit</sub> = β ‖zₑ − sg(z<sub>q</sub>)‖²</td>
    <td style="border: 1px solid black; padding: 6px;">Keeps encoder outputs (zₑ) close to their quantized embeddings (z<sub>q</sub>), preventing oscillation and stabilizing the latent space.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><b>Codebook Update (EMA)</b></td>
    <td style="border: 1px solid black; padding: 6px;">e<sub>j</sub> ← decay × e<sub>j</sub> + (1 − decay) × mean(zₑ)</td>
    <td style="border: 1px solid black; padding: 6px;">Applies exponential moving-average updates to each embedding vector based on assigned encoder outputs, enabling smoother non-gradient codebook learning.</td>
  </tr>
</table>

<br>
It applies updates to each embedding vector’s EMA of the mean of the assigned encoder outputs, allowing smoother and non-gradient codebook learning.

**Total loss:
L_total = L_recon + L_commit**

EMA updates are done outside from gradient descent, this makes sure that we have a balanced and stable codebook.

**Additional Notes:** Optimizer is AdamW; intervention weight β = 0.25, balancing fidelity vs. stability; EMA decay 0.99, driving the smoothness of codebook updates; Perplexity is tracked as a metric, not part of the loss; it quantifies how many codebook entries are actively used.

## Dataset Details

Following image contains the details about the dataset.

Dataset splits and image size details can be seen in following image.


**Data Normalization**

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 6px;">Location</th>
    <th style="border: 1px solid black; padding: 6px;">Function</th>
    <th style="border: 1px solid black; padding: 6px;">Type of Normalization</th>
    <th style="border: 1px solid black; padding: 6px;">Applied When</th>
    <th style="border: 1px solid black; padding: 6px;">Output Range</th>
    <th style="border: 1px solid black; padding: 6px;">Purpose</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;">Load data 2D function</td>
    <td style="border: 1px solid black; padding: 6px;">Z-score normalization ((x − mean) / std)</td>
    <td style="border: 1px solid black; padding: 6px;">Intensity standardization per slice</td>
    <td style="border: 1px solid black; padding: 6px;">When <code>normImage=True</code> in dataset</td>
    <td style="border: 1px solid black; padding: 6px;">≈ [−3, 3]</td>
    <td style="border: 1px solid black; padding: 6px;">Normalize MRI contrast and brightness</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;">Normalize minmax per image function</td>
    <td style="border: 1px solid black; padding: 6px;">Min–max normalization ((x − min) / (max − min))</td>
    <td style="border: 1px solid black; padding: 6px;">Per-image scaling for evaluation/visualization</td>
    <td style="border: 1px solid black; padding: 6px;">When called directly in <code>utils</code> or <code>predict</code></td>
    <td style="border: 1px solid black; padding: 6px;">[0, 1]</td>
    <td style="border: 1px solid black; padding: 6px;">Convert tensors for SSIM computation or display</td>
  </tr>
</table>
<br>


**Original vs Preprocessed data**


## Project Structure

<table>
  <tr>
    <th>File</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><b>dataset.py</b></td>
    <td>Loads and preprocesses data for MRI slices by converting NIfTI images into PyTorch tensors, normalizing using either Z-score or min-max, and sets up other such as train and validation test datasets.</td>
  </tr>
  <tr>
    <td><b>modules.py</b></td>
    <td>Specifies the complete VQ-VAE model architecture, from the Encoder, Decoder, and Residual blocks, as well as the VQ layer and vector quantizer with exponential-moving-averages. This is the primary model specification called when running the training as well as inference of the VQ-VAE.</td>
  </tr>
  <tr>
    <td><b>utils.py</b></td>
    <td>Training and evaluation utility functions, including setting random seeds, computing SSIM, image normalization, loss on the reconstruction, saving model checkpoints.</td>
  </tr>
  <tr>
    <td><b>visualization_utils.py</b></td>
    <td>This file will also include plotting utilities for visualizing training curves, codebook usage, latent representations, and sample reconstructions. This will help us understand our model’s performance better.</td>
  </tr>
  <tr>
    <td><b>train.py</b></td>
    <td>A training loop implementation that loads data, initializes the model, optimizer, and scheduler, tracks the loss and ssim metrics, and saves the best checkpoint in terms of performance.</td>
  </tr>
  <tr>
    <td><b>predict.py</b></td>
    <td>Inference with the trained checkpoint. This includes reconstructing MRI slices, computing quantitative metrics such as SSIM and the loss, generating a montage and codebook visualizations, and saving the outputs.</td>
  </tr>
</table>


### Hyperparameter Tuning

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 6px;">Parameter</th>
    <th style="border: 1px solid black; padding: 6px;">Component</th>
    <th style="border: 1px solid black; padding: 6px;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>embedding_dim (D)</code></td>
    <td style="border: 1px solid black; padding: 6px;">Vector Quantizer</td>
    <td style="border: 1px solid black; padding: 6px;">Size of each code vector; controls latent feature richness.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>num_embeddings (K)</code></td>
    <td style="border: 1px solid black; padding: 6px;">Vector Quantizer</td>
    <td style="border: 1px solid black; padding: 6px;">Number of discrete codes in the codebook; determines quantization granularity.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>commitment_cost (β)</code></td>
    <td style="border: 1px solid black; padding: 6px;">Vector Quantizer</td>
    <td style="border: 1px solid black; padding: 6px;">Weight for the loss term that encourages encoder outputs to stay close to codebook vectors.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>ema_decay</code></td>
    <td style="border: 1px solid black; padding: 6px;">Vector Quantizer</td>
    <td style="border: 1px solid black; padding: 6px;">Exponential Moving Average update rate for stable codebook adaptation.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>hidden_channels</code></td>
    <td style="border: 1px solid black; padding: 6px;">Encoder/Decoder</td>
    <td style="border: 1px solid black; padding: 6px;">Width of convolutional layers; determines representational capacity.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>res_hidden_channels</code></td>
    <td style="border: 1px solid black; padding: 6px;">Residual Blocks</td>
    <td style="border: 1px solid black; padding: 6px;">Channels within residual layers for feature refinement.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>num_res_layers</code></td>
    <td style="border: 1px solid black; padding: 6px;">Residual Blocks</td>
    <td style="border: 1px solid black; padding: 6px;">Number of stacked residual layers; deeper stacks capture complex spatial details.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>lr</code></td>
    <td style="border: 1px solid black; padding: 6px;">Training</td>
    <td style="border: 1px solid black; padding: 6px;">Learning rate for the optimizer.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>div_weight</code></td>
    <td style="border: 1px solid black; padding: 6px;">Regularizer</td>
    <td style="border: 1px solid black; padding: 6px;">Balances diversity loss to prevent codebook collapse.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 6px;"><code>noise_std</code></td>
    <td style="border: 1px solid black; padding: 6px;">Training</td>
    <td style="border: 1px solid black; padding: 6px;">Gaussian noise for encoder outputs during early epochs to encourage exploration.</td>
  </tr>
</table>
<br>


## Training outputs


**Results summary:**

Mean Test SSIM: 0.9 24 ± 0.

Codebook Perplexity : ≈ 334

Epochs Trained : 20

**During training:**

- Loss curves decreased in a canonical way, through the time of the epochs.
- SSIM values displayed steady growth, peaking around ~0.93 on the validation set.
- Perplexity values were stabilized around 320-340, indicating robust codebook activity.


**Codebook Usage at epoch 20**

## Prediction outputs


**predict_summary.json**

{

"mean_test_ssim": 0.9281258694827557,

"num_samples": 12

}

**Reconstruction Quality:** High structural similarity between validation and test sets, with tissue boundaries and internal contrast preserved and few artifacts or texture loss.


**Codebook usage**

## Observation and Insights

The model has high reconstruction fidelity with sharp boundaries and smooth transitions, uses the entire 512 entries effectively and minimal overfitting, and  stabilizes code book assignment through noise annealing in the early stages of training.


## Conclusion

This project implements VQVAE successfully for Hip MRI image reconstruction with high accuracy and interpretability.

SSIM ≈ 0.9 24 ± 0.

Loss ≈ 0.02 9

Perplexity ≈ 334

The measure and evaluation metrics indicate better generalization and no overfitting.

## Future Work

Future directions for this project are investigating multi-scale or hierarchical VQ-VAE architectures to capture finer anatomical details at multiple representation levels, extending the framework to 3D or temporal MRI reconstruction for volumetric and dynamic imaging tasks. Also, incorporating perceptual or adversarial losses e.g., those used in VQ-GAN to enhance the image sharpness and realism, and conducting cross-dataset validation to evaluate the robustness and generalization across various MRI datasets.

## References

1. Google DeepMind. (2021). *Sonnet VQ-VAE Example.*  Retrieved from [https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb) *(A useful implementation for reference and comparison with the VQ-VAE model.)*

2. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning.* *Advances in Neural Information Processing Systems (NeurIPS).*  [https://arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937)

3. Razavi, A., van den Oord, A., & Vinyals, O. (2019). *Generating Diverse High-Fidelity Images with VQ-VAE-2.*  *Advances in Neural Information Processing Systems (NeurIPS).* [https://arxiv.org/abs/1906.00446](https://arxiv.org/abs/1906.00446)

4. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.*  *International Conference on Learning Representations (ICLR).  [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

5. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW).* *International Conference on Learning Representations (ICLR).[https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)



