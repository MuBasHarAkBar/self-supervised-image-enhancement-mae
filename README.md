# Self-Supervised Representation Learning for Image Enhancement

A CNN-based masked autoencoder (MAE-style) that reconstructs clean CIFAR-10 images from noisy images with 75% of the patches hidden. It learns without using class labels.

## Method
1. Add Gaussian noise (sigma = 0.15) to a clean image
2. Hide 75% of the 4x4 patches at random
3. Encoder-decoder CNN reconstructs the clean image
4. Loss = full-image MSE + masked-region MSE + SSIM loss

## Results (300 test images)
| Metric | Noisy input | Enhanced output |
|---|---|---|
| PSNR (dB) | [X] | [X] |
| SSIM | [X] | [X] |

![Qualitative results](results/fig4_qualitative_enhancement_results.png)
![PSNR/SSIM comparison](results/fig5_psnr_ssim_comparison.png)

## Run it
    pip install -r requirements.txt
    python mae_image_enhancement.py
CIFAR-10 downloads automatically. A GPU is recommended. Outputs are saved to `report_outputs/`.

## Tech
Python, PyTorch, torchvision, scikit-image, NumPy, Matplotlib

## Limitations
CIFAR-10 images are only 32x32, and the model is small, so fine details are smoothed.
