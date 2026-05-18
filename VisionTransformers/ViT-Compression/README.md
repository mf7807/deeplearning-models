# Vision Transformer (ViT) - Model Compression

## Objective

Apply model compression techniques to a pretrained Vision Transformer (ViT) fine-tuned on Tiny ImageNet in order to reduce model size while maintaining classification performance.


## Base Model

- Architecture: Vision Transformer (ViT)
- Framework: PyTorch
- Dataset: Tiny ImageNet
- Input: 224 × 224 RGB images
- Output: 200-class image classification

### Baseline Performance

- Accuracy: **84.22%**
- Model Size: **329.11 MB**


## Compression Technique

- Applied PyTorch dynamic quantization
- Quantized fully connected layers
- No retraining required after quantization
- Evaluated directly on the Tiny ImageNet validation set


## Results

| Model | Accuracy | Model Size |
|---|---|---|
| Original ViT | 84.22% | 329.11 MB |
| Quantized ViT | 81.70% | 165.52 MB |


## Observations and Analysis

- Dynamic quantization reduced the model size by nearly 50%.
- Accuracy decreased only slightly after compression.
- Quantization worked effectively for reducing transformer model storage while preserving most performance.
- Compared to CNN compression experiments, transformer inference on CPU was slower during evaluation.


## Future Work

- Apply pruning to Vision Transformer models
- Explore quantization-aware training (QAT)
- Compare different transformer architectures such as Swin Transformer and MobileViT
- Experiment with manual and non-linear quantization techniques
