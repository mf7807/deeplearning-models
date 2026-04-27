# ResNet50 — Model Compression (CIFAR-10)

## Objective

Apply and compare different deep learning model compression techniques on a fine-tuned ResNet50 model trained on CIFAR-10. The goal is to reduce model size while maintaining performance.


## Base Model

- Architecture: ResNet50 (Pretrained)
- Dataset: CIFAR-10
- Input: 224 × 224 RGB images
- Output: 10-class classification

**Baseline Performance**
- Accuracy: **92.52% (Model used for pruning)** ,**93.53% (Model used for quantizations)**
- Model Size: ~200 MB



## Compression Techniques

### 1. Structured Pruning

Pruning removes less important weights from the model to reduce complexity.

**Approach**
- Pruned 20% and 30% of weights
- Fine-tuned after pruning for recovery
- Trained only Layer4 and FC

**Results**

| Pruning | Accuracy |
|--------|---------|
| 20%    | 90.78%  |
| 30%    | 86.76%  |

**Observations**
- Moderate pruning (20%) preserves most performance  
- Higher pruning leads to noticeable accuracy drop  
- Fine-tuning helps recover performance after pruning  


### 2. Dynamic Quantization (PyTorch)

Weights are converted from float32 to int8 using built-in PyTorch quantization.

**Approach**
- Applied to fully connected layers
- No retraining required
- Evaluated directly after quantization

**Results**

- Accuracy: **93.51%**
- Model Size: 204.42 MB, Quantized Model Size: ~90 MB

**Observations**
- Significant reduction in model size (~50%)  
- Minimal impact on accuracy  
- Efficient and easy-to-apply compression method  


### 3. Manual Quantization

#### a. Symmetric Quantization

Weights are scaled using a zero-centered range.

- Accuracy: **93.13%**

#### b. Asymmetric Quantization

Weights are scaled using a non-zero offset.

- Accuracy: **93.28%**

**Observations**
- Both methods preserved accuracy well  
- Asymmetric quantization performed slightly better  
- Better suited for non-zero-centered weight distributions  


### 4. Non-Linear Quantization (K-Means)

Weights are clustered into discrete values using k-means.

**Results**

| Clusters (k) | Accuracy |
|-------------|----------|
| 16          | 57.75%   |
| 32          | 87.87%   |
| 60          | **92.66%**   |
| 64          | 87.71%   |

**Observations**
- Small k (high compression) → large accuracy drop  
- Increasing k improves accuracy significantly  
- k=60 achieved near-baseline performance  
- Increasing k beyond effective weight diversity did not improve results  
- K-means requires careful tuning compared to linear quantization  


## Key Insights

- Compression introduces a trade-off between **model size and accuracy**  
- Dynamic quantization provided the best balance of **simplicity and performance**  
- Manual quantization techniques showed strong performance with minimal loss  
- K-means (non-linear) can achieve high accuracy but requires tuning  
- Increasing cluster size improves accuracy but reduces compression efficiency  
- Fine-tuning helps recover performance after aggressive compression  


## Code Reference

Each technique is implemented in separate scripts within this folder:

- `pruning.py`
- `dynamic_quantization.py`
- `symmetric_quantization.py`
- `asymmetric_quantization.py`
- `kmeans_quantization.py`



## Conclusion

Model compression can significantly reduce model size while preserving performance when applied carefully. Among the methods explored, dynamic and manual quantization provided the most efficient compression, while k-means demonstrated the trade-off between compression strength and accuracy.


## Future Work

- Apply compression to Tiny ImageNet models  
- Combine pruning + quantization for hybrid compression  
- Explore quantization-aware training (QAT)  
- Optimize k-means clustering per layer  
