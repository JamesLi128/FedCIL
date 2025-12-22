# Knowledge Distillation in FCIL: GLFC and Hybrid Replay Approaches

## 1. Global-Local Forgetting Compensation (GLFC)

### Overview

GLFC introduces a dual-component local forgetting compensation mechanism that operates at each client without requiring explicit exemplar storage. The method consists of:

1. **Class-Aware Gradient Compensation Loss (L_GC)**: Balances the learning pace of new classes against the forgetting pace of old classes through gradient reweighting
2. **Class-Semantic Relation Distillation Loss (L_RD)**: Preserves inter-class semantic relationships from the previous task

### Mathematical Formulation

#### Class-Aware Gradient Compensation Loss

The gradient compensation loss reweights each sample's cross-entropy loss by the inverse of its class-specific gradient magnitude:

$$L_{GC} = \frac{1}{b} \sum_{i=1}^{b} \frac{|G_i^t|}{\bar{G}_i} \cdot \mathcal{L}_{CE}(\hat{P}(x_i^t, \Theta), y_i^t)$$

Where the gradient term is defined as:

$$G_i^t = \hat{P}(x_i^t, \Theta)_{y_i^t} - 1$$

This represents the gradient of the cross-entropy loss with respect to the logit of the true class neuron.

The normalization factor is class-dependent:

$$\bar{G}_i = \begin{cases} 
G_n & \text{if } y_i^t \in \mathcal{Y}^t \text{ (new class)} \\
G_o & \text{if } y_i^t \in \mathcal{Y}^{t-1} \text{ (old class)}
\end{cases}$$

Where the class-specific gradient statistics are computed as:

$$G_n = \frac{1}{b_n} \sum_{i: y_i^t \in \mathcal{Y}^t} |G_i^t|$$

$$G_o = \frac{1}{b_o} \sum_{i: y_i^t \in \mathcal{Y}^{t-1}} |G_i^t|$$

**Intuition**: New classes typically have larger gradient magnitudes (more abundant training data and steeper loss landscape), so dividing by $G_n$ reduces their contribution. Old classes have smaller magnitudes (fewer exemplars, flatter loss landscape), so dividing by the smaller $G_o$ amplifies their contribution. This achieves balanced learning paces.

#### Class-Semantic Relation Distillation Loss

Rather than freezing old class predictions, GLFC preserves the semantic relationships between classes learned in the previous task:

$$L_{RD} = D_{KL}(\hat{P}^t(X_b, \Theta^{r,t}) \| \bar{Y}^t(X_b, \Theta^{t-1}))$$

The current model's output distribution over all classes (old + new):

$$\hat{P}^t(X_b, \Theta^{r,t}) = [\hat{p}_1(x_b), \ldots, \hat{p}_{C_p + C_t}(x_b)] \in \mathbb{R}^{b \times (C_p + C_t)}$$

The target distribution, which preserves old class semantics but uses one-hot encoding for new classes:

$$\bar{Y}^t(X_b, \Theta^{t-1}) = [\hat{P}^{t-1}(X_b, \Theta^{t-1})_{:, 1:C_p} \; \| \; \text{one-hot}(y_b^t)_{:, C_p+1:C_p+C_t}]$$

Here $\Theta^{r,t}$ denotes the replica of old task model parameters without gradient updates (used only for target generation).

**Key Innovation**: Old class predictions come from the previous model (soft targets), preserving learned confusion patterns and semantic distances. New class predictions are one-hot encoded since the previous model has no knowledge of them. This prevents the new model from changing how it relates old classes to each other.

#### Combined Local Loss

At each client during round $t \geq 2$ (after first task):

$$L_{\text{local}}^t = \lambda_1 \cdot L_{GC} + \lambda_2 \cdot L_{RD} + (1 - \lambda_1 - \lambda_2) \cdot L_{CE}^{\text{new}}$$

Typical hyperparameter settings: $\lambda_1 = 0.5$, $\lambda_2 = 0.5$ for $t \geq 2$, and $L_{CE}^{\text{new}}$ is standard cross-entropy on new class examples.

### Training Algorithm (Client-Side)

```
Algorithm: GLFC Local Training at Client k, Task t
Input: X_batch = {new task samples, exemplars of old classes}
       θ_{t-1}^k = model from previous task
       θ_replica^{t-1,k} = frozen copy of θ_{t-1}^k
       
1. Initialize θ_t^k ← θ_{t-1}^k
2. for epoch in 1 to E do
3.   for batch in X_batch do
4.     Compute gradients: G_i^t ← ∂L_CE/∂ŷ_{y_i^t} for each sample i
5.     Compute class statistics: G_n, G_o from current batch
6.     Compute L_GC using reweighted cross-entropy
7.     Compute L_RD using KL divergence with replica model
8.     L_total ← λ_1·L_GC + λ_2·L_RD
9.     θ_t^k ← θ_t^k - α·∇L_total
10.  end for
11. end for
12. Return θ_t^k
```

### Server-Side Aggregation with Proxy

GLFC employs a lightweight proxy server mechanism to coordinate class-level knowledge:

$$\theta_t^* = \frac{1}{K} \sum_{k=1}^K \theta_t^k$$

The proxy server maintains global class prototypes (centroids) using a Lennard-Jones potential formulation to prevent inter-class overlap:

$$L_{\text{sep}} = \sum_{c_i, c_j: i < j} \exp(-\|p_i - p_j\|_2)$$

This separation loss encourages class prototypes to be well-distributed in the embedding space, which is synchronized with clients periodically.

---

## 2. Hybrid Replay (HR) - ICLR 2025

### Overview

The Hybrid Replay approach combines three complementary mechanisms:

1. **Latent Exemplar Replay**: Stores compressed exemplars in latent space (128-256 dimensions) rather than pixel space
2. **Data-Free Synthetic Generation**: Generates new samples from learned class centroids without accessing original data
3. **Dual-Path Knowledge Distillation**: Distills knowledge through both encoder and decoder pathways

### Architecture

The core architecture is a customized variational autoencoder (VAE):

- **Encoder**: Maps images to latent features: $f_\phi: \mathcal{X} \to \mathcal{Z}$ (parametrized by $\phi$)
- **Decoder**: Reconstructs images from latent codes: $g_\psi: \mathcal{Z} \to \mathcal{X}$ (parametrized by $\psi$)
- **Classification Head**: Linear layer on top of latent features: $h_\omega: \mathcal{Z} \to \mathcal{Y}$ (parametrized by $\omega$)

Combined parameters: $\Theta = \{\phi, \psi, \omega\}$

### Mathematical Formulation

#### Variational Autoencoder Loss with Centroid Clustering

The base loss function combines three terms:

$$L_{\text{VAE}}(x, \hat{x}, z) = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + \beta \cdot D_{KL}(q(z|x) \| p(z)) + \lambda \cdot L_{\text{centroid}}(z)$$

**Reconstruction Term**:
$$L_{\text{recon}} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] = \|x - g_\psi(z)\|_2^2$$

This ensures the decoder can still reconstruct samples from previous tasks.

**KL Regularization**:
$$D_{KL}(q(z|x) \| p(z)) = \sum_{j=1}^{d_z} \left( -\frac{1}{2} + \frac{1}{2}(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2) \right)$$

This encourages the latent distribution to match the prior $p(z) = \mathcal{N}(0, I)$, maintaining smooth interpolation between class regions.

**Centroid Clustering Loss**:
$$L_{\text{centroid}}(z) = \frac{1}{b} \sum_{i=1}^b \|z_i - m_{y_i}\|_2^2$$

Where $m_y$ is the exponentially moving average of latent features for class $y$:

$$m_y^{(t+1)} = \rho \cdot m_y^{(t)} + (1-\rho) \cdot \frac{1}{n_y} \sum_{i: y_i = y} z_i$$

with momentum coefficient $\rho = 0.99$.

#### Encoder Knowledge Distillation

To prevent catastrophic forgetting, the encoder's learned representation space must remain stable when the model is updated with new task data:

$$L_{KD}^{\text{enc}} = \|f_\phi(x_{\text{old}}, \Theta_{t-1}) - f_\phi(x_{\text{old}}, \Theta_t)\|_2^2$$

Where $x_{\text{old}}$ are exemplars or synthesized samples from previous tasks. This loss ensures that the updated encoder produces similar latent representations for old data as the previous encoder did.

#### Decoder Knowledge Distillation

Similarly, the decoder must preserve its ability to reconstruct old samples under the new model:

$$L_{KD}^{\text{dec}} = \|g_\psi(f_\phi(x_{\text{old}}, \Theta_{t-1}), \Theta_{t-1}) - g_\psi(f_\phi(x_{\text{old}}, \Theta_t), \Theta_t)\|_2^2$$

**Interpretation**: 
- Left side: Previous encoder + previous decoder (frozen reference)
- Right side: Current encoder + current decoder (what we're training)

This forces the new decoder to reconstruct samples the same way as before, even though the encoder might produce slightly different latent codes.

#### Combined Loss During Class-Incremental Task

When task $t$ arrives with new classes $\mathcal{Y}^t$:

$$L_{\text{total}}^t = L_{\text{VAE}}(x^t) + \alpha \cdot L_{KD}^{\text{enc}}(x_{\text{old}}) + \beta \cdot L_{KD}^{\text{dec}}(x_{\text{old}}) + \gamma \cdot L_{CE}(x^t)$$

Where:
- $L_{\text{VAE}}(x^t)$ operates on new task samples
- $L_{KD}^{\text{enc}}, L_{KD}^{\text{dec}}$ operate on old task exemplars/synthesized samples
- $L_{CE}$ is classification loss on new classes
- Typical hyperparameters: $\alpha = 0.5$, $\beta = 0.5$, $\gamma = 1.0$

### Exemplar Management

#### Latent Exemplar Selection

For each class, HR stores $m$ exemplars in latent space (not pixel space). Selection uses the centroid distance criterion:

$$\mathcal{E}_c = \{x_i : x_i = \arg\text{min}_m \|f_\phi(x_i) - m_c\|_2\}$$

This selects the $m$ samples whose latent features are closest to their class centroid—representing the most "typical" samples.

**Memory Efficiency**: Storing latent codes of dimension 128-256 requires orders of magnitude less memory than storing full images. For 190 exemplars per client with 128-dim latents: $190 \times 128 \times 4 \text{ bytes} \approx 97 \text{ KB}$ vs. $190 \times 32 \times 32 \times 3 \times 4 \approx 23 \text{ MB}$ for RGB images.

#### Data-Free Synthetic Generation

When exemplars are exhausted or to increase diversity, HR generates synthetic samples from learned class centroids:

$$\tilde{x}_y^{\text{synthetic}} = g_\psi(\text{sample from } \mathcal{N}(m_y, \sigma_y I))$$

Where $\sigma_y$ is a learned or fixed standard deviation controlling diversity around the centroid. This leverages the VAE decoder to generate realistic samples without accessing original data.

### Training Algorithm (Task t)

```
Algorithm: Hybrid Replay Training at Client k, Task t
Input: D_new = {new task training samples}
       E_old = {latent exemplars from tasks 1...t-1}
       θ_{t-1}^k = model from previous task
       m_c^{t-1} = class centroids from previous task

1. Initialize θ_t^k ← θ_{t-1}^k, m_c^t ← m_c^{t-1}
2. Merge batch: D_batch ← {D_new, E_old}
3. for epoch in 1 to E do
4.   for minibatch in D_batch do
5.     Sample z ~ q(z|x) for each x in minibatch
6.     Compute L_VAE, L_centroid using new task samples only
7.     Compute L_KD^enc, L_KD^dec using old exemplars
8.     Compute L_CE on new classes
9.     L_total ← L_VAE + α·L_KD^enc + β·L_KD^dec + γ·L_CE
10.    θ_t^k ← θ_t^k - α_lr·∇L_total
11.    Update centroids: m_c^t ← ρ·m_c^{t-1} + (1-ρ)·m_c^{batch}
12.  end for
13. end for
14. Extract latent exemplars: E_new ← TopM(f(D_new), m_c^t)
15. Return θ_t^k, E_new, m_c^t
```

### Server-Side Aggregation

After local training, clients send only model weights (not data or exemplars):

$$\Theta_t^* = \frac{1}{K} \sum_{k=1}^K \Theta_t^k$$

Optional: Server can average class centroids across clients:

$$m_c^* = \frac{1}{K} \sum_{k=1}^K m_c^k$$

This global centroid averaging helps align class representations across heterogeneous data distributions.

### Ablation Study Results (CIFAR-100, Configuration: 10/10/50/5)

| Component | Accuracy | Forgetting |
|-----------|----------|-----------|
| Full HR | 65.84% | 8.2% |
| HR - L_KD^enc | 63.97% | 11.1% |
| HR - L_KD^dec | 64.23% | 10.5% |
| HR - L_centroid | 64.14% | 10.8% |
| HR - Synthetic Gen | 63.48% | 12.3% |
| HR - Exemplar Replay | 62.91% | 14.1% |

The results show that encoder distillation (1.87% drop) has larger impact than decoder distillation (1.61% drop), suggesting that representation stability is more critical than reconstruction stability in this setting.

---

## Comparative Analysis: GLFC vs. HR

| Aspect | GLFC | HR |
|--------|------|-----|
| **Memory** | Gradient buffers (~500 dims) | Latent exemplars (128-256 dims) |
| **Data-Free** | Yes (exemplars via optimization) | Partial (synthesized generation) |
| **KD Method** | KL-divergence semantic | L2 encoder + L2 decoder |
| **Computation** | Medium (gradient reweighting) | High (VAE + classification + KD) |
| **Privacy** | High (no exemplars stored) | Medium (latent features stored) |
| **CIFAR-100 Perf** | 66.9% | 65.84% |
| **Scalability** | Better to 1000+ clients | Limited by latent storage |

---

## Key Implementation Insights

### GLFC Implementation

The critical implementation detail is separating gradient statistics between new and old classes before computing the reweighting factor. Naive implementations that compute a single global gradient magnitude will fail to achieve the balance effect.

```python
# Correct approach
G_n = torch.mean(torch.abs(grad_new_classes))
G_o = torch.mean(torch.abs(grad_old_classes))
weight_factor = torch.where(y in old_classes, G_o, G_n)
weighted_loss = (weight_factor * ce_loss).mean()

# Incorrect approach (fails to balance)
G_global = torch.mean(torch.abs(grad_all))
weighted_loss = (grad_magnitude / G_global * ce_loss).mean()
```

### HR Implementation

The exponential moving average centroid update is crucial for stability. Using batch centroids directly without momentum causes the latent space to drift. The momentum coefficient $\rho = 0.99$ means centroids change slowly, preserving old class regions.

```python
# Correct approach with momentum
centroid_ema = 0.99 * centroid_prev + 0.01 * centroid_batch

# Incorrect approach (causes drift)
centroid_direct = centroid_batch
```

Additionally, the dual KD losses should be applied **only to old task samples**, not new task samples. Applying KD to new task data forces the model to mimic the old encoder's representation of new classes, which it has never seen before.

