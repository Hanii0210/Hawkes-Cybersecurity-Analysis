# Point Process Modeling for Network Security  
## Inference, Declustering, and Statistical Learning of Hawkes Processes

---

## Highlights (TL;DR)

- Linear-time $O(N)$ likelihood evaluation (vs. $O(N^2$))  
- Stable inference via **Branching EM (Stochastic Declustering)**  
- Statistically valid uncertainty via **Parametric Bootstrap**  
- Empirical validation on real-world cybersecurity traffic  
- Interpretable metric (branching ratio $\eta$) for attack classification  

---

## Abstract

Modern cyber-attacks exhibit strong temporal dependence and cascading behavior, which cannot be captured by traditional Poisson-based models.  

This project develops a rigorous statistical framework based on **Hawkes Processes**, enabling:

- Modeling of **self-exciting attack dynamics**
- Recovery of **latent triggering structures**
- Quantification of **endogenous vs exogenous threats**

We implement a full inference pipeline from scratch, combining **stochastic declustering**, **efficient likelihood computation**, and **statistical validation**, and apply it to real network traffic data.

---
## Implementation Highlights

This project implements and empirically validates the following
components of the Hawkes process inference pipeline:

1. **Stable Inference via Branching EM (Stochastic Declustering)**  
   Implements the stochastic declustering framework of Zhuang et al.
   (2002), demonstrating that naive mixture-EM causes component collapse
   due to shared history collinearity, and resolving it via event-level
   branching attribution.

2. **Linear-Time O(N) Likelihood Evaluation**  
   Applies the Markovian recursion of Ozaki (1979) to reduce likelihood
   computation from O(N²) to O(N).  Empirically verifies the complexity
   reduction via a controlled timing benchmark with log-log slope fitting.

3. **Statistically Valid Uncertainty Quantification**  
   Replaces invalid nonparametric event resampling with parametric
   bootstrap (simulate → refit → collect), correctly preserving the
   self-exciting causal structure of the point process.

4. **Interpretable Cybersecurity Metric**  
   Applies the branching ratio η = α/β as a quantitative indicator of
   attack infectivity, distinguishing automated background scanning
   (η → 0) from coordinated APT lateral movement (η → 1).

---

## Methodology

### Hawkes Process Model

We consider a univariate Hawkes process with exponential kernel:

$$
\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}
$$

where:
- $\mu$: baseline intensity (exogenous events)  
- $\alpha, \beta$: excitation parameters  
- $\eta = \alpha / \beta$: branching ratio  

---

### Stochastic Declustering (Branching EM)

We model the latent structure:

- Each event is either:
  - Background (Poisson)  
  - Triggered by a previous event  

The EM algorithm:

- **E-step:** Estimate triggering probabilities  
- **M-step:** Update parameters via weighted likelihood  

This avoids degeneracy common in naive mixture models.

---

### Computational Optimization

Using recursive updates:

$$
A_i = e^{-\beta (t_i - t_{i-1})}(1 + A_{i-1})
$$

we reduce likelihood computation to linear time.

---

### Uncertainty Quantification

We employ **Parametric Bootstrap**:

1. Fit model → obtain parameters  
2. Simulate synthetic datasets  
3. Re-estimate parameters  
4. Construct confidence intervals  

---

## Experimental Design

### Dataset

- Real-world network traffic dataset: **UNSW-NB15**  
- Temporal extraction of attack events  

---

### Evaluation Goals

- Validate parameter recovery  
- Verify $O(N)$ scaling  
- Analyze attack dynamics via $\eta$  

---

### Key Findings

- MLE exhibits:

$$
O(N^{-1/2})
$$

convergence behavior  

- Branching EM significantly improves:
  - Stability  
  - Interpretability  

- Distinct regimes of $\eta$ correspond to:
  - Random scanning vs. coordinated attacks  

---

## Strategic Interpretation: Branching Ratio ($\eta$)

$$
\eta = \frac{\alpha}{\beta}
$$

| Regime | Interpretation | Security Implication |
|--------|--------------|--------------------|
| $\eta \to 0$  | Exogenous (random scanning) | Baseline filtering |
| $\eta \to 1$  | Highly endogenous (APT behavior) | Immediate containment |

---

## Repository Structure

```text
├── data/
│   └── UNSW-NB15_1.csv
├── src/
│   ├── simulator.py
│   ├── inference.py
│   └── data_loader.py
├── 01_Hawkes_Analysis.ipynb
├── 02_Cybersecurity_Application.ipynb
├── requirements.txt
└── README.md
```

---

## Reproducibility

### Environment

```bash
pip install -r requirements.txt
```

---

### Execution

1. `01_Hawkes_Analysis.ipynb`  
2. `02_Cybersecurity_Application.ipynb`  

---

## Limitations

- Assumes exponential kernel (may miss long-memory effects)  
- Univariate model (no cross-excitation between nodes)  
- Dataset preprocessing may introduce bias  

---

## Future Work

- Multivariate Hawkes processes (networked attacks)  
- Non-parametric kernel estimation  
- Online / streaming inference  
- Integration with real-time intrusion detection systems  

---

## References

- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90. https://doi.org/10.1093/biomet/58.1.83  

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1–22. https://www.jstor.org/stable/2984875  

- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory*, 27(1), 23–31. https://doi.org/10.1109/TIT.1981.1056305  

- Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering of space-time earthquake occurrences. *Journal of the American Statistical Association*, 97(458), 369–380. https://doi.org/10.1198/016214502760046925  

## Author

**[Zihan Xu]**

- Email: [haniizihanxu@gmail.com]  
  
