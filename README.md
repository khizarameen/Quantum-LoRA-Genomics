# Quantum-LoRA-Genomics
## 🎯 Purpose & Motivation

The primary purpose of this project is to explore and benchmark **Quantum Machine Learning (QML)** combined with **Large-Scale Genomic Foundation Models (DNABERT-2)**. 

Traditional fine-tuning of large transformer models requires updating millions of parameters, which is computationally heavy and struggles in low-data biological regimes. This project addresses these challenges by:

1. **Extreme Parameter Efficiency (Quantum-LoRA):** Proving that a 4-qubit variational quantum circuit (`PennyLane`) can act as an effective quantum bottleneck layer, achieving comparable performance while training **190x fewer parameters** compared to classical fine-tuning.
2. **Bridging Genomics and Quantum Computing:** Investigating how hybrid quantum-classical architectures handle complex biological sequence patterns (such as promoter vs. non-promoter regions).
3. **Multi-Task Genomic Modeling:** Demonstrating a unified workflow that handles both discrete classification and continuous regression (gene expression strength) using a single frozen foundation backbone.
