# SFIN: Semantic Field Interference Network

## Overview

SFIN is a neural language model architecture that explores quantum-inspired mechanisms for natural language processing. Building upon the foundation of transformer models, SFIN incorporates complex-valued representations and principles from quantum mechanics to model semantic relationships in text in novel ways. This approach conceptualizes language understanding as a quantum-like system where meanings exist in superposition and interfere with each other—offering a mathematical framework to capture the multifaceted nature of human cognition.

## Key Features

- **Complex-Valued Representations**: Extends transformer embeddings to use both real and imaginary components.
- **Quantum-Inspired Attention**: Adapts attention mechanisms to incorporate quantum interference principles.
- **Multi-Scale Processing**: Simultaneously processes text at word, phrase, and sentence levels.
- **External Memory Enhancement**: Integrates memory mechanisms inspired by Neural Turing Machines.
- **Quantum-Inspired Evaluation Metrics**: Uses concepts like von Neumann entropy for model analysis.
- **Explainable Attention Visualization**: Provides tools for understanding the model's attention patterns.

## Architecture Details

SFIN integrates several innovative components:

1. **ComplexEmbedding**: Extends conventional embeddings to the complex-valued space.
2. **PositionalEncoding**: Adapts positional embeddings for complex-valued representations.
3. **EntangledInterferenceLayer**: Reimagines attention with quantum-inspired interference.
4. **HierarchicalInterferenceModule**: Implements multi-scale processing inspired by hierarchical transformers.
5. **AdvancedWaveFunctionCollapse**: Novel output layer inspired by quantum measurement.
6. **MemoryModule**: External memory system building on Memory Networks and Neural Turing Machines.
7. **CrossModalFusion**: Optional component for integrating multi-modal inputs.

## Training and Evaluation

SFIN uses standard language modeling procedures with quantum-inspired enhancements:

- **Training Loop**: Incorporates curriculum learning for collapse mechanisms.
- **Mixed-Precision Training**: Utilizes PyTorch amp for efficient training.
- **Visualization**: Integrated TensorBoard support for monitoring progress.
- **Hyperparameter Optimization**: Uses Optuna to fine-tune model parameters.
- **Evaluation Metrics**: Combines classical and quantum-inspired measures (e.g., von Neumann entropy).

## Usage

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers
- Datasets
- Optuna
- TensorBoard
- tqdm
- matplotlib
- sklearn
- scipy

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/waefrebeorn/SFIN.git
   cd SFIN
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train a model:**
   ```bash
   python sfin.py --mode train --model_dim 768 --epochs 3
   ```

4. **Evaluate a trained model:**
   ```bash
   python sfin.py --mode evaluate --checkpoint checkpoints/sfin_best_model.pt
   ```

5. **Generate text:**
   ```bash
   python sfin.py --mode generate --checkpoint checkpoints/sfin_final_model.pt
   ```

6. **Explore model internals:**
   ```bash
   python sfin.py --mode explain --checkpoint checkpoints/sfin_final_model.pt
   ```

### Command Line Arguments

- `--mode`: Operation mode (`train`, `evaluate`, `generate`, `explain`)
- `--hyperopt`: Enable hyperparameter optimization
- `--checkpoint`: Path to model checkpoint
- `--batch_size`: Override automatic batch size selection
- `--learning_rate`: Set learning rate (default: 6e-5)
- `--epochs`: Number of training epochs (default: 15 for quantum settings)
- `--model_dim`: Model hidden dimension size
- `--seed`: Random seed for reproducibility
- `--enable_memory`: Enable memory operations during training (disabled by default)
- `--gradient_accumulation_steps`: Steps to accumulate gradients (useful for limited GPU memory)

## Visualization

SFIN includes a suite of visualization tools for model interpretability:

- **Attention Heatmaps**: Visualize attention patterns for each layer and head.
- **Token Importance Maps**: Assess token contributions during text generation.
- **Gradient Flow Analysis**: Monitor gradient propagation through the network.
- **Quantum Entropy Metrics**: Track entropy-based measurements.
- **Entanglement Visualization**: Display the strength and nature of interference effects.

Visualizations are saved to the `visualizations/` directory during training and evaluation.

## Research Context

SFIN builds upon and extends several important research threads:

1. **Transformer Architecture**: Based on the principles introduced by Vaswani et al. (2017).
2. **Quantum NLP**: Inspired by quantum natural language processing work (e.g., Coecke, Sadrzadeh & Clark, 2010; Meichanetzidis et al., 2020).
3. **Complex-Valued Neural Networks**: Extends ideas from Hirose (2012) and Trabelsi et al. (2018).
4. **Memory-Augmented Networks**: Incorporates elements from Memory Networks (Weston et al., 2015) and Neural Turing Machines (Graves et al., 2014).
5. **Quantum Cognition**: Grounded in theoretical models by Busemeyer & Bruza (2012) and Pothos & Busemeyer (2013).

## Future Directions

Potential extensions and improvements include:

- Pre-training on larger and more diverse corpora.
- Adapting SFIN for additional NLP tasks beyond language modeling.
- Experimenting with alternative quantum-inspired collapse mechanisms.
- Integration with quantum computing frameworks.
- Expanding capabilities for cross-lingual and multi-modal applications.

## Citation

If you use SFIN in your research, please cite this implementation along with the foundational works:

```bibtex
@software{sfin2025,
  author = {WaefreBeorn},
  title = {SFIN: Semantic Field Interference Network},
  year = {2025},
  url = {https://github.com/waefrebeorn/SFIN}
}
```

Additional references:
- Busemeyer, J. R., & Bruza, P. D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Coecke, B., Sadrzadeh, M., & Clark, S. (2010). Mathematical foundations for a compositional distributional model of meaning.
- Meichanetzidis, K., Toumi, A., de Felice, G., & Coecke, B. (2020). Grammar-aware sentence classification on quantum computers.
- Hirose, A. (2012). *Complex-Valued Neural Networks: Advances and Applications*. John Wiley & Sons.
- Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines.
- Weston, J., Chopra, S., & Bordes, A. (2015). Memory Networks.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We extend our sincere thanks to the researchers and practitioners whose foundational work in quantum cognition, complex-valued neural networks, and memory-augmented architectures has greatly influenced SFIN. Special thanks to:

- **Quantum Cognition Pioneers**: Busemeyer & Bruza for their groundbreaking theories.
- **Quantum NLP Innovators**: Coecke, Sadrzadeh & Clark; Meichanetzidis et al. for laying the groundwork in quantum-inspired language processing.
- **Complex-Valued Network Researchers**: Hirose and colleagues for advancing complex neural network methodologies.
- **Memory-Augmentation Visionaries**: Graves et al. and Weston et al. for their contributions to memory-based architectures.

---

By integrating a diverse set of inspirations and cutting-edge techniques, SFIN aims to push the boundaries of natural language understanding. We hope this project serves as both a tool and a foundation for further research and development in quantum-inspired NLP.
