# ReinforceAttack

## Project Metadata
### Authors
- **Team:** Ahmed Abdelkader Ahmed
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Patch attacks modify only a small, localized region of an image yet can force a victim model to misclassify—either toward a specific target label or away from the ground-truth label. This paper introduces PatchAttack, a query-efficient black-box method that learns both where to place patches and what they should look like. The attack can be modeled as a reinforcement-learning (RL) problem: an agent chooses a patch (either a monochrome patch or a crop from a learned texture dictionary) and its location (the action space), queries the victim model (the environment) with the patched image, receives a reward based on the model’s output probabilities, and updates its policy to optimize next actions. Figure 1 illustrates this loop.

![Figure 1: black-box texture-based patch attack via reinforcement learning.](Picture1.png)
*Figure 1.* Black-box texture-based patch attack. Source: [arXiv:2004.05682](https://arxiv.org/abs/2004.05682).


On ImageNet models (ResNet, DenseNet, ResNeXt, MobileNetV2), PatchAttack’s textured variant (TPA) reduces top-1 accuracy to <1% with only ~3% of pixels modified and ~10³ queries in the non-targeted case, and achieves ≈100% target-label accuracy while modifying on the order of 10% of the image. Beyond raw success rate, it is markedly more query-efficient than Metropolis–Hastings patch search and remains effective against defenses such as Feature Denoising (Denoise-ResNet152) and shape-biased networks trained on Stylized-ImageNet.


## Problem Statement
The problem is to cause a classifier to err using **only query access** by altering a **small, localized** image region.

#### Mathematical Modeling
Let $f(x;\theta)$ be a deep neural network, where $x$ is the input image and $\theta$ are model parameters. An adversarial example is obtained by applying a transformation $g(\cdot)$ to $x$ and optimizing a loss $L(y, y_0)$ with $y = f(g(x);\theta)$; here $y_0$ denotes a target class and $y$ the ground-truth label. In **targeted** attacks, the objective is to increase the score for $y_0$; in **non-targeted** attacks, the objective is to induce misclassification.

For **patch-based** attacks, the modification is restricted to a small region $E \subseteq \{(u,v)\mid u \in [0,H),\, v \in [0,W)\}$. Pixels inside $E$ are transformed by $T(\cdot)$ while others remain unchanged:

$$
x_{u,v} =
\begin{cases}
T(x_{u,v}), & (u,v)\in E,\\
x_{u,v}, & \text{otherwise.}
\end{cases}
$$

The region $E$ is selected by a search mechanism $s(\cdot)$ over a candidate space $S$ and may depend on the specific image $x$ and network $f$. $H,W$ denote image height and width; $(u,v)$ are pixel coordinates.

**Research question:** Given only model outputs, can a compact, localized modification—specified by $E$ and $T$—reliably induce targeted or non-targeted misclassification while respecting small-area and practical query budgets?



## Application Area and Project Domain
Write 1-2 technical paragraphs (feel free to add images if you would like).

## What is the paper trying to do, and what are you planning to do?
Write 1-2 technical paragraphs (feel free to add images if you would like).


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
