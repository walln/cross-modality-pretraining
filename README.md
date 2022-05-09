# Cross Modality Learning

### Overview

Applying transfer learning to large(ish) transformer models with varying degrees of pretraining and fine-tuning to evaluate the value of certain data modalities for pretraining. Trying to determine if certain data types such as natural language have more intrensic value for bootstraping models on downstream tasks or even different modalities.

### Credits

Much of this codebase is based on work done in the paper "Pretrained Transformers As Universal Computation Engines" from Lu et. al. While much of the code has been adapted for our use case and lots of new features have been added and other have been removed, a significant amount of the code can be accredited to the work from this paper.

## Usage

### Installation

1. Install dependencies

   ```
   $ pip install -r requirements.txt
   ```

2. Import experiment to run

   ```python
   from scripts.experiments import gpt2_full_train_cifar
   if __name__ == "__main__":
       gpt2_full_train_cifar()
   ```
