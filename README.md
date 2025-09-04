# Fine-Tuning BERT Model Using HuggingFace IMDB Dataset and Trainer

## Overview
This project aims to fine-tune a BERT (Bidirectional Encoder Representations from Transformers) model using HuggingFace's Trainer API with the IMDB dataset. The goal is to leverage the state-of-the-art capabilities of BERT for sentiment analysis on movie reviews, achieving high accuracy while demonstrating the ease of use of HuggingFace's tools.

## Setup Instructions
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ChapelFob80930/Fine-Tuning-Bert-Model-Using-HuggingFace-IMDB-Dataset-and-Trainer-from-HuggingFace.git
   cd Fine-Tuning-Bert-Model-Using-HuggingFace-IMDB-Dataset-and-Trainer-from-HuggingFace
   ```

2. **Create and Activate a Virtual Environment**  
   ```bash
   python -m venv bert-env
   source bert-env/bin/activate  # On Windows use `bert-env\Scripts\activate`
   ```

3. **Install Required Packages**  
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions
1. **Download the IMDB Dataset**  
   The dataset can be downloaded automatically via the script provided in the repository. Ensure you have internet access.

2. **Run the Fine-Tuning Script**  
   Execute the following command to start fine-tuning:
   ```bash
   python fine_tune.py --model_name bert-base-uncased --output_dir ./model
   ```

3. **Evaluate the Model**  
   After training, you can evaluate the model performance by running:
   ```bash
   python evaluate.py --model_dir ./model
   ```

## Description of Main Notebooks
- **fine_tune.py**: This script contains the code for loading the IMDB dataset, pre-processing the data, and fine-tuning the BERT model using HuggingFace's Trainer API.
- **evaluate.py**: This script is used to evaluate the fine-tuned model on a test set, providing metrics such as accuracy and F1 score.
- **visualization.ipynb**: A Jupyter notebook for visualizing the training process, including loss curves and evaluation metrics.

## Citation and Acknowledgments
- **HuggingFace**: We thank the HuggingFace team for providing the Transformers library, which makes fine-tuning models like BERT straightforward and efficient.
- **IMDB Dataset**: The IMDB dataset used in this project is available at [IMDB](https://www.imdb.com/interfaces/). We acknowledge the importance of this dataset in the field of sentiment analysis.

## Additional Notes and Recommendations
- Ensure that your system has sufficient RAM and GPU capabilities for efficient training.
- It is recommended to experiment with different hyperparameters to find the optimal settings for your specific needs.
- For further information and advanced usage, refer to the [HuggingFace documentation](https://huggingface.co/docs/transformers/index).