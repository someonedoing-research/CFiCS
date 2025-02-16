# CFiCS: Graph-Based Classification of Common Factors and Microcounseling Skills

This repository contains the code accompanying the paper "CFiCS: Graph-Based Classification of Common Factors and Microcounseling Skills."

## Overview

CFiCS introduces a novel approach utilizing graph-based techniques to classify common factors and microcounseling skills in therapeutic sessions. By representing interactions as graphs, our model captures the nuanced relationships and patterns inherent in counseling dialogues, leading to improved classification performance.

## Repository Structure

- `baselines/`: Contains baseline models used for comparative analysis.
- `config/`: Configuration files for training and evaluation setups.
- `data/`: Sample datasets and data processing scripts.
- `data_loading.py`: Script for loading and preprocessing data.
- `eval.py`: Evaluation metrics and procedures.
- `loss.py`: Custom loss functions used in training.
- `metrics.py`: Implementation of performance metrics.
- `model.py`: Definition of the graph-based classification model.
- `train.py`: Training loop and model optimization routines.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To utilize the CFiCS model for classifying counseling session data, follow the steps below:

1. **Data Preparation**:
   - **Format**: Ensure your data is in a compatible format, such as CSV or JSON, containing transcripts of counseling sessions.
   - **Structure**: Each entry should include necessary fields like speaker labels, timestamps, and utterances.
   - **Preprocessing**: Use the `data_loading.py` script to preprocess your data. This script will convert raw transcripts into graph representations suitable for model input.

2. **Configuration**:
   - **Model Settings**: Adjust hyperparameters and model configurations in `config/model_config.yml` to suit your dataset and objectives.
   - **Training Parameters**: Modify `config/train_config.yml` to set parameters such as learning rate, batch size, and the number of training epochs.

3. **Training the Model**:
   - Execute the `train.py` script to initiate the training process. This script will utilize the configurations set previously and the preprocessed data to train the CFiCS model.
   - Monitor training metrics to ensure the model is learning appropriately. Adjust configurations as necessary based on performance.

4. **Evaluation**:
   - After training, run `eval.py` to evaluate the model's performance on a test dataset. This will provide metrics to assess how well the model classifies common factors and microcounseling skills.

5. **Baseline Comparisons**:
   - The `baselines/` directory contains alternative models for comparative analysis. Evaluate these models using the same training and evaluation procedures to benchmark the performance of the CFiCS model.

6. **Customization**:
   - **Model Architecture**: The `model.py` file defines the graph-based classification model. Modify this file to experiment with different architectures or to tailor the model to specific needs.
   - **Loss Functions and Metrics**: Customize loss calculations in `loss.py` and evaluation metrics in `metrics.py` to align with your project's requirements.
