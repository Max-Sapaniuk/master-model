# Fake News Classification Project

This project implements a fake news detection system using RoBERTa, a transformer-based model, fine-tuned on a custom dataset. It also includes scripts for converting the model to Core ML for integration with iOS applications.

## Project Structure

```
.
├── main.py                # Main script for training the model
├── convertModel.py        # Script for converting the trained model to Core ML
├── createVocab.py         # Script for saving the tokenizer vocabulary
├── dataset/               # Directory containing training and test datasets
│   ├── train.csv          # Training data
│   ├── test.csv           # Test data
│   └── ukr_processed.csv  # Additional dataset for Ukrainian language content
└── working/               # Directory for storing the trained model and logs
    ├── model/             # Trained model
    ├── logs/              # Training logs
    └── fake-news-model    # Saved trained model
```

## Requirements

- Python 3.11
- PyTorch
- Transformers
- Scikit-learn
- CoreMLTools

## Scripts Overview

### `main.py`
This script trains a fake news classifier using a RoBERTa model. The process includes:

1. **Loading Datasets**: It loads the training (`train.csv`), testing (`test.csv`), and an additional Ukrainian dataset (`ukr_processed.csv`).
2. **Preprocessing**: Missing values are filled, and relevant features (`title`, `author`, `text`) are combined into a single `content` column.
3. **Data Preparation**: The dataset is split into training and validation sets, and tokenized using RoBERTa's tokenizer.
4. **Training**: The model is fine-tuned with a weighted loss function to handle class imbalance, and early stopping is applied to avoid overfitting.
5. **Saving the Model**: The trained model is saved to the `./working/fake-news-model` directory.

### `convertModel.py`
This script converts the trained PyTorch model to Core ML format for use in iOS applications:

1. **Model and Tokenizer Loading**: It loads the trained model and tokenizer.
2. **Model Wrapping**: A custom wrapper is created to return only the logits from the model, necessary for Core ML conversion.
3. **Tracing and Saving**: The model is traced with a sample input, saved as a `.pt` file, and then converted to Core ML format (`FakeNewsClassifier.mlpackage`).

### `createVocab.py`
This script saves the tokenizer's vocabulary, which is needed for the iOS app to tokenize input text in the same way as during training:

1. **Tokenizer Loading**: It loads the pre-trained RoBERTa tokenizer.
2. **Vocabulary Saving**: The tokenizer's vocabulary is saved to a local directory.

## Training Process

1. **Prepare your dataset**: Place your CSV files (`train.csv`, `test.csv`, `ukr_processed.csv`) in the `dataset/` folder.
2. **Run `main.py`**: This script will preprocess the dataset, tokenize it, and start the training process. The model will be saved in the `working/` directory after training.
   ```bash
   python main.py
   ```
3. **Save the Trained Model**: After training, the model will be saved in the `working/fake-news-model` directory.

## Model Conversion

After training the model, you can convert it to Core ML format for use on iOS by running the following script:

```bash
python convertModel.py
```

This will generate a `.mlpackage` file, `FakeNewsClassifier.mlpackage`, that can be used in iOS applications.

## Vocabulary Export

To export the tokenizer's vocabulary, which is essential for tokenizing input text on iOS, run the following:

```bash
python createVocab.py
```

This will create a directory called `tokenizer/` containing the saved vocabulary.
