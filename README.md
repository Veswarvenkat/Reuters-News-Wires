# Reuters Newswire Multi-Class Classification

This project implements a neural network using TensorFlow and Keras to classify news wires from the Reuters dataset into **46 different topic categories**. It demonstrates a common approach for multi-class text classification.

## Project Overview

The notebook walks through the process of building a text classifier:
* **Data Loading:** Uses the built-in `tensorflow.keras.datasets.reuters` dataset, including 11,228 news wires and limiting the vocabulary to the top 10,000 words.
* **Preprocessing:**
    * Pads/truncates the sequences of word IDs to a fixed length (`max_newswire_length = 200`).
    * **One-hot encodes** the integer labels (0-45) into vectors suitable for multi-class classification (e.g., label 3 becomes `[0,0,0,1,0,...0]`).
* **Model Building:** Defines a `Sequential` model using `Embedding`, `GlobalAveragePooling1D`, and `Dense` layers.
* **Training:** Trains the model on the training data using `categorical_crossentropy` loss, monitoring performance on a validation split.
* **Evaluation:** Evaluates the final model's accuracy on the unseen test data.
* **Visualization:** Plots the training and validation accuracy and loss over epochs.
* **Prediction:** Shows an example of predicting the category for a single news wire.

---

## Dataset: Reuters Newswires

* **Source:** `tensorflow.keras.datasets.reuters`
* **Content:** Short news wires.
* **Size:** 8,982 training samples, 2,246 testing samples.
* **Vocabulary Size:** Limited to the top `vocab_size = 10000` most frequent words.
* **Classes:** 46 distinct topic categories (e.g., 'earn', 'acq', 'money-fx', 'grain', 'crude', etc.).

---

## Model Architecture

The neural network is built using the Keras `Sequential` API:

1.  **Embedding Layer:**
    * Input: Integer-encoded sequences (max length 200).
    * `input_dim`: 10,000 (vocabulary size).
    * `output_dim`: 128 (embedding vector size).
    * `input_length`: 200.
    * Converts word indices into dense vectors.
    * Output shape: `(batch_size, 200, 128)`

2.  **GlobalAveragePooling1D Layer:**
    * Averages the word embeddings across the sequence dimension.
    * Creates a single fixed-size vector representing the entire news wire.
    * Output shape: `(batch_size, 128)`

3.  **Dense Hidden Layer:**
    * Fully connected layer.
    * Units: 64
    * Activation: `relu`
    * Learns higher-level feature combinations.
    * Output shape: `(batch_size, 64)`

4.  **Dense Output Layer:**
    * Final classification layer.
    * Units: `num_classes` (46) - one for each category.
    * Activation: `softmax` - Outputs a probability distribution over the 46 classes, summing to 1.
    * Output shape: `(batch_size, 46)`

**Compilation:**
* **Optimizer:** `adam`
* **Loss Function:** `categorical_crossentropy` (standard for multi-class classification with one-hot encoded labels)
* **Metrics:** `accuracy`

---

## Requirements

You'll need the following Python libraries:

* `tensorflow`
* `numpy`
* `matplotlib`

Install them using pip:
```bash
pip install tensorflow numpy matplotlib
