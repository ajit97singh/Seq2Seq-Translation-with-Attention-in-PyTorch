# Seq2Seq Translation with Attention in PyTorch

This project implements a sequence-to-sequence (Seq2Seq) model with attention in PyTorch.
The model is trained to translate sentences from English to Russian using an encoder-decoder architecture and an attention mechanism.
The data for this project is a set of many thousands of English to Russian translation pairs.
This question on Open Data Stack Exchange pointed me to the open translation site https://tatoeba.org/ which has downloads available at https://tatoeba.org/eng/downloads - and better yet, someone did the extra work of splitting language pairs into individual text files here: https://www.manythings.org/anki/

The English to Russian pairs are too big to include in the repository, so download to data/eng-rus.txt before continuing. The file is a tab separated list of translation pairs:

#### Note: This is a generic project and can be used to create a translator between any language pair by simply editing he input_lang variable in main.py and downloading the corresponding file from above mentioned sources.

## Project Overview

This repository contains:
- A Seq2Seq model with attention for neural machine translation (NMT).
- Data preprocessing steps to tokenize and transform sentences.
- Training and evaluation scripts.
- Example results of model-generated translations.

### Key Components

- **Encoder**: RNN-based encoder that encodes the input sentence into a context vector.
- **Decoder with Attention**: RNN-based decoder with an attention layer that generates the translated sentence based on the encoder's output.
- **Attention Mechanism**: Helps the decoder focus on relevant parts of the input sentence, improving translation quality, especially on longer sentences.

## Requirements

- Python 3.8+
- PyTorch
- TorchText
- NLTK (for tokenization)
- numpy
- matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```


### The Seq2Seq Model
A Recurrent Neural Network, or RNN, is a network that operates on a sequence and uses its own output as input for subsequent steps.
A Sequence to Sequence network, or seq2seq network, or Encoder Decoder network, is a model consisting of two RNNs called the encoder and decoder. 
The encoder reads an input sequence and outputs a single vector, and the decoder reads that vector to produce an output sequence.


### Recommended Readings
I assume you have at least installed PyTorch, know Python, and understand Tensors:
- https://pytorch.org/ For installation instructions
- [Deep Learning with PyTorch: A 60 Minute Blitz to get started with PyTorch in general](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Learning PyTorch with Examples for a wide and deep overview](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- [PyTorch for Former Torch Users if you are former Lua Torch user](https://pytorch.org/tutorials/beginner/former_torchies_tutorial.html)

It would also be useful to know about Sequence to Sequence networks and how they work:
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)


