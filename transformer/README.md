# Transformer Model

This repository contains the implementation of a Transformer model using PyTorch. The model includes an encoder, decoder, and various supporting modules such as multi-head attention and positional encoding.

![Transformer Architecture](.images/trans_arch.png)
*Reference: Attention is All You Need paper*

## Project Structure

```
.
├── pycache/
├── dataloader.py
├── decoder_block.py
├── decoder.py
├── encoder_block.py
├── encoder.py
├── images/
├── inference.py
├── model.pth
├── multi_head_attention.py
├── position_wise_feedforward.py
├── positional_encoding.py
├── README.md
├── train.py
├── transformer.py
```

### Files and Modules

- **data.txt**: Contains the dataset used for training the model.
- **dataloader.py**: Handles the loading and preprocessing of data.
- **decoder_block.py**: Implements the `DecoderBlock` class, which is a component of the decoder.
- **decoder.py**: Implements the `Decoder` class, which processes the target sequence.
- **encoder_block.py**: Implements the `EncoderBlock` class, which is a component of the encoder.
- **encoder.py**: Implements the `Encoder` class, which processes the input sequence.
- **multi_head_attention.py**: Implements the `MultiHeadAttention` class, which performs the attention mechanism.
- **position_wise_feedforward.py**: Implements the `PositionalWiseFeedForward` class, which is used in the encoder and decoder blocks.
- **positional_encoding.py**: Implements the `PositionalEncoding` class, which adds positional information to the input embeddings.
- **train.py**: Contains the training loop and functions for training the model.
- **transformer.py**: Implements the `Transformer` class, which combines the encoder and decoder.
- **inference.py**: Contains the code for running inference using the trained model.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/transformer-model.git
    cd transformer-model
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Prepare your dataset and place it in `data.txt`.

2. Train the model:
    ```sh
    python train.py
    ```

3. Run inference:
    ```sh
    python inference.py
    ```

## Model Components

### Encoder

The encoder is implemented in [encoder.py](encoder.py) and consists of multiple encoder blocks defined in [encoder_block.py](encoder_block.py).

### Decoder

The decoder is implemented in [decoder.py](decoder.py) and consists of multiple decoder blocks defined in [decoder_block.py](decoder_block.py).

### Multi-Head Attention

The multi-head attention mechanism is implemented in [multi_head_attention.py](multi_head_attention.py).

### Positional Encoding

Positional encoding is implemented in [positional_encoding.py](positional_encoding.py).

### Data Loading

Data loading and preprocessing are handled in [dataloader.py](dataloader.py).

### Training

The training loop is defined in [train.py](train.py). It uses the `train` function to train the model for a specified number of epochs.

```
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████| 196/196 [00:18<00:00, 10.63it/s, accuracy=92.5, loss=0.274]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:01<00:00, 26.18it/s]
Epoch: 1, Train loss: 1.786, Train acc: 0.564, Val loss: 0.188, Val acc: 0.954 Epoch time = 19.791s
Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [00:20<00:00,  9.58it/s, accuracy=98.9, loss=0.0365]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:01<00:00, 26.94it/s]
Epoch: 2, Train loss: 0.123, Train acc: 0.970, Val loss: 0.010, Val acc: 0.999 Epoch time = 21.850s
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [00:17<00:00, 10.94it/s, accuracy=99.6, loss=0.0137]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:01<00:00, 26.82it/s]
Epoch: 3, Train loss: 0.010, Train acc: 0.998, Val loss: 0.003, Val acc: 1.000 Epoch time = 19.275s
```

## Inference

To test the model with an example input:

```sh
python inference.py
```

Example result:

```
Input: helloworld
Output: _dlrowolleh_
```

## Notes

The model is used to train for reversing the input string.

## Reference

For a detailed guide on implementing your own Transformer model, refer to this article:
[https://towardsdatascience.com/a-complete-guide-to-write-your-own-transformers-29e23f371ddd](https://towardsdatascience.com/a-complete-guide-to-write-your-own-transformers-29e23f371ddd)
