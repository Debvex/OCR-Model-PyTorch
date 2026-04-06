"""
ASTER: Attentional Scene Text Recognition
Attention-based Decoder with Bahdanau attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class AttentionMechanism(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AttentionMechanism, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        # Linear layers for attention
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attention_vector = nn.Linear(attention_dim, 1)

    def forward(self, encoder_features, decoder_hidden):
        """
        Calculate attention weights and context vector

        Args:
            encoder_features: (B, T, encoder_dim) - features from encoder at all time steps
            decoder_hidden: (B, decoder_dim) - current decoder hidden state

        Returns:
            context: (B, encoder_dim) - weighted sum of encoder features
            attention_weights: (B, T) - attention weights for each encoder position
        """
        B, T, _ = encoder_features.size()

        # Project encoder features
        encoder_proj = self.encoder_proj(encoder_features)  # (B, T, attention_dim)

        # Project decoder hidden state and expand
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(
            1
        )  # (B, 1, attention_dim)

        # Calculate attention energies
        energy = torch.tanh(encoder_proj + decoder_proj)  # (B, T, attention_dim)
        attention_weights = self.attention_vector(energy).squeeze(2)  # (B, T)
        attention_weights = F.softmax(attention_weights, dim=1)  # (B, T)

        # Calculate context vector as weighted sum
        attention_weights_expanded = attention_weights.unsqueeze(1)  # (B, 1, T)
        context = torch.bmm(attention_weights_expanded, encoder_features).squeeze(
            1
        )  # (B, encoder_dim)

        return context, attention_weights


class AttentionDecoder(nn.Module):
    """
    Attention-based Decoder with LSTM
    """

    def __init__(
        self,
        num_classes,
        encoder_dim,
        embedding_dim,
        decoder_dim,
        attention_dim,
        dropout=0.5,
    ):
        super(AttentionDecoder, self).__init__()

        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        # Embedding layer
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        # Attention mechanism
        self.attention = AttentionMechanism(encoder_dim, decoder_dim, attention_dim)

        # LSTM cell
        # Input: embedding + context vector
        self.lstm = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)

        # Output projection
        self.output_projection = nn.Linear(decoder_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.uniform_(param, -0.1, 0.1)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.output_projection.weight, -0.1, 0.1)
        nn.init.constant_(self.output_projection.bias, 0)

    def forward_step(self, input_token, hidden_state, cell_state, encoder_features):
        """
        Single forward step

        Args:
            input_token: (B,) - current input token
            hidden_state: (B, decoder_dim) - current hidden state
            cell_state: (B, decoder_dim) - current cell state
            encoder_features: (B, T, encoder_dim) - encoder features

        Returns:
            output: (B, num_classes) - prediction logits
            hidden_state: (B, decoder_dim) - updated hidden state
            cell_state: (B, decoder_dim) - updated cell state
            attention_weights: (B, T) - attention weights
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (B, embedding_dim)
        embedded = self.dropout(embedded)

        # Calculate attention context
        context, attention_weights = self.attention(encoder_features, hidden_state)

        # Concatenate embedding and context
        lstm_input = torch.cat(
            [embedded, context], dim=1
        )  # (B, embedding_dim + encoder_dim)

        # LSTM step
        hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))

        # Output projection
        hidden_state = self.dropout(hidden_state)
        output = self.output_projection(hidden_state)  # (B, num_classes)

        return output, hidden_state, cell_state, attention_weights

    def forward(self, encoder_features, targets=None, teacher_forcing_ratio=0.5):
        """
        Forward pass

        Args:
            encoder_features: (B, T, encoder_dim) - encoder features
            targets: (B, max_length) - ground truth labels (optional, for training)
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            outputs: (B, max_length, num_classes) - prediction logits
            attention_weights: list of (B, T) - attention weights at each step
        """
        B = encoder_features.size(0)
        device = encoder_features.device

        # Determine max length
        if targets is not None:
            max_length = targets.size(1)
        else:
            max_length = 25  # Default max length for inference

        # Initialize LSTM states
        # Initialize from mean of encoder features
        encoder_mean = encoder_features.mean(dim=1)  # (B, encoder_dim)
        hidden_state = torch.zeros(B, self.decoder_dim).to(device)
        cell_state = torch.zeros(B, self.decoder_dim).to(device)

        # Start token (assuming 0 is start token)
        input_token = torch.zeros(B, dtype=torch.long).to(device)

        outputs = []
        attention_weights_list = []

        for t in range(max_length):
            # Forward step
            output, hidden_state, cell_state, attention_weights = self.forward_step(
                input_token, hidden_state, cell_state, encoder_features
            )

            outputs.append(output)
            attention_weights_list.append(attention_weights)

            # Teacher forcing
            if targets is not None and random.random() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                # Use predicted token
                input_token = output.argmax(dim=1)

        outputs = torch.stack(outputs, dim=1)  # (B, max_length, num_classes)

        return outputs, attention_weights_list

    def greedy_decode(
        self, encoder_features, max_length=25, start_token=0, end_token=1
    ):
        """
        Greedy decoding for inference

        Args:
            encoder_features: (B, T, encoder_dim) - encoder features
            max_length: maximum decoding length
            start_token: start token index
            end_token: end token index

        Returns:
            predictions: (B, max_length) - predicted token indices
            attention_weights: list of (B, T) - attention weights
        """
        B = encoder_features.size(0)
        device = encoder_features.device

        # Initialize LSTM states
        hidden_state = torch.zeros(B, self.decoder_dim).to(device)
        cell_state = torch.zeros(B, self.decoder_dim).to(device)

        # Start token
        input_token = torch.full((B,), start_token, dtype=torch.long).to(device)

        predictions = []
        attention_weights_list = []

        for t in range(max_length):
            # Forward step
            output, hidden_state, cell_state, attention_weights = self.forward_step(
                input_token, hidden_state, cell_state, encoder_features
            )

            # Get predicted token
            predicted_token = output.argmax(dim=1)
            predictions.append(predicted_token)
            attention_weights_list.append(attention_weights)

            # Check for end token
            if (predicted_token == end_token).all():
                break

            # Use predicted token as next input
            input_token = predicted_token

        predictions = torch.stack(predictions, dim=1)  # (B, num_steps)

        return predictions, attention_weights_list


class AttentionDecoderV2(nn.Module):
    """
    Attention Decoder with separate attention LSTM and character LSTM
    Based on ASTER paper architecture
    """

    def __init__(
        self,
        num_classes,
        encoder_dim,
        embedding_dim,
        hidden_dim,
        attention_dim,
        dropout=0.5,
    ):
        super(AttentionDecoderV2, self).__init__()

        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Embedding
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        # Attention LSTM - processes previous context and embedding
        self.attention_lstm = nn.LSTMCell(encoder_dim + embedding_dim, hidden_dim)

        # Attention mechanism
        self.attention = AttentionMechanism(encoder_dim, hidden_dim, attention_dim)

        # Character LSTM - combines context with embedding
        self.character_lstm = nn.LSTMCell(encoder_dim + embedding_dim, hidden_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in [self.attention_lstm, self.character_lstm]:
            for name, param in module.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.output_projection.weight, -0.1, 0.1)
        nn.init.constant_(self.output_projection.bias, 0)

    def forward_step(
        self,
        input_token,
        prev_context,
        prev_hidden1,
        prev_cell1,
        prev_hidden2,
        prev_cell2,
        encoder_features,
    ):
        """
        Single forward step with dual LSTM architecture

        Args:
            input_token: (B,) - current input token
            prev_context: (B, encoder_dim) - previous context vector
            prev_hidden1, prev_cell1: states for attention LSTM
            prev_hidden2, prev_cell2: states for character LSTM
            encoder_features: (B, T, encoder_dim) - encoder features

        Returns:
            output: (B, num_classes) - prediction logits
            context: (B, encoder_dim) - new context vector
            updated states for both LSTMs
            attention_weights: (B, T)
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (B, embedding_dim)
        embedded = self.dropout(embedded)

        # Attention LSTM
        attention_input = torch.cat([prev_context, embedded], dim=1)
        hidden1, cell1 = self.attention_lstm(
            attention_input, (prev_hidden1, prev_cell1)
        )
        hidden1 = self.dropout(hidden1)

        # Calculate attention
        context, attention_weights = self.attention(encoder_features, hidden1)

        # Character LSTM
        character_input = torch.cat([context, embedded], dim=1)
        hidden2, cell2 = self.character_lstm(
            character_input, (prev_hidden2, prev_cell2)
        )
        hidden2 = self.dropout(hidden2)

        # Output projection
        output = self.output_projection(hidden2)  # (B, num_classes)

        return output, context, hidden1, cell1, hidden2, cell2, attention_weights

    def forward(self, encoder_features, targets=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        """
        B = encoder_features.size(0)
        device = encoder_features.device

        max_length = targets.size(1) if targets is not None else 25

        # Initialize states
        encoder_mean = encoder_features.mean(dim=1)
        prev_context = encoder_mean

        hidden1 = torch.zeros(B, self.hidden_dim).to(device)
        cell1 = torch.zeros(B, self.hidden_dim).to(device)
        hidden2 = torch.zeros(B, self.hidden_dim).to(device)
        cell2 = torch.zeros(B, self.hidden_dim).to(device)

        input_token = torch.zeros(B, dtype=torch.long).to(device)

        outputs = []
        attention_weights_list = []

        for t in range(max_length):
            output, prev_context, hidden1, cell1, hidden2, cell2, attention_weights = (
                self.forward_step(
                    input_token,
                    prev_context,
                    hidden1,
                    cell1,
                    hidden2,
                    cell2,
                    encoder_features,
                )
            )

            outputs.append(output)
            attention_weights_list.append(attention_weights)

            if targets is not None and random.random() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = output.argmax(dim=1)

        outputs = torch.stack(outputs, dim=1)

        return outputs, attention_weights_list


if __name__ == "__main__":
    print("Testing Attention Decoder...")

    # Test parameters
    B, T, encoder_dim = 2, 25, 512
    num_classes = 37  # 26 letters + 10 digits + blank
    embedding_dim = 256
    decoder_dim = 256
    attention_dim = 256
    max_length = 25

    # Create dummy encoder features
    encoder_features = torch.randn(B, T, encoder_dim)

    # Test AttentionDecoder
    print("\n1. Testing AttentionDecoder:")
    decoder = AttentionDecoder(
        num_classes, encoder_dim, embedding_dim, decoder_dim, attention_dim
    )
    outputs, attention_weights = decoder(encoder_features, max_length=max_length)
    print(f"Encoder features shape: {encoder_features.shape}")
    print(f"Output shape: {outputs.shape}")
    print(
        f"Expected: (B, max_length, num_classes) = ({B}, {max_length}, {num_classes})"
    )
    print(f"Attention weights length: {len(attention_weights)}")
    print(f"First attention weights shape: {attention_weights[0].shape}")

    # Test AttentionDecoderV2
    print("\n2. Testing AttentionDecoderV2:")
    decoder_v2 = AttentionDecoderV2(
        num_classes, encoder_dim, embedding_dim, decoder_dim, attention_dim
    )
    outputs_v2, attention_weights_v2 = decoder_v2(
        encoder_features, max_length=max_length
    )
    print(f"Output shape: {outputs_v2.shape}")
    print(
        f"Expected: (B, max_length, num_classes) = ({B}, {max_length}, {num_classes})"
    )

    # Test greedy decode
    print("\n3. Testing greedy_decode:")
    predictions, _ = decoder.greedy_decode(encoder_features, max_length=25)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Expected: (B, max_length) = ({B}, {max_length})")

    print("\nAttention Decoder test passed!")
