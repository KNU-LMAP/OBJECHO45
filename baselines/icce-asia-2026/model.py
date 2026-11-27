"""The four waveform models of the ICCE-Asia 2026 baselines.

ConvNeXt1D and ConvNeXt1D+BiGRU, each for the 4 ms stereo and 32 ms mono clips. The layer
order inside every `nn.Sequential` matches the paper's training code, because the released
weights are keyed by position (e.g. `encoder.3.enc.0.weight`).
"""
import torch
import torch.nn as nn

NUM_CLASSES = 45
DROPOUT = 0.1
HIDDEN_SIZE = 44    # GRU hidden size; bidirectional ==> pooled vector = 2 * 44

class EncoderBlock(nn.Module):
    """ConvNeXt residual block.

    `normalized_shape` is the number of time steps at this stage of the network.
    LayerNorm is built for one fixed length, so each block is tied to the stage
    it sits in.
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding="same", groups=32),  # groups=32 -> depthwise: one filter per channel
            nn.LayerNorm(normalized_shape),     # over the time axis
            nn.Conv1d(32, 128, kernel_size=1, stride=1, padding="same"),    # pointwise expand 32 --> 128
            nn.GELU(),
            nn.Conv1d(128, 32, kernel_size=1, stride=1, padding="same"),    # pointwise project 128 --> 32
        )

    def forward(self, x):
        return x + self.enc(x)


class AttentionPool(nn.Module):
    """Pool (N, L, H) --> (N, H) with a learned weight per time step; the weights sum to 1."""

    def __init__(self, hidden_size, attn_size=32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, attn_size),  # (N, L, H) --> (N, L, attn_size)
            nn.Tanh(),
            nn.Linear(attn_size, 1),    # (N, L, attn_size) --> (N, L, 1): one score per step
        )

    def forward(self, x):
        w = self.score(x)   # (N, L, 1); raw scores
        w = torch.softmax(w, dim=1)     # normalize over the L time steps so they sum to 1
        return (x * w).sum(dim=1)   # (N, L, H) * (N, L, 1) --> weighted sum over L --> (N, H)


class ConvNeXt1D4ms(nn.Module):
    """ConvNeXt1D for 4 ms stereo input: (N, 2, 1000) --> (N, n_classes)"""

    def __init__(self, n_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        # Front-end block (dense stem + strided convolution)
        self.frontend = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=1, padding="same"),       # (N, 2, 1000) --> (N, 32, 1000)
            nn.GELU(),
            nn.LayerNorm(1000),
            nn.Conv1d(32, 32, kernel_size=4, stride=4)  # (N, 32, 1000) --> (N, 32, 250)
        )

        # Encoder block ( (ConvNeXt1D + Downsampling) x3 )
        self.encoder = nn.Sequential(
            EncoderBlock(250),
            nn.LayerNorm(250),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),  # (N, 32, 250) --> (N, 32, 62)

            EncoderBlock(62),
            nn.LayerNorm(62),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 62) --> (N, 32, 15)

            EncoderBlock(15),
            nn.LayerNorm(15),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 15) --> (N, 32, 3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),       # (N, 32, 3) --> (N, 96)
            nn.Dropout(dropout),
            nn.Linear(96, n_classes),   # (N, 96) --> (N, n_classes)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.encoder(x)
        return self.classifier(x)


class ConvNeXt1D32ms(nn.Module):
    """ConvNeXt1D for 32 ms mono input: (N, 1, 8000) --> (N, n_classes)."""

    def __init__(self, n_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        # Front-end block (dense stem + strided convolution)
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding="same"),   # (N, 1, 8000) --> (N, 32, 8000)
            nn.GELU(),
            nn.LayerNorm(8000),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),  # (N, 32, 8000) --> (N, 32, 2000)
        )

        # Encoder block ( (ConvNeXt1D + Downsampling) x3 )
        self.encoder = nn.Sequential(
            EncoderBlock(2000),
            nn.LayerNorm(2000),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 2000) --> (N, 32, 500)

            EncoderBlock(500),
            nn.LayerNorm(500),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 500) --> (N, 32, 125)

            EncoderBlock(125),
            nn.LayerNorm(125),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 125) --> (N, 32, 31)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),               # (N, 32, 31) --> (N, 992)
            nn.Dropout(dropout),
            nn.Linear(992, n_classes)   # (N, 992) --> (N, n_classes)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.encoder(x)
        return self.classifier(x)


class ConvNeXt1DBiGRU4ms(nn.Module):
    """ConvNeXt1D+BiGRU for 4 ms stereo inputs: (N, 2, 1000) --> (N, n_classes)"""

    def __init__(self, n_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        # Front-end block (dense stem + strided convolution)
        self.frontend = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=1, padding="same"),  # (N, 2, 1000) --> (N, 32, 1000)
            nn.GELU(),
            nn.LayerNorm(1000),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 1000) --> (N, 32, 250)
        )

        # Encoder block (ConvNeXt1D x3, with *depthwise downsampling* between them)
        self.encoder = nn.Sequential(
            EncoderBlock(250),
            nn.LayerNorm(250),
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),     # (N, 32, 250) --> (N, 32, 62)

            EncoderBlock(62),
            nn.LayerNorm(62),
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),     # (N, 32, 62) --> (N, 32, 15)

            EncoderBlock(15),
            nn.LayerNorm(15),       # no more downsampling
        )

        # Recurrent path
        self.gru = nn.GRU(input_size=32, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)     # (N, 15, 32) --> (N, 15, HIDDEN_SIZE * 2)
        self.pool = AttentionPool(hidden_size=HIDDEN_SIZE*2)        # (N, 15, HIDDEN_SIZE * 2) --> (N, HIDDEN_SIZE * 2)

        # Convolutional path
        self.convhead = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),  # depthwise conv: (N, 32, 15) --> (N, 32, 3)
            nn.Flatten(),                                           # (N, 32, 3) --> (N, 96)
        )

        # Concatenation head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_SIZE * 2 + 96, n_classes),           # (N, 184) --> (N, n_classes)
        )

    def forward(self, x):
        x = self.frontend(x)        # (N, 2, 1000) --> (N, 32, 250)
        x = self.encoder(x)         # (N, 32, 250) --> (N, 32, 15)

        # Recurrent path
        gru_in = x.transpose(1, 2)      # transpose swaps C and L: (N, 32, 15) --> (N, 15, 32)
        gru_out, _ = self.gru(gru_in)   # (N, 15, 32) --> (N, 15, HIDDEN_SIZE * 2)
        rnn_out = self.pool(gru_out)    # (N, 15, HIDDEN_SIZE * 2) --> (N, HIDDEN_SIZE * 2)

        # Convolutional path
        cnn_out = self.convhead(x)      # (N, 32, 15) --> (N, 96)

        # Concatenate the RNN and CNN paths
        concat = torch.cat([rnn_out, cnn_out], dim=1)   # (N, HIDDEN_SIZE * 2 + 96): (N, 184)

        return self.head(concat)    # (N, 184) --> (N, n_classes)


class ConvNeXt1DBiGRU32ms(nn.Module):
    """ConvNeXt1D+BiGRU for 32 ms mono inputs: (N, 1, 8000) --> (N, n_classes)"""

    def __init__(self, n_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        # Front-end block (dense stem + strided convolution)
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding="same"),  # (N, 1, 8000) --> (N, 32, 8000)
            nn.GELU(),
            nn.LayerNorm(8000),
            nn.Conv1d(32, 32, kernel_size=4, stride=4),     # (N, 32, 8000) --> (N, 32, 2000)
        )

        # Encoder block (ConvNeXt1D x3, with *depthwise downsampling* after each of them, followed by a final normalization)
        self.encoder = nn.Sequential(
            EncoderBlock(2000),
            nn.LayerNorm(2000),
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),     # (N, 32, 2000) --> (N, 32, 500)

            EncoderBlock(500),
            nn.LayerNorm(500),
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),     # (N, 32, 500) --> (N, 32, 125)

            EncoderBlock(125),
            nn.LayerNorm(125),
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),      # (N, 32, 125) --> (N, 32, 31)
            nn.LayerNorm(31),
        )

        # Recurrent path
        self.gru = nn.GRU(input_size=32, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)     # (N, 31, 32) --> (N, 31, 2 * HIDDEN_SIZE)
        self.pool = AttentionPool(hidden_size=HIDDEN_SIZE*2)      # (N, 31, 2 * HIDDEN_SIZE) --> (N, 2 * HIDDEN_SIZE)

        # Convolutional path
        self.convhead = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=4, stride=4, groups=32),      # (N, 32, 31) --> (N, 32, 7)
            nn.Flatten(),       # (N, 32, 7) --> (N, 224)
        )

        # Concatenation head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_SIZE*2 + 224, n_classes),      # (N, HIDDEN_SIZE*2 + 224) --> (N, n_classes)
        )

    def forward(self, x):
        x = self.frontend(x)    # (N, 1, 8000) --> (N, 32, 2000)
        x = self.encoder(x)     # (N, 32, 2000) --> (N, 32, 31)

        # Recurrent path
        gru_in = x.transpose(1, 2)  # swaps the position of C and L: (N, 32, 31) --> (N, 31, 32)
        gru_out, _ = self.gru(gru_in)   # (N, 31, 32) --> (N, 31, 2 * HIDDEN_SIZE)
        rnn_out = self.pool(gru_out)    # (N, 31, 2 * HIDDEN_SIZE) --> (N, 2 * HIDDEN_SIZE)

        # Convolutional path
        cnn_out = self.convhead(x)      # (N, 32, 31) --> (N, 224)

        # Concatenation
        concat = torch.cat([rnn_out, cnn_out], dim=1)   # (N, 2*HIDDEN_SIZE + 224): (N, 312)

        return self.head(concat)    # (N, 312) --> (N, n_classes)


# Model registry: a lookup table from each model's name in the paper to its class
MODELS = {
    "ConvNeXt1D": ConvNeXt1D4ms,
    "ConvNeXt1D (32 ms)": ConvNeXt1D32ms,
    "ConvNeXt1D+BiGRU": ConvNeXt1DBiGRU4ms,
    "ConvNeXt1D+BiGRU (32 ms)": ConvNeXt1DBiGRU32ms,
}
