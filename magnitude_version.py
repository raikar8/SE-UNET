import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False)

# Create a sample spectrogram
batch_size = 2
freq_bins = 257  # For n_fft=512
time_frames = 128
input_spec = torch.randn(batch_size, 1, freq_bins, time_frames)

print(f"Input shape: {input_spec.shape}")
# Output: torch.Size([2, 1, 257, 128])

# Apply convolution
output = conv(input_spec)
print(f"Output shape: {output.shape}")
# Output: torch.Size([2, 64, 257, 128])

print(f"\nWhat happened?")
print(f"- Batch size: {batch_size} → {batch_size} (unchanged)")
print(f"- Channels: 1 → 64 (learned 64 different feature detectors)")
print(f"- Frequency: {freq_bins} → {output.shape[2]} (preserved by padding)")
print(f"- Time: {time_frames} → {output.shape[3]} (preserved by padding)")


class ConvBnRelU(nn.Module):
    """Single convolution with batch norm and activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

# Test it
block = ConvBnRelU(in_channels=1, out_channels=64)
output = block(input_spec)
#utput = block(output)

print(f"After ConvBnReLU:")
print(f"  Shape: {output.shape}")
print(f"  Mean: {output.mean().item():.4f}")
print(f"  Std: {output.std().item():.4f}")
print(f"  Min: {output.min().item():.4f}")
print(f"  Max: {output.max().item():.4f}")


class DoubleConvBlock(nn.Module):
    """
        Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU

        This is the fundamental building block of U-Net.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Example:
             block = DoubleConvBlock(in_channels=1, out_channels=64)
             x = torch.randn(2, 1, 257, 128)  # (B, C, F, T)
             y = block(x)
             print(y.shape)  # torch.Size([2, 64, 257, 128])
    """

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.conv1 = ConvBnRelU(in_channels, out_channels)
        self.conv2 = ConvBnRelU(out_channels, out_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x

# Let's test it!
print("=" * 70)
print("Testing DoubleConvBlock")
print("=" * 70)

# Create a sample spectrogram (magnitude only)
batch_size = 2
freq_bins = 257  # For n_fft=512
time_frames = 128
input_spec = torch.randn(batch_size, 1, freq_bins, time_frames)

print(f"\nInput spectrogram shape: {input_spec.shape}")
print(f"  - Batch size: {batch_size}")
print(f"  - Channels: 1 (magnitude only)")
print(f"  - Frequency bins: {freq_bins}")
print(f"  - Time frames: {time_frames}")

# Create DoubleConvBlock
double_conv = DoubleConvBlock(in_channels=1, out_channels=64)

# Forward pass
output = double_conv(input_spec)

print(f"\nOutput shape: {output.shape}")
print(f"  - Batch size: {batch_size} (unchanged)")
print(f"  - Channels: 64 (learned 64 feature maps)")
print(f"  - Frequency bins: {freq_bins} (preserved by padding)")
print(f"  - Time frames: {time_frames} (preserved by padding)")

# Count parameters
total_params = sum(p.numel() for p in double_conv.parameters())
print(f"\nTotal parameters in DoubleConvBlock: {total_params:,}")

print("\nBreakdown:")
print(f"  Conv1 weights: {double_conv.conv1.conv.weight.shape}")
print(f"    → 64 filters × 1 input channel × 3 × 3 = {64*1*3*3:,} weights")
print(f"  Conv2 weights: {double_conv.conv2.conv.weight.shape}")
print(f"    → 64 filters × 64 input channels × 3 × 3 = {64*64*3*3:,} weights")


class EncoderBlock(nn.Module):
    """
        Encoder block: DoubleConv → MaxPool

        This combines feature extraction (DoubleConv) with
        spatial downsampling (MaxPool).

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            - Pooled output: Downsampled features for next encoder level
            - Skip features: Full-resolution features for skip connection

        Example:
             encoder = EncoderBlock(in_channels=1, out_channels=64)
             x = torch.randn(2, 1, 257, 128)
             pooled, skip = encoder(x)
             print(f"Pooled: {pooled.shape}")    # [2, 64, 128, 64] (half size)
             print(f"Skip: {skip.shape}")        # [2, 64, 257, 128] (full size)
        """

    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        # Feature extraction without changing spatial dimensions
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

        # Downsampling: 2×2 max pooling with stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x:torch.Tensor):

        features = self.double_conv(x)
        pooled_feature = self.pool(features)

        return  pooled_feature,features

# Test the EncoderBlock
print("\n" + "=" * 70)
print("Testing EncoderBlock")
print("=" * 70)

encoder = EncoderBlock(in_channels=1, out_channels=64)

x = torch.randn(2, 1, 257, 128)
pooled,skip = encoder(x)

print(f"\nInput shape: {x.shape}")
print(f"\nOutput after EncoderBlock:")
print(f"  Pooled (for next encoder): {pooled.shape}")
print(f"    Frequency: 257 → {pooled.shape[2]} (halved)")
print(f"    Time: 128 → {pooled.shape[3]} (halved)")
print(f"\n  Skip features (preserved): {skip.shape}")
print(f"    Same as input spatial dimensions!")
print(f"\nWhy do we return skip?")
print(f"  → We'll concatenate it with decoder later")
print(f"  → Preserves high-resolution detail from early layers")

class DecoderBlock(nn.Module):
    """
       Decoder block: Upsample → Concatenate → DoubleConv

       This is where the skip connections magic happens!

       Args:
           in_channels (int): Channels from previous decoder layer
           skip_channels (int): Channels from skip connection (from encoder)
           out_channels (int): Output channels

       Example:
            decoder = DecoderBlock(
           ...     in_channels=512,
           ...     skip_channels=256,
           ...     out_channels=256
           ... )
            x_from_decoder = torch.randn(2, 512, 64, 32)  # Previous layer output
            x_from_encoder = torch.randn(2, 256, 128, 64)  # Skip connection
            output = decoder(x_from_decoder, x_from_encoder)
            print(output.shape)  # torch.Size([2, 256, 128, 64])
       """
    def __init__(self, in_channels:int, skip_channels:int, out_channels:int):
        super().__init__()

        # Transposed convolution: upsample while reducing channels
        # kernel_size=2, stride=2 doubles the spatial dimensions

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

        # After upsampling, we'll concatenate with skip features
        # So input to DoubleConv will be: (upsampled_channels + skip_channels)
        concat_channels = (in_channels // 2) + skip_channels

        # Process concatenated features
        self.double_conv = DoubleConvBlock(concat_channels, out_channels)


    def forward(self, x:torch.Tensor, skip:torch.Tensor) -> torch.Tensor:
        """
                Args:
                    x (torch.Tensor): Features from previous decoder layer
                    skip (torch.Tensor): Skip connection from encoder

                Returns:
                    torch.Tensor: Decoded features
         """

        # Upsample (transpose convolution)
        x = self.upsample(x)  # Double spatial dimensions, halve channels

        # Handle potential size mismatch (can happen with odd dimensions)
        if x.shape != skip.shape:
            x = self._align_dimensions(x, skip)

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)

        # Process concatenated features
        x = self.double_conv(x)

        return x


    @staticmethod
    def _align_dimensions(x:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """
                Handle size mismatch due to odd dimensions.

                Example: 257 / 2 = 128.5 → becomes 128, then upsamples to 257 or 256?
                This method aligns x to match target spatial dimensions.
        """

        # Calculate differences
        diff_h = target.size(2) - x.size(2)  # Frequency dimension
        diff_w = target.size(3) - x.size(3)  # Time dimension

        # Pad if needed
        x = torch.nn.functional.pad(
            x,
            [diff_w // 2, diff_w - diff_w // 2,
             diff_h // 2, diff_h - diff_h // 2]
        )

        return x

# Test the DecoderBlock
print("\n" + "=" * 70)
print("Testing DecoderBlock")
print("=" * 70)

decoder = DecoderBlock(
    in_channels=512,
    skip_channels=256,
    out_channels=256
)

x_decoder = torch.randn(2, 512, 32, 16)    # From previous decoder layer
x_skip = torch.randn(2, 256, 64, 32)       # From encoder skip connection

output = decoder(x_decoder, x_skip)

print(f"\nInput from decoder: {x_decoder.shape}")
print(f"Skip connection from encoder: {x_skip.shape}")
print(f"\nStep-by-step:")
print(f"  1. Upsample 512 channels at (32×16):")
print(f"     → ConvTranspose2d(512, 256, k=2, s=2)")
print(f"     → (2, 256, 64, 32)")
print(f"  2. Concatenate with skip (2, 256, 64, 32):")
print(f"     → torch.cat → (2, 512, 64, 32)")
print(f"  3. DoubleConv(512 → 256):")
print(f"     → (2, 256, 64, 32)")
print(f"\nFinal output: {output.shape}")
print(f"  ✓ Spatial dimensions match skip connection")
print(f"  ✓ Channels reduced from 512 → 256")


# ============================================================================
# Testing
# ============================================================================

print("=" * 70)
print("Testing All Components - Complete U-Net Blocks")
print("=" * 70)

batch_size = 2
freq_bins = 257
time_frames = 128

# Input
x = torch.randn(batch_size, 1, freq_bins, time_frames)

# ============ Test 1: ConvBnReLU ============
print("\n--- Test 1: ConvBnReLU ---")
conv = ConvBnRelU(1, 64)
y = conv(x)
print(f"Input:  {x.shape}")
print(f"Output: {y.shape}")
print(f"✓ Works!")

# ============ Test 2: DoubleConvBlock ============
print("\n--- Test 2: DoubleConvBlock ---")
double_conv = DoubleConvBlock(1, 64)
y = double_conv(x)
print(f"Input:  {x.shape}")
print(f"Output: {y.shape}")
print(f"✓ Works! (Two conv layers)")

# ============ Test 3: EncoderBlock ============
print("\n--- Test 3: EncoderBlock ---")
encoder = EncoderBlock(1, 64)
pooled, skip = encoder(x)
print(f"Input:    {x.shape}")
print(f"Pooled:   {pooled.shape}")
print(f"Skip:     {skip.shape}")
print(f"✓ Works! (Returns both outputs)")

# ============ Test 4: DecoderBlock ============
print("\n--- Test 4: DecoderBlock ---")
decoder = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)

x_decoder = torch.randn(batch_size, 512, 32, 16)
skip_features = torch.randn(batch_size, 256, 64, 32)

print(f"Decoder input: {x_decoder.shape}")
print(f"Skip input:    {skip_features.shape}")

# ✓ CORRECT: Pass TWO arguments
output = decoder(x_decoder, skip_features)

print(f"Output:        {output.shape}")
print(f"✓ Works! (Accepts TWO arguments)")

# ============ Test 5: Stacking Encoders and Decoders ============
print("\n--- Test 5: Complete Encoder-Decoder Chain ---")

# Encoding path
print("\nEncoding:")
x = torch.randn(batch_size, 1, freq_bins, time_frames)
print(f"Input: {x.shape}")


encoders = nn.ModuleList([EncoderBlock(1,64),EncoderBlock(64,128),EncoderBlock(128,256)])

skip_connections =[]

for i, enc in enumerate(encoders):
    x,skip = enc(x)
    skip_connections.append(skip)
    print(f"Encoder {i + 1}: {x.shape}, skip: {skip.shape}")


# Bottleneck
print(f"\nBottleneck: {x.shape}")
bottleneck = DoubleConvBlock(256, 512)
x = bottleneck(x)
print(f"After bottleneck: {x.shape}")



# Decoding path
print("\nDecoding:")
decoders = nn.ModuleList([
    DecoderBlock(512, 256, 256),
    DecoderBlock(256, 128, 128),
    DecoderBlock(128, 64, 64),
])

# Reverse skip connections for decoder
skip_connections = skip_connections[::-1]

for i, dec in enumerate(decoders):
    # ✓ CORRECT: Pass TWO arguments to decoder
    x = dec(x, skip_connections[i])
    print(f"Decoder {i+1}: {x.shape}")

print(f"\n✓ All tests passed! No TypeError!")
print("=" * 70)

from typing import List, Tuple

class Encoder(nn.Module):
    """
        Complete encoder pathway of U-Net.

        Progressively downsamples while increasing channels.

        Args:
            in_channels (int): Input channels (1 for magnitude)
            encoder_channels (List[int]): Channel sizes at each level
                                          e.g., [64, 128, 256, 512]

        Returns:
            - Bottleneck features (lowest resolution, highest channels)
            - List of skip connections (for decoder)

        Example:
             encoder = Encoder(in_channels=1, encoder_channels=[64, 128, 256, 512])
             x = torch.randn(2, 1, 257, 128)
             bottleneck, skips = encoder(x)
             print(f"Bottleneck: {bottleneck.shape}")  # [2, 512, 32, 16]
             print(f"Skip 0: {skips[0].shape}")        # [2, 64, 257, 128]
    """

    def __init__(self, in_channels:int, encoder_channels:List[int]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()

        # Create encoder blocks for each level
        # Input: in_channels → First encoder level: encoder_channels[0]
        # Then: encoder_channels[i] → encoder_channels[i+1]

        channels = [in_channels] + encoder_channels

        for i in range(len(encoder_channels)):
            self.encoder_blocks.append(EncoderBlock(channels[i], channels[i+1]))


    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
                Encode input through all encoder levels.

                Args:
                    x (torch.Tensor): Input spectrogram (B, 1, F, T)

                Returns:
                    Tuple containing:
                    - bottleneck: Final encoded features (lowest resolution)
                    - skip_connections: List of features from each encoder level (for decoder)
        """

        skip_connections = []

        print("\n" + "=" * 70)
        print("ENCODER FORWARD PASS")
        print("=" * 70)
        print(f"Input: {x.shape}")

        for i,encoder_block in enumerate(self.encoder_blocks):
            x,skip = encoder_block(x)
            skip_connections.append(skip)

            print(f"Encoder Level {i + 1}:")
            print(f"  Pooled: {x.shape} (for next level)")
            print(f"  Skip: {skip.shape} (for decoder)")

        # x now contains the bottleneck features
        bottleneck = x

        print(f"\nBottleneck: {bottleneck.shape}")
        print("=" * 70)

        return bottleneck, skip_connections

# Test the Encoder
print("\n\nTesting Encoder Architecture")
print("="*70)

batch_size = 2
freq_bins = 257
time_frames = 128

# Create encoder with 4 levels
encoder = Encoder(
    in_channels=1,
    encoder_channels=[64, 128, 256, 512]
)

# Forward pass
x = torch.randn(batch_size, 1, freq_bins, time_frames)
bottleneck, skip_connections = encoder(x)

print(f"\nSummary:")
print(f"  Input: {x.shape}")
print(f"  Bottleneck: {bottleneck.shape}")
print(f"  Number of skip connections: {len(skip_connections)}")
for i, skip in enumerate(skip_connections):
    print(f"    Skip {i}: {skip.shape}")

class Bottleneck(nn.Module):
    """
        Bottleneck layer at the center of U-Net.

        This is where the encoder meets the decoder.
        We apply additional DoubleConvBlocks to refine the features
        before passing them to the decoder.

        Args:
            in_channels (int): Channels from encoder
            out_channels (int): Channels for decoder

        Example:
            >>> bottleneck = Bottleneck(in_channels=512, out_channels=512)
            >>> x = torch.randn(2, 512, 16, 8)
            >>> y = bottleneck(x)
            >>> print(y.shape)  # torch.Size([2, 512, 16, 8])
    """
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        # Usually in_channels == out_channels for bottleneck
        # But we allow flexibility
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
                Process bottleneck features.

                Args:
                    x: Features from encoder

                Returns:
                    Refined features at same resolution
        """

        return self.double_conv(x)

# Test Bottleneck
print("\n\nTesting Bottleneck")
print("="*70)

bottleneck = Bottleneck(in_channels=512, out_channels=512)

x_bottleneck = torch.randn(batch_size, 512, 16, 8)
print(f"Input: {x_bottleneck.shape}")

y_bottleneck = bottleneck(x_bottleneck)
print(f"Output: {y_bottleneck.shape}")
print(f"✓ Bottleneck preserves spatial dimensions")

from typing import List, Tuple, Union

class Decoder(nn.Module):
    """
    Complete decoder pathway of U-Net.

    Progressively upsamples while decreasing channels.
    Concatenates skip connections from encoder at each level.

    Args:
        decoder_channels (List[int]): Channel sizes at each level
                                      e.g., [512, 256, 128, 64]

    Example:
         decoder = Decoder(decoder_channels=[512, 256, 128, 64])
         x = torch.randn(2, 512, 16, 8)
         skip_connections = [...]  # From encoder
         y = decoder(x, skip_connections)
         print(y.shape)  # torch.Size([2, 64, 257, 128])
    """

    def __init__(self, decoder_channels:List[int]):
        super().__init__()

        self.decoder_blocks = nn.ModuleList()
        # Create decoder blocks
        # Each level: in_channels, skip_channels, out_channels
        # The skip_channels is the same as decoder_channels at that level

        for i in range(len(decoder_channels)-1):
            self.decoder_blocks.append(DecoderBlock(in_channels=decoder_channels[i],skip_channels=decoder_channels[i], out_channels=decoder_channels[i+1]))


    def forward(self, x:torch.Tensor, skip_connections_encoder:List[torch.Tensor]) -> torch.Tensor:
        """
                Decode through all decoder levels.

                Args:
                    x (torch.Tensor): Bottleneck features
                    skip_connections (List[torch.Tensor]): Skip features from encoder (in order)

                Returns:
                    torch.Tensor: Decoded output at original resolution
                    :param x:
                    :param skip_connections_encoder:
        """

        # Reverse skip connections (decoder goes in reverse order)

        # ✓ Type checking and conversion

        print("=" * 70)

        print(type(skip_connections_encoder))
        print(type(skip_connections_encoder[0]))
        # print(skip_connections)

        print("=" * 70)
        if not isinstance(skip_connections_encoder, list):
            raise TypeError(
                f"❌ ERROR: skip_connections must be a List[torch.Tensor]\n"
                f"   Got: {type(skip_connections_encoder)}\n"
                f"   Make sure Encoder returns skip_connections as a LIST!\n"
                f"   Correct: return bottleneck, skip_connections_list\n"
                f"   Wrong: return bottleneck, single_tensor"
            )

        skip_connections_encoder = skip_connections_encoder[::-1]

        print("\n" + "=" * 70)
        print("DECODER FORWARD PASS")
        print("=" * 70)
        print(f"Bottleneck: {x.shape}")

        for i,decoder_block in enumerate(self.decoder_blocks):

            skip_connection = skip_connections_encoder[i]

            x = decoder_block(x, skip_connection)
            print(f"Decoder Level {i + 1}:")
            print(f"  Output: {x.shape}")

        print("=" * 70)

        return x

# Test Decoder
print("\n\nTesting Decoder")
print("="*70)

decoder = Decoder(decoder_channels=[512, 256, 128, 64])

# Simulate encoder outputs
x_bottleneck = torch.randn(batch_size, 512, 16, 8)
skip_connections = [
    torch.randn(batch_size, 64, 257, 128),   # From encoder level 1
    torch.randn(batch_size, 128, 128, 64),   # From encoder level 2
    torch.randn(batch_size, 256, 64, 32),    # From encoder level 3
    torch.randn(batch_size, 512, 32, 16),    # From encoder level 4
]

print("="*70)

print(type(skip_connections))
print(type(skip_connections[0]))
#print(skip_connections)

print("="*70)


y_decoder = decoder(x_bottleneck, skip_connections)

print(f"\nDecoder output: {y_decoder.shape}")
print(f"✓ Successfully upsampled back to original resolution!")



class UNet(nn.Module):
    """
        Complete U-Net for magnitude-only speech enhancement.

        Correct channel progression:
        - Input: 1 channel
        - Encoder levels: [64, 128, 256, 512]
        - Bottleneck: 512 channels
        - Decoder levels: [512, 256, 128, 64]
        - Output: 1 channel
    """

    def __init__(self, in_channels:int, out_channels:int, encoder_channels:List[int] = None):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]


        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder_channels = encoder_channels

        # ============ ENCODER ============
        self.encoder_blocks = nn.ModuleList()

        channels = [in_channels] + encoder_channels

        for i in range(len(encoder_channels)):
            self.encoder_blocks.append(EncoderBlock(channels[i], channels[i+1]))

        # ============ BOTTLENECK ============
        # Input: encoder_channels[-1] (512)
        # Output: encoder_channels[-1] (512) - same size, just refined features
        self.bottleneck = DoubleConvBlock(
            encoder_channels[-1],
            encoder_channels[-1]
        )

        # ============ DECODER ============
        # Decoder channels go from highest to lowest
        # [512, 256, 128, 64]
        # This creates 3 decoder blocks: 512→256, 256→128, 128→64
        # But we need 4! So we need [512, 512, 256, 128, 64]

        decoder_channels = [encoder_channels[-1]] + encoder_channels[::-1]
        # Result: [512, 512, 256, 128, 64] ✓ 5 elements = 4 decoder blocks

        skip_channels_reversed = encoder_channels[::-1]  # [512, 256, 128, 64]
        # skip_channels_reversed = [512, 256, 128, 64]

        self.decoder_blocks = nn.ModuleList()

        for i in range(len(decoder_channels)-1):
            self.decoder_blocks.append(DecoderBlock(in_channels=decoder_channels[i],skip_channels=skip_channels_reversed[i], out_channels=decoder_channels[i+1]))

        # ============ FINAL OUTPUT ============
        self.final_conv = nn.Conv2d(
            decoder_channels[-1],
            out_channels,
            kernel_size=1
        )

        # Initialize weights
        self._initialize_weights()
        self._print_summary(decoder_channels, skip_channels_reversed)


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        input_shape = x.shape
        skip_connections = []

        print("\n" + "=" * 70)
        print("ENCODER")
        print("=" * 70)
        print(f"Input: {x.shape}")

        for i, encoder_block in enumerate(self.encoder_blocks):
            x,skip = encoder_block(x)
            skip_connections.append(skip)

            print(f"Level {i + 1}: pooled {x.shape}, skip {skip.shape}")

        bottleneck_input = x

        # ============ BOTTLENECK ============
        print("\n" + "=" * 70)
        print("BOTTLENECK")
        print("=" * 70)
        print(f"Input: {bottleneck_input.shape}")

        x = self.bottleneck(bottleneck_input)

        print(f"Output: {x.shape}")

        # ============ DECODER ============
        print("\n" + "=" * 70)
        print("DECODER")
        print("=" * 70)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            print(f"Level {i + 1}: input {x.shape}, skip {skip.shape}", end=" → ")
            x = decoder_block(x, skip)
            print(f"output {x.shape}")

        # ============ FINAL OUTPUT ============
        print("\n" + "=" * 70)
        print("FINAL OUTPUT")
        print("=" * 70)
        print(f"  Before final conv: {x.shape}")

        x = self.final_conv(x)
        print(f"  After final conv: {x.shape}")
        print(f"  Expected: {input_shape}")

        if x.shape != input_shape:
            print(f"  ⚠️  WARNING: Output shape doesn't match input!")
            print(f"     Input:  {input_shape}")
            print(f"     Output: {x.shape}")
        else:
            print(f"  ✓ Output shape matches input!")

        return x

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _print_summary(self, decoder_channels, skip_channels_reversed):
        """Print architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("U-NET ARCHITECTURE SUMMARY")
        print("=" * 70)
        print(f"Input/Output: 1 channel (magnitude spectrogram)")
        print(f"Encoder channels: {self.encoder_channels}")
        print(f"Decoder channels: {decoder_channels}")
        print(f"Skip channels (encoder reversed): {skip_channels_reversed}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 70)


# ============================================================================
# TESTING
# ============================================================================

print("\n" + "#" * 70)
print("# COMPLETE MAGNITUDE-ONLY U-NET FOR SPEECH ENHANCEMENT")
print("#" * 70)

# Create model
model = UNet(
    in_channels=1,
    out_channels=1,
    encoder_channels=[64, 128, 256, 512]
)

# Create input
batch_size = 2
freq_bins = 257
time_frames = 128

x = torch.randn(batch_size, 1, freq_bins, time_frames)

# Forward pass
try:
    y = model(x)

    print("\n" + "=" * 70)
    print("✓ SUCCESS! U-NET WORKING PERFECTLY!")
    print("=" * 70)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Spatial dimensions preserved: {x.shape[2:]} → {y.shape[2:]}")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback

    traceback.print_exc()



