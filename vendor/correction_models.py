
from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, ReLU,
    Conv3D, Conv3DTranspose, MaxPool3D, Subtract,
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Layer,
    MultiHeadAttention, Lambda, Multiply, Add,
)
from tensorflow.keras.models import Model


# ═══════════════════════════════════════════════════════════════════════════
#  ORIGINAL MODELS — preserved unchanged for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

def unet_3d(input_shape, output_channel, kernel_size=3, filters_root=32,
            conv_times=3, up_down_times=4, if_relu=False, if_residule=False):
    def conv3d_relu(input_, filters_, kernel_size_, name):
        output_ = Conv3D(filters=filters_, kernel_size=kernel_size_,
                         padding="same", name=name + "/Conv3D")(input_)
        return ReLU(name=name + "/Activation")(output_)

    def conv3d_transpose_relu(input_, filters_, kernel_size_, name):
        output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_,
                                  padding="same", strides=(2, 2, 1),
                                  name=name + "/Conv3DTranspose")(input_)
        return ReLU(name=name + "/Activation")(output_)

    skip_connection = []
    ipt = Input(input_shape, name="UNet3D/Keras_Input")
    net = conv3d_relu(ipt, filters_root, kernel_size, name="UNet3D/InputConv")

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(conv_times):
            net = conv3d_relu(net, filters, kernel_size,
                              name=f"UNet3D/Down_{layer}/ConvLayer_{i}")
        skip_connection.append(net)
        net = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1),
                        name=f"UNet3D/Down_{layer}/MaxPool3D")(net)

    filters = 2 ** up_down_times * filters_root
    for i in range(conv_times):
        net = conv3d_relu(net, filters, kernel_size,
                          name=f"UNet3D/Bottom/ConvLayer_{i}")

    for layer in range(up_down_times - 1, -1, -1):
        filters = 2 ** layer * filters_root
        net = conv3d_transpose_relu(net, filters, kernel_size,
                                    name=f"UNet3D/Up_{layer}/UpSample")
        net = Concatenate(axis=-1, name=f"UNet3D/Up_{layer}/SkipConnection")(
            [net, skip_connection[layer]])
        for i in range(conv_times):
            net = conv3d_relu(net, filters, kernel_size,
                              name=f"UNet3D/Up_{layer}/ConvLayer_{i}")

    net = Conv3D(filters=output_channel, kernel_size=1, padding="same",
                 name="UNet3D/OutputConv")(net)
    if if_relu:
        net = ReLU(name="UNet3D/extra_Activation")(net)
    if if_residule:
        net = Subtract()([ipt, net])
    return Model(inputs=ipt, outputs=net)


def unet_2d(input_shape, output_channel, kernel_size=3, filters_root=32,
            conv_times=3, up_down_times=4, if_relu=False, if_residule=False):
    def conv2d_relu(input_, filters_, kernel_size_, name):
        output_ = Conv2D(filters=filters_, kernel_size=kernel_size_,
                         padding="same", name=name + "/Conv2D")(input_)
        return ReLU(name=name + "/Activation")(output_)

    def conv2d_transpose_relu(input_, filters_, kernel_size_, name):
        output_ = Conv2DTranspose(filters=filters_, kernel_size=kernel_size_,
                                  padding="same", strides=(2, 2),
                                  name=name + "/Conv2Transpose")(input_)
        return ReLU(name=name + "/Activation")(output_)

    skip_connection = []
    ipt = Input(input_shape, name="UNet2D/Keras_Input")
    net = conv2d_relu(ipt, filters_root, kernel_size, name="UNet2D/InputConv")

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(conv_times):
            net = conv2d_relu(net, filters, kernel_size,
                              name=f"UNet2D/Down_{layer}/ConvLayer_{i}")
        skip_connection.append(net)
        net = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                        name=f"UNet2D/Down_{layer}/MaxPool2D")(net)

    filters = 2 ** up_down_times * filters_root
    for i in range(conv_times):
        net = conv2d_relu(net, filters, kernel_size,
                          name=f"UNet2D/Bottom/ConvLayer_{i}")

    for layer in range(up_down_times - 1, -1, -1):
        filters = 2 ** layer * filters_root
        net = conv2d_transpose_relu(net, filters, kernel_size,
                                    name=f"UNet2D/Up_{layer}/UpSample")
        net = Concatenate(axis=-1, name=f"UNet2D/Up_{layer}/SkipConnection")(
            [net, skip_connection[layer]])
        for i in range(conv_times):
            net = conv2d_relu(net, filters, kernel_size,
                              name=f"UNet2D/Up_{layer}/ConvLayer_{i}")

    net = Conv2D(filters=output_channel, kernel_size=1, padding="same",
                 name="UNet2D/OutputConv")(net)
    if if_relu:
        net = ReLU(name="UNet2D/extra_Activation")(net)
    if if_residule:
        net = Subtract()([ipt, net])
    return Model(inputs=ipt, outputs=net)


def classifier_2d(input_shape, filters_root=16, conv_blocks=4, dropout=0.2,
                  dense_units=64, use_batch_norm=True):
    ipt = Input(input_shape, name="Cls2D/Keras_Input")
    net = ipt
    for layer in range(conv_blocks):
        filters = (2 ** layer) * filters_root
        net = Conv2D(filters, 3, padding="same",
                     name=f"Cls2D/Block_{layer}/Conv2D_0")(net)
        if use_batch_norm:
            net = BatchNormalization(name=f"Cls2D/Block_{layer}/BN_0")(net)
        net = ReLU(name=f"Cls2D/Block_{layer}/Act_0")(net)
        net = Conv2D(filters, 3, padding="same",
                     name=f"Cls2D/Block_{layer}/Conv2D_1")(net)
        if use_batch_norm:
            net = BatchNormalization(name=f"Cls2D/Block_{layer}/BN_1")(net)
        net = ReLU(name=f"Cls2D/Block_{layer}/Act_1")(net)
        net = MaxPool2D((2, 2), strides=(2, 2),
                        name=f"Cls2D/Block_{layer}/MaxPool2D")(net)
    net = GlobalAveragePooling2D(name="Cls2D/Head/GAP")(net)
    net = Dense(int(dense_units), activation="relu", name="Cls2D/Head/Dense")(net)
    if float(dropout) > 0.0:
        net = Dropout(float(dropout), name="Cls2D/Head/Dropout")(net)
    out = Dense(1, activation="sigmoid", name="Cls2D/Head/Output")(net)
    return Model(inputs=ipt, outputs=out)


# ═══════════════════════════════════════════════════════════════════════════
#  INNOVATION 1+2 — Complex-Valued (2+1)D Convolution
# ═══════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="IMG")
class ComplexConv3D(Layer):
    """Complex-valued 3D convolution.

    Data convention (throughout this file):
      Feature tensor shape: (B, H, W, E, 2·F)
        — first F channels  = real parts  of F complex feature maps
        — last  F channels  = imaginary parts

    For the raw input from cache (B, H, W, E, 2):
        F_in = 1 → real part = x[..., :1], imag part = x[..., 1:]

    Complex multiplication rule applied per filter:
        out_r = W_r ⊛ x_r  −  W_i ⊛ x_i
        out_i = W_r ⊛ x_i  +  W_i ⊛ x_r

    W_r and W_i are two independent real-valued Conv3D layers.
    Together they parameterise a complex weight tensor W = W_r + i·W_i.
    """

    def __init__(self, filters: int, kernel_size, padding: str = "same",
                 use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = use_bias
        self.conv_r = Conv3D(filters, kernel_size, padding=padding,
                             use_bias=use_bias, name="Wr")
        self.conv_i = Conv3D(filters, kernel_size, padding=padding,
                             use_bias=use_bias, name="Wi")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return tf.concat([out_r, out_i], axis=-1)  # (B,H,W,E, 2·filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel_size": self.kernel_size,
                    "padding": self.padding, "use_bias": self.use_bias})
        return cfg


@tf.keras.utils.register_keras_serializable(package="IMG")
class ComplexConv3DTranspose(Layer):
    """Complex-valued transposed 3D convolution (upsampling).

    Mirrors ComplexConv3D exactly: two real Conv3DTranspose layers (Wr, Wi)
    implement the complex conjugate transpose rule:
        out_r = Wr^T ⊛ x_r  −  Wi^T ⊛ x_i
        out_i = Wr^T ⊛ x_i  +  Wi^T ⊛ x_r

    Without this, using two independent real transposed convolutions for Re and Im
    breaks the complex algebraic structure in the decoder path — the kernels are
    completely unrelated and the phase relationship is not preserved on upsampling.
    """

    def __init__(self, filters: int, kernel_size, padding: str = "same",
                 strides=(2, 2, 1), use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.padding     = padding
        self.strides     = strides
        self.use_bias    = use_bias
        self.conv_r = Conv3DTranspose(filters, kernel_size, padding=padding,
                                      strides=strides, use_bias=use_bias, name="Wr")
        self.conv_i = Conv3DTranspose(filters, kernel_size, padding=padding,
                                      strides=strides, use_bias=use_bias, name="Wi")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return tf.concat([out_r, out_i], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel_size": self.kernel_size,
                    "padding": self.padding, "strides": self.strides,
                    "use_bias": self.use_bias})
        return cfg


@tf.keras.utils.register_keras_serializable(package="IMG")
class ComplexInstanceNorm(Layer):
    """Instance normalisation for complex feature maps (B, H, W, E, 2·F).

    Normalises real and imaginary parts jointly using their combined statistics,
    which preserves the relative magnitude ratio between Re and Im (i.e. phase).

    Separate learnable scale γ and shift β per complex channel (per F).
    """

    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        C = int(input_shape[-1])
        if C % 2 != 0:
            raise ValueError(
                f"ComplexInstanceNorm expects even last dim (2*F), got {C}."
            )
        F = C // 2    # number of complex channels
        # gamma operates on the complex channel dimension (size F, not 2F)
        self.gamma = self.add_weight(name="gamma", shape=(F,), initializer="ones")
        # Retain beta for backward-compatible checkpoint loading, but keep it
        # non-trainable and unused in the forward pass. The previous real-only
        # bias introduced an avoidable asymmetry between real and imaginary
        # branches, and leaving it trainable would trigger no-gradient warnings.
        self.beta  = self.add_weight(
            name="beta",
            shape=(F,),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        # Compute magnitude per complex channel for normalisation reference
        mag = tf.sqrt(tf.square(x_r) + tf.square(x_i) + self.epsilon)
        # Normalise over spatial+echo axes (1,2,3), per sample, per channel
        axes = list(range(1, len(x.shape) - 1))
        mean_mag = tf.reduce_mean(mag, axis=axes, keepdims=True)
        scale = mean_mag + self.epsilon
        gamma = tf.cast(self.gamma, x.dtype)
        # Normalise both parts by the same scale and gain.
        x_r_n = (x_r / scale) * gamma
        x_i_n = (x_i / scale) * gamma
        return tf.concat([x_r_n, x_i_n], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["epsilon"] = self.epsilon
        return cfg


@tf.keras.utils.register_keras_serializable(package="IMG")
class ComplexReLU(Layer):
    """CReLU activation: apply ReLU independently to Re and Im parts.

    Simple, differentiable, widely used in complex-valued DL.
    Alternative to modReLU (|z|·ReLU(|z|+b)/|z|·exp(iθ)) which is harder
    to train and provides no consistent advantage for MRI correction tasks.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.concat([tf.nn.relu(x_r), tf.nn.relu(x_i)], axis=-1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="IMG")
class EchoRealOnlyProjection(Layer):
    """Force one echo to be real-valued while preserving [..., E, 2] shape."""

    def __init__(self, echo_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        if int(echo_index) < 0:
            raise ValueError("echo_index must be non-negative")
        self.echo_index = int(echo_index)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        echo_count = x.shape[3]
        if echo_count is not None and self.echo_index >= int(echo_count):
            raise ValueError(
                f"echo_index={self.echo_index} is outside the model echo dimension {echo_count}"
            )
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        before = x_i[:, :, :, : self.echo_index, :]
        echo = x_i[:, :, :, self.echo_index : self.echo_index + 1, :]
        after = x_i[:, :, :, self.echo_index + 1 :, :]
        x_i = tf.concat([before, tf.zeros_like(echo), after], axis=3)
        return tf.concat([x_r, x_i], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["echo_index"] = self.echo_index
        return cfg


def img_output_magnitude(t: tf.Tensor) -> tf.Tensor:
    """Magnitude of a complex image-domain output with channel layout [..., Re, Im]."""
    real, imag = tf.split(t, num_or_size_splits=2, axis=-1)
    return tf.squeeze(tf.sqrt(tf.square(real) + tf.square(imag) + 1e-8), axis=-1)


def img_phase_cosine_error(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    mask_threshold: float = 0.03,
    expect_provided_mask: bool = False,
) -> tf.Tensor:
    """Foreground-masked phase cosine error on complex outputs."""
    return img_phase_cosine_error_for_echoes(
        y_true,
        y_pred,
        start_echo=0,
        end_echo=None,
        mask_threshold=mask_threshold,
        expect_provided_mask=expect_provided_mask,
    )


def img_phase_cosine_error_for_echoes(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    start_echo: int = 0,
    end_echo: Optional[int] = None,
    mask_threshold: float = 0.03,
    expect_provided_mask: bool = False,
) -> tf.Tensor:
    """Foreground-masked phase cosine error for a selected echo range."""
    if expect_provided_mask:
        y_true = y_true[..., :2]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tr, ti = tf.split(y_true, num_or_size_splits=2, axis=-1)
    pr, pi = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    tmag = tf.sqrt(tf.square(tr) + tf.square(ti) + 1e-8)
    pmag = tf.sqrt(tf.square(pr) + tf.square(pi) + 1e-8)
    cos = (tr * pr + ti * pi) / (tmag * pmag + 1e-8)
    echo1 = tmag[..., 0:1, :]
    scale = tf.reduce_max(echo1, axis=(1, 2), keepdims=True)
    base_mask = tf.cast(echo1 >= tf.maximum(scale * float(mask_threshold), 1e-8), cos.dtype)
    cos_sel = cos[:, :, :, int(start_echo) : end_echo, :]
    mask = tf.repeat(base_mask, repeats=tf.shape(cos_sel)[3], axis=3)
    return tf.reduce_sum((1.0 - cos_sel) * mask) / (tf.reduce_sum(mask) + 1e-8)


def _complex_factorized_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """One (2+1)D complex factorized block:
        ComplexConv3D(3×3×1) → ComplexInstanceNorm → CReLU
        ComplexConv3D(1×1×3) → ComplexInstanceNorm → CReLU
    """
    # Spatial: 3×3×1 — spatial context within each echo, preserving complex structure
    x = ComplexConv3D(filters, kernel_size=(3, 3, 1), name=f"{name}/SpatialConv")(x)
    x = ComplexInstanceNorm(name=f"{name}/SpatialNorm")(x)
    x = ComplexReLU(name=f"{name}/SpatialAct")(x)
    # Echo: 1×1×3 — mix adjacent echo times at each spatial position
    x = ComplexConv3D(filters, kernel_size=(1, 1, 3), name=f"{name}/EchoConv")(x)
    x = ComplexInstanceNorm(name=f"{name}/EchoNorm")(x)
    x = ComplexReLU(name=f"{name}/EchoAct")(x)
    return x


# ═══════════════════════════════════════════════════════════════════════════
#  INNOVATION 3 — Echo-Axis Multi-Head Self-Attention
# ═══════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="IMG")
class EchoAxisAttention(Layer):
    """Multi-head self-attention across the echo-time dimension.

    At each spatial position (h,w), the E echo feature vectors are treated as
    a sequence of length E and standard scaled dot-product attention is applied.

    This is strictly more expressive than SE:
      SE:   maps (B,E) → (B,E) scalar per echo, independent of position
      This: maps (B,H,W,E,C) → (B,H,W,E,C) with positional echo context

    Memory strategy: apply only at bottleneck and optionally lowest encoder level
    where H and W are small (12×12 for a 192×192 input with 4 down-sampling levels).
    At these levels spatial flattening is tractable.

    Parameters
    ----------
    num_echoes : int
        E — length of the echo sequence.
    num_heads  : int
        Number of attention heads. Default 2 (E=10, small sequence).
    key_dim    : int
        Dimension of Q/K projections per head.
    """

    def __init__(self, num_echoes: int, num_heads: int = 2,
                 key_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_echoes = num_echoes
        self.num_heads  = num_heads
        self.key_dim    = key_dim
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            attention_axes=None, name="mha"
        )
        self.epsilon = 1e-5

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.ln_gamma = self.add_weight(name="ln_gamma", shape=(c,), initializer="ones")
        self.ln_beta = self.add_weight(name="ln_beta", shape=(c,), initializer="zeros")
        super().build(input_shape)

    def _manual_layer_norm(self, x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x_hat = (x - mean) * tf.math.rsqrt(var + tf.cast(self.epsilon, x.dtype))
        return x_hat * tf.cast(self.ln_gamma, x.dtype) + tf.cast(self.ln_beta, x.dtype)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: (B, H, W, E, C)
        # B/H/W/E are dynamic (vary per batch); C must be static so that
        # MultiHeadAttention can build its Q/K/V projection weights at graph
        # construction time.  Using tf.shape(x)[4] returns a Tensor, which
        # leaves the last dim unknown to MHA and causes mis-built projections
        # or graph errors.  x.shape[-1] is always statically known here
        # because filters_root is a Python int fixed at model-build time.
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        E = tf.shape(x)[3]
        C = x.shape[-1]   # STATIC — required for MHA weight initialisation
        # Reshape: collapse batch and spatial dims → (B·H·W, E, C)
        x_flat = tf.reshape(x, [B * H * W, E, C])
        # Pre-norm → attention → residual
        x_norm    = self._manual_layer_norm(x_flat)
        attended  = self.mha(x_norm, x_norm, training=training)
        x_out     = x_flat + attended
        return tf.reshape(x_out, [B, H, W, E, C])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_echoes": self.num_echoes, "num_heads": self.num_heads,
                    "key_dim": self.key_dim})
        return cfg


# ═══════════════════════════════════════════════════════════════════════════
#  INNOVATION 4 — Combined Physics-Consistent + Magnitude-Aligned Loss
# ═══════════════════════════════════════════════════════════════════════════

def make_img_loss(
    lambda_mag:  float = 0.3,
    lambda_phys: float = 0.1,
    loss_type: str = "l2",
    te_ms: Optional[List[float]] = None,
    use_te_weighting: bool = True,
    mask_mode: str = "echo1_true",
    mask_threshold: float = 0.03,
    expect_provided_mask: bool = False,
    echo1_real_only: bool = False,
    echo1_index: int = 0,
    lambda_e1_real: float = 0.0,
    lambda_e1_mag: float = 0.0,
    lambda_e1_imag_zero: float = 0.0,
) -> Callable:
    """Combined loss for complex multi-echo MRI correction.

    Three terms:
      L_complex  — MSE on raw (Re,Im) channels  (existing baseline)
      L_mag      — MSE on magnitudes |z|         (aligns loss with evaluation)
      L_phys     — T2* mono-exponential consistency on predicted output

    L = (1 - λ_m - λ_p) · L_complex  +  λ_m · L_mag  +  λ_p · L_phys

    T2* consistency (L_phys):
      log|S(TE)| should be linear in TE at each (h,w) position.
      Deviation from linearity = second finite difference Δ²log|S| ≠ 0.
      L_phys = mean( (Δ²log|S_pred|)² )

    Parameters
    ----------
    lambda_mag  : weight for the magnitude reconstruction term
    lambda_phys : weight for T2* consistency term
    loss_type   : reconstruction penalty, either ``l1`` or ``l2``
    te_ms       : echo times in ms. Must be explicitly provided when
                  lambda_phys > 0 and use_te_weighting=True.

    Usage (in train_keras.py, mri_3decho branch of _compile_model):
        model.compile(optimizer=optimizer,
                      loss=make_img_loss(lambda_mag=0.3, lambda_phys=0.1),
                      metrics=metrics)
    """
    lam_m = float(lambda_mag)
    lam_p = float(lambda_phys)
    lam_c = 1.0 - lam_m - lam_p
    recon_loss_type = str(loss_type).strip().lower()
    if recon_loss_type not in ("l1", "l2"):
        raise ValueError(
            f"Unsupported loss_type={loss_type!r}. Expected 'l1' or 'l2'."
        )
    e1_enabled = bool(echo1_real_only)
    e1_idx = int(echo1_index)
    lam_e1_r = float(lambda_e1_real)
    lam_e1_m = float(lambda_e1_mag)
    lam_e1_i = float(lambda_e1_imag_zero)
    te = None

    if te_ms is not None:
        try:
            te_values = [float(t) for t in te_ms]
        except (TypeError, ValueError) as exc:
            raise ValueError("make_img_loss received a non-numeric train.physics.te_ms value.") from exc
        if len(te_values) < 3:
            raise ValueError("make_img_loss requires at least 3 echo times when train.physics.te_ms is provided.")
        if any(curr <= prev for prev, curr in zip(te_values[:-1], te_values[1:])):
            raise ValueError("make_img_loss requires train.physics.te_ms to be strictly increasing.")
        te = tf.constant([t / 1e3 for t in te_values], dtype=tf.float32)
    elif lam_p > 0.0 and use_te_weighting:
        raise ValueError(
            "make_img_loss requires explicit train.physics.te_ms when lambda_phys > 0 "
            "and use_te_weighting=True. Provide the true echo times, or set "
            "lambda_phys=0, or disable use_te_weighting."
        )

    if lam_c < 0.0:
        import warnings
        warnings.warn(
            f"make_img_loss: lambda_mag={lam_m} + lambda_phys={lam_p} = {lam_m+lam_p:.2f} > 1.0. "
            "The complex-domain MSE term (L_complex) weight is clamped to 0. "
            "Consider reducing lambda_mag or lambda_phys so they sum to <= 1.",
            stacklevel=2,
        )
        lam_c = 0.0
    mode = str(mask_mode).lower()
    if mode not in ("none", "echo1_true", "echo1_pred", "provided"):
        raise ValueError(f"Unsupported mask_mode={mask_mode}. Expected none|echo1_true|echo1_pred|provided.")
    if e1_idx < 0:
        raise ValueError("echo1_index must be non-negative.")

    def _mag(t: tf.Tensor) -> tf.Tensor:
        """Compute magnitude from (…, 2·F) complex feature convention.
        For F=1 (output layer): last dim = [Re, Im]."""
        t = tf.cast(t, tf.float32)
        F = t.shape[-1] // 2
        return tf.sqrt(tf.square(t[..., :F]) + tf.square(t[..., F:]) + 1e-8)

    def _reconstruction_penalty(error: tf.Tensor) -> tf.Tensor:
        if recon_loss_type == "l1":
            return tf.reduce_mean(tf.abs(error))
        return tf.reduce_mean(tf.square(error))

    def _split_true_and_mask(y_true: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        if expect_provided_mask:
            y_true_complex = y_true[..., :2]
            provided_mask = y_true[..., 2:3]
            return y_true_complex, provided_mask
        return y_true, None

    def _build_physics_mask(
        y_true_complex: tf.Tensor,
        y_pred: tf.Tensor,
        provided_mask: Optional[tf.Tensor],
    ) -> Optional[tf.Tensor]:
        if mode == "none":
            return None
        if mode == "provided":
            if provided_mask is None:
                return None
            return tf.clip_by_value(provided_mask, 0.0, 1.0)

        if mode == "echo1_true":
            ref = tf.squeeze(_mag(y_true_complex), axis=-1)
        else:
            ref = tf.squeeze(_mag(y_pred), axis=-1)

        e1 = ref[..., 0:1]
        scale = tf.reduce_max(e1, axis=(1, 2), keepdims=True)
        thresh = tf.maximum(scale * float(mask_threshold), 1e-8)
        mask2d = tf.cast(e1 >= thresh, tf.float32)
        num_echo = tf.shape(ref)[-1]
        return tf.repeat(mask2d, repeats=num_echo, axis=-1)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_complex, provided_mask = _split_true_and_mask(y_true)
        y_true_complex = tf.cast(y_true_complex, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if provided_mask is not None:
            provided_mask = tf.cast(provided_mask, tf.float32)

        # ── Complex MSE ──────────────────────────────────────────────────
        l_c = _reconstruction_penalty(y_pred - y_true_complex)

        # ── Magnitude MSE ─────────────────────────────────────────────────
        l_m = _reconstruction_penalty(_mag(y_pred) - _mag(y_true_complex))

        # ── T2* consistency on predicted output ───────────────────────────
        if lam_p > 0.0:
            # y_pred: (B, H, W, E, 2)  where E = number of echoes
            # magnitude per echo: (B, H, W, E)
            mag_pred = tf.squeeze(_mag(y_pred), axis=-1)   # (B, H, W, E)

            # Log-magnitude: log|S(TE)| should be linear in TE
            log_mag = tf.math.log(mag_pred + 1e-8)         # (B, H, W, E)

            diff1 = log_mag[..., 1:] - log_mag[..., :-1]

            if use_te_weighting:
                te32 = tf.cast(te, log_mag.dtype)
                dte = te32[1:] - te32[:-1]
                dte = tf.maximum(dte, tf.cast(1e-8, log_mag.dtype))
                slope = diff1 / dte[None, None, None, :]
                dt_mid = 0.5 * (te32[2:] - te32[:-2])
                dt_mid = tf.maximum(dt_mid, tf.cast(1e-8, log_mag.dtype))
                phys_term = (slope[..., 1:] - slope[..., :-1]) / dt_mid[None, None, None, :]
            else:
                phys_term = diff1[..., 1:] - diff1[..., :-1]

            mask = _build_physics_mask(y_true_complex, y_pred, provided_mask)
            if mask is None:
                l_p = tf.reduce_mean(tf.square(phys_term))
            else:
                mask_core = tf.cast(mask[..., 1:-1], phys_term.dtype)
                weighted = tf.square(phys_term) * mask_core
                l_p = tf.reduce_sum(weighted) / (tf.reduce_sum(mask_core) + tf.cast(1e-8, phys_term.dtype))
        else:
            l_p = tf.cast(0.0, y_pred.dtype)

        total = lam_c * l_c + lam_m * l_m + lam_p * l_p

        if e1_enabled:
            yt_e1 = y_true_complex[:, :, :, e1_idx : e1_idx + 1, :]
            yp_e1 = y_pred[:, :, :, e1_idx : e1_idx + 1, :]
            yt_r, yt_i = tf.split(yt_e1, num_or_size_splits=2, axis=-1)
            yp_r, yp_i = tf.split(yp_e1, num_or_size_splits=2, axis=-1)
            yt_mag = tf.sqrt(tf.square(yt_r) + tf.square(yt_i) + 1e-8)
            yp_mag = tf.sqrt(tf.square(yp_r) + tf.square(yp_i) + 1e-8)
            l_e1_r = _reconstruction_penalty(yp_r - yt_r)
            l_e1_m = _reconstruction_penalty(yp_mag - yt_mag)
            l_e1_i = _reconstruction_penalty(yp_i)
            total = total + lam_e1_r * l_e1_r + lam_e1_m * l_e1_m + lam_e1_i * l_e1_i

        return total

    if e1_enabled:
        loss.__name__ = (
            f"img_{recon_loss_type}_lc{lam_c:.2f}_lm{lam_m:.2f}_lp{lam_p:.2f}"
            f"_e1r{lam_e1_r:.2f}_e1m{lam_e1_m:.2f}_e1i{lam_e1_i:.2f}"
        )
    else:
        loss.__name__ = (
            f"img_{recon_loss_type}_lc{lam_c:.2f}_lm{lam_m:.2f}_lp{lam_p:.2f}"
        )
    return loss


DISENTANGLE_VARIANTS = ("disentangle", "disentangle_phys")


def _complex_slice_echo(x: tf.Tensor, echo_index: int, name: str) -> tf.Tensor:
    return Lambda(lambda t: t[:, :, :, echo_index : echo_index + 1, :], name=name)(x)


def _complex_mag_feature(x: tf.Tensor, keepdims: bool = True, name: Optional[str] = None) -> tf.Tensor:
    def _fn(t: tf.Tensor) -> tf.Tensor:
        xr, xi = tf.split(t, num_or_size_splits=2, axis=-1)
        mag = tf.sqrt(tf.square(xr) + tf.square(xi) + 1e-8)
        return tf.reduce_mean(mag, axis=-1, keepdims=keepdims)
    return Lambda(_fn, name=name)(x)


def _fft_magnitude_volume(x: tf.Tensor, name: str) -> tf.Tensor:
    def _fn(t: tf.Tensor) -> tf.Tensor:
        xr, xi = tf.split(t, num_or_size_splits=2, axis=-1)
        z = tf.complex(tf.squeeze(xr, axis=-1), tf.squeeze(xi, axis=-1))
        z = tf.transpose(z, perm=[0, 3, 1, 2])
        fft = tf.signal.fft2d(z)
        mag = tf.math.log1p(tf.abs(fft))
        mag = tf.transpose(mag, perm=[0, 2, 3, 1])
        return mag[..., None]
    return Lambda(_fn, name=name)(x)


def _repeat_along_echo(x: tf.Tensor, repeats: int, name: str) -> tf.Tensor:
    return Lambda(lambda t: tf.repeat(t, repeats=repeats, axis=3), name=name)(x)


def _repeat_gate_hw(x: tf.Tensor, scale: int, name: str) -> tf.Tensor:
    def _fn(t: tf.Tensor) -> tf.Tensor:
        t = tf.repeat(t, repeats=scale, axis=1)
        t = tf.repeat(t, repeats=scale, axis=2)
        return t
    return Lambda(_fn, name=name)(x)


@tf.keras.utils.register_keras_serializable(package="IMG")
class DisentangleRegularizer(Layer):
    """Branch-separation penalty for anatomy/echo/artifact latents."""

    def __init__(self, lambda_ortho: float = 0.02, lambda_invariance: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ortho = float(lambda_ortho)
        self.lambda_invariance = float(lambda_invariance)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "lambda_ortho": self.lambda_ortho,
            "lambda_invariance": self.lambda_invariance,
        })
        return cfg

    @staticmethod
    def _summary_map(x: tf.Tensor) -> tf.Tensor:
        xr, xi = tf.split(x, num_or_size_splits=2, axis=-1)
        mag = tf.sqrt(tf.square(xr) + tf.square(xi) + 1e-8)
        return tf.reduce_mean(mag, axis=-1)

    @staticmethod
    def _pairwise_cosine(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        a_flat = tf.reshape(a, [tf.shape(a)[0], -1])
        b_flat = tf.reshape(b, [tf.shape(b)[0], -1])
        a_flat = tf.math.l2_normalize(a_flat, axis=-1)
        b_flat = tf.math.l2_normalize(b_flat, axis=-1)
        return tf.reduce_mean(tf.reduce_sum(a_flat * b_flat, axis=-1))

    def call(self, inputs, **kwargs):
        z_a, z_e, z_m = inputs
        a_map = self._summary_map(z_a)
        e_map = self._summary_map(z_e)
        m_map = self._summary_map(z_m)

        ortho = (
            tf.square(self._pairwise_cosine(a_map, e_map)) +
            tf.square(self._pairwise_cosine(a_map, m_map)) +
            tf.square(self._pairwise_cosine(e_map, m_map))
        ) / 3.0

        a_echo_var = tf.reduce_mean(tf.math.reduce_variance(a_map, axis=3))
        m_echo_mean = tf.reduce_mean(m_map, axis=3, keepdims=True)
        m_echo_var = tf.reduce_mean(tf.math.reduce_variance(m_map, axis=3))
        invariance = a_echo_var + 0.5 * tf.square(
            self._pairwise_cosine(a_map, tf.repeat(m_echo_mean, repeats=tf.shape(a_map)[3], axis=3))
        )

        self.add_loss(self.lambda_ortho * ortho + self.lambda_invariance * invariance)
        self.add_metric(ortho, name=f"{self.name}_ortho", aggregation="mean")
        self.add_metric(a_echo_var, name=f"{self.name}_a_echo_var", aggregation="mean")
        self.add_metric(m_echo_var, name=f"{self.name}_m_echo_var", aggregation="mean")
        return inputs


def _resolve_latent_splits(total_filters: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    total = max(3, int(total_filters))
    ratios = [max(0.0, float(r)) for r in ratios]
    denom = sum(ratios)
    if denom <= 0.0:
        ratios = [0.5, 0.3, 0.2]
        denom = 1.0
    raw = [max(1, int(round(total * (r / denom)))) for r in ratios]
    diff = total - sum(raw)
    raw[0] += diff
    while raw[0] < 1:
        for idx in (1, 2):
            if raw[idx] > 1:
                raw[idx] -= 1
                raw[0] += 1
                if raw[0] >= 1:
                    break
    return int(raw[0]), int(raw[1]), int(raw[2])

# ═══════════════════════════════════════════════════════════════════════════
#  AUGMENTED IMG MODEL — unet_3d_img
# ═══════════════════════════════════════════════════════════════════════════

def unet_3d_img(
    input_shape:     Tuple[int, ...],
    output_channel:  int   = 2,
    filters_root:    int   = 32,
    conv_times:      int   = 3,
    up_down_times:   int   = 4,
    num_echoes:      int   = 10,
    attn_heads:      int   = 2,
    attn_key_dim:    int   = 16,
    use_factorized:  bool  = True,
    use_attention:   bool  = True,
    echo1_real_head: bool  = False,
    echo1_index:     int   = 0,
    echo1_head_filters: Optional[int] = None,
    echo1_head_depth:   int = 2,
    echo1_head_residual: bool = True,
    if_relu:         bool  = False,
    if_residule:     bool  = False,
) -> Model:
    """IMG UNet-3D — complex-valued, (2+1)D factorized, with echo attention.

    Parameters
    ----------
    input_shape   : (H, W, E, 2) — height, width, echoes, [Re,Im]
    output_channel: must be 2 for complex output (Re, Im)
    filters_root  : base complex filter count F. Each level uses 2·F real channels.
    conv_times    : factorized blocks per UNet level
    up_down_times : encoder/decoder depth
    num_echoes    : E (must match input_shape[2])
    attn_heads    : attention heads in EchoAxisAttention
    attn_key_dim  : key dimension per head
    if_relu       : apply CReLU to final output
    if_residule   : global residual (subtract model output from input)

    Config JSON changes needed (filters_root can be reduced vs original since
    complex convolutions have 2x the representational capacity per filter):
        "unet3d": { "model": "unet3d_img",  // use new model
                    "filters_root": 32,      // was 64; complex=equivalent capacity
                    ... }
    """
    assert output_channel == 2, "unet_3d_img requires output_channel=2 (complex Re,Im)"

    def _slice_echo(x: tf.Tensor, idx: int, name: str) -> tf.Tensor:
        return Lambda(lambda t: t[:, :, :, idx : idx + 1, :], name=name)(x)

    def _conv_block(x: tf.Tensor, filters: int, base: str) -> tf.Tensor:
        for i in range(conv_times):
            if use_factorized:
                x = _complex_factorized_block(x, filters, name=f"{base}/FBlock_{i}")
            else:
                x = ComplexConv3D(filters, kernel_size=(3, 3, 3), name=f"{base}/CBlock_{i}/Conv")(x)
                x = ComplexInstanceNorm(name=f"{base}/CBlock_{i}/Norm")(x)
                x = ComplexReLU(name=f"{base}/CBlock_{i}/Act")(x)
        return x

    def _upconv(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        # ComplexConv3DTranspose preserves the complex algebraic structure:
        #   out_r = Wr^T * x_r - Wi^T * x_i
        #   out_i = Wr^T * x_i + Wi^T * x_r
        # The previous implementation (two independent real Conv3DTranspose layers)
        # broke this by using completely unrelated kernels for Re and Im.
        x = ComplexConv3DTranspose(
            filters, kernel_size=3, padding="same", strides=(2, 2, 1),
            name=f"{name}/CTranspose"
        )(x)
        x = ComplexInstanceNorm(name=f"{name}/UpNorm")(x)
        x = ComplexReLU(name=f"{name}/UpAct")(x)
        return x

    # ── Input stem ────────────────────────────────────────────────────────
    ipt = Input(input_shape, name="IMGUNet/Input")
    # Project (B,H,W,E,2) → (B,H,W,E,2·filters_root) with complex conv
    net = ComplexConv3D(filters_root, kernel_size=(3, 3, 1),
                        name="IMGUNet/Stem/SpatialConv")(ipt)
    net = ComplexInstanceNorm(name="IMGUNet/Stem/Norm")(net)
    net = ComplexReLU(name="IMGUNet/Stem/Act")(net)

    # ── Encoder ───────────────────────────────────────────────────────────
    skips = []
    for lvl in range(up_down_times):
        filters = (2 ** lvl) * filters_root
        net = _conv_block(net, filters, base=f"IMGUNet/Down_{lvl}")
        skips.append(net)
        net = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1),
                        name=f"IMGUNet/Down_{lvl}/Pool")(net)

    # ── Bottleneck + Echo-Axis Attention ──────────────────────────────────
    filters = (2 ** up_down_times) * filters_root
    net = _conv_block(net, filters, base="IMGUNet/Bottom")
    # Echo attention applied at bottleneck only (H,W are 12×12 here for 192 input)
    # Operate on real and imaginary channels independently with same attention
    if use_attention:
        net_r, net_i = tf.split(net, num_or_size_splits=2, axis=-1)
        net_r = EchoAxisAttention(
            num_echoes=num_echoes,
            num_heads=attn_heads,
            key_dim=attn_key_dim,
            name="IMGUNet/Bottom/EchoAttn_Re",
        )(net_r)
        net_i = EchoAxisAttention(
            num_echoes=num_echoes,
            num_heads=attn_heads,
            key_dim=attn_key_dim,
            name="IMGUNet/Bottom/EchoAttn_Im",
        )(net_i)
        net = tf.concat([net_r, net_i], axis=-1)

    # ── Decoder ───────────────────────────────────────────────────────────
    for lvl in range(up_down_times - 1, -1, -1):
        filters = (2 ** lvl) * filters_root
        net = _upconv(net, filters, name=f"IMGUNet/Up_{lvl}/UpSample")
        net = Concatenate(axis=-1, name=f"IMGUNet/Up_{lvl}/Skip")(
            [net, skips[lvl]])
        net = _conv_block(net, filters, base=f"IMGUNet/Up_{lvl}")

    # ── Output head ───────────────────────────────────────────────────────
    # Project 2·filters_root complex channels → 1 complex channel (Re,Im)
    # Use ComplexConv3D(1, kernel_size=1) → output shape (B,H,W,E,2)
    decoder_features = net
    net = ComplexConv3D(1, kernel_size=1, name="IMGUNet/OutConv")(net)

    if if_relu:
        net = ComplexReLU(name="IMGUNet/OutAct")(net)
    if if_residule:
        net = Subtract(name="IMGUNet/Residual")([ipt, net])

    if echo1_real_head:
        e1_idx = int(echo1_index)
        if e1_idx < 0 or e1_idx >= int(num_echoes):
            raise ValueError(f"echo1_index={e1_idx} is outside num_echoes={num_echoes}")
        head_filters = int(echo1_head_filters or filters_root)
        head_depth = max(1, int(echo1_head_depth))

        e1_decoder = _slice_echo(decoder_features, e1_idx, name="IMGUNet/Echo1Head/DecoderEcho")
        echo_context = Lambda(
            lambda t: tf.reduce_mean(t, axis=3, keepdims=True),
            name="IMGUNet/Echo1Head/AllEchoContext",
        )(decoder_features)
        e1_input = _slice_echo(ipt, e1_idx, name="IMGUNet/Echo1Head/InputEcho")
        e1_shared = _slice_echo(net, e1_idx, name="IMGUNet/Echo1Head/SharedEcho")
        e1_feat = Concatenate(axis=-1, name="IMGUNet/Echo1Head/FeatureConcat")(
            [e1_decoder, echo_context, e1_input, e1_shared]
        )
        for head_idx in range(head_depth):
            e1_feat = Conv3D(
                head_filters,
                kernel_size=(3, 3, 1),
                padding="same",
                activation="relu",
                name=f"IMGUNet/Echo1Head/Refine_{head_idx}",
            )(e1_feat)
        e1_delta = Conv3D(
            1,
            kernel_size=1,
            padding="same",
            activation=None,
            name="IMGUNet/Echo1Head/RealDelta",
        )(e1_feat)
        e1_shared_real = Lambda(lambda t: t[..., 0:1], name="IMGUNet/Echo1Head/SharedReal")(e1_shared)
        if bool(echo1_head_residual):
            e1_real = Add(name="IMGUNet/Echo1Head/RealResidual")([e1_shared_real, e1_delta])
        else:
            e1_real = e1_delta
        e1_imag = Lambda(lambda t: tf.zeros_like(t), name="IMGUNet/Echo1Head/ZeroImag")(e1_real)
        e1_complex = Concatenate(axis=-1, name="IMGUNet/Echo1Head/ComplexEcho")([e1_real, e1_imag])

        echo_pieces = []
        if e1_idx > 0:
            echo_pieces.append(Lambda(lambda t: t[:, :, :, :e1_idx, :], name="IMGUNet/Echo1Head/Before")(net))
        echo_pieces.append(e1_complex)
        if e1_idx + 1 < int(num_echoes):
            echo_pieces.append(Lambda(lambda t: t[:, :, :, e1_idx + 1 :, :], name="IMGUNet/Echo1Head/After")(net))
        net = Concatenate(axis=3, name="IMGUNet/Echo1Head/MergedOutput")(echo_pieces)

    return Model(inputs=ipt, outputs=net, name="unet_3d_img")


def build_unet3d_disentangle_variant(
    input_shape: Tuple[int, ...],
    variant: str,
    output_channel: int = 2,
    filters_root: int = 32,
    conv_times: int = 3,
    up_down_times: int = 4,
    num_echoes: Optional[int] = None,
    attn_heads: int = 2,
    attn_key_dim: int = 16,
    anchor_echo_index: int = 0,
    latent_ratios: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    use_fft_artifact_branch: bool = True,
    lambda_ortho: float = 0.02,
    lambda_invariance: float = 0.02,
    if_relu: bool = False,
    if_residule: bool = False,
) -> Model:
    """Direct clean-image corrector with disentangled anatomy, echo, and artifact latents."""
    del if_residule
    v = str(variant).strip().lower()
    if v not in DISENTANGLE_VARIANTS:
        raise ValueError(
            f"Unsupported disentangle variant: {variant}. "
            f"Expected one of {'|'.join(DISENTANGLE_VARIANTS)}."
        )
    if output_channel != 2:
        raise ValueError("build_unet3d_disentangle_variant requires output_channel=2.")
    if num_echoes is None:
        if len(input_shape) < 3:
            raise ValueError(f"Expected input_shape=(H,W,E,2), got {input_shape}")
        num_echoes = int(input_shape[2])
    if not 0 <= int(anchor_echo_index) < int(num_echoes):
        raise ValueError(f"anchor_echo_index={anchor_echo_index} must be in [0, {int(num_echoes)-1}]")

    def _conv_block(x: tf.Tensor, filters: int, base: str) -> tf.Tensor:
        for i in range(conv_times):
            x = _complex_factorized_block(x, filters, name=f"{base}/FBlock_{i}")
        return x

    def _slice_echo(x: tf.Tensor, idx: int, name: str) -> tf.Tensor:
        return Lambda(lambda t: t[:, :, :, idx : idx + 1, :], name=name)(x)

    def _upconv(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        x = ComplexConv3DTranspose(
            filters, kernel_size=3, padding="same", strides=(2, 2, 1),
            name=f"{name}/CTranspose",
        )(x)
        x = ComplexInstanceNorm(name=f"{name}/UpNorm")(x)
        x = ComplexReLU(name=f"{name}/UpAct")(x)
        return x

    ipt = Input(input_shape, name="Disentangle/Input")
    net = ComplexConv3D(filters_root, kernel_size=(3, 3, 1), name="Disentangle/Stem/SpatialConv")(ipt)
    net = ComplexInstanceNorm(name="Disentangle/Stem/Norm")(net)
    net = ComplexReLU(name="Disentangle/Stem/Act")(net)

    skips = []
    for lvl in range(up_down_times):
        filters = (2 ** lvl) * filters_root
        net = _conv_block(net, filters, base=f"Disentangle/Down_{lvl}")
        skips.append(net)
        net = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), name=f"Disentangle/Down_{lvl}/Pool")(net)

    bottom_filters = (2 ** up_down_times) * filters_root
    net = _conv_block(net, bottom_filters, base="Disentangle/Bottom")
    net_r, net_i = tf.split(net, num_or_size_splits=2, axis=-1)
    net_r = EchoAxisAttention(
        num_echoes=num_echoes,
        num_heads=attn_heads,
        key_dim=attn_key_dim,
        name="Disentangle/Bottom/EchoAttn_Re",
    )(net_r)
    net_i = EchoAxisAttention(
        num_echoes=num_echoes,
        num_heads=attn_heads,
        key_dim=attn_key_dim,
        name="Disentangle/Bottom/EchoAttn_Im",
    )(net_i)
    net = tf.concat([net_r, net_i], axis=-1)

    latent_total = max(filters_root * 4, bottom_filters // 2)
    anatomy_filters, echo_filters, artifact_filters = _resolve_latent_splits(
        latent_total,
        ratios=latent_ratios,
    )

    anchor = _complex_slice_echo(ipt, int(anchor_echo_index), name="Disentangle/Anchor/EchoSlice")
    anchor = ComplexConv3D(anatomy_filters, kernel_size=(3, 3, 1), name="Disentangle/Anchor/Stem")(anchor)
    anchor = ComplexInstanceNorm(name="Disentangle/Anchor/StemNorm")(anchor)
    anchor = ComplexReLU(name="Disentangle/Anchor/StemAct")(anchor)
    for lvl in range(up_down_times):
        anchor = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), name=f"Disentangle/Anchor/Pool_{lvl}")(anchor)
    anchor = ComplexConv3D(anatomy_filters, kernel_size=1, name="Disentangle/Anchor/Project")(anchor)
    anchor = ComplexInstanceNorm(name="Disentangle/Anchor/ProjectNorm")(anchor)
    anchor = ComplexReLU(name="Disentangle/Anchor/ProjectAct")(anchor)

    z_a = ComplexConv3D(anatomy_filters, kernel_size=(3, 3, 1), name="Disentangle/LatentA/Spatial")(net)
    z_a = ComplexInstanceNorm(name="Disentangle/LatentA/SpatialNorm")(z_a)
    z_a = ComplexReLU(name="Disentangle/LatentA/SpatialAct")(z_a)
    z_a = Lambda(lambda t: tf.reduce_mean(t, axis=3, keepdims=True), name="Disentangle/LatentA/EchoPool")(z_a)
    z_a = Concatenate(axis=-1, name="Disentangle/LatentA/AnchorFuse")([z_a, anchor])
    z_a = ComplexConv3D(anatomy_filters, kernel_size=1, name="Disentangle/LatentA/FuseProject")(z_a)
    z_a = ComplexInstanceNorm(name="Disentangle/LatentA/FuseNorm")(z_a)
    z_a = ComplexReLU(name="Disentangle/LatentA/FuseAct")(z_a)
    z_a_rep = _repeat_along_echo(z_a, repeats=int(num_echoes), name="Disentangle/LatentA/Broadcast")

    z_e = _complex_factorized_block(net, echo_filters, name="Disentangle/LatentE/Factorized")
    z_e_r, z_e_i = tf.split(z_e, num_or_size_splits=2, axis=-1)
    z_e_r = EchoAxisAttention(
        num_echoes=num_echoes,
        num_heads=attn_heads,
        key_dim=attn_key_dim,
        name="Disentangle/LatentE/EchoAttn_Re",
    )(z_e_r)
    z_e_i = EchoAxisAttention(
        num_echoes=num_echoes,
        num_heads=attn_heads,
        key_dim=attn_key_dim,
        name="Disentangle/LatentE/EchoAttn_Im",
    )(z_e_i)
    z_e = tf.concat([z_e_r, z_e_i], axis=-1)

    z_m = ComplexConv3D(artifact_filters, kernel_size=(3, 3, 1), name="Disentangle/LatentM/Spatial")(net)
    z_m = ComplexInstanceNorm(name="Disentangle/LatentM/SpatialNorm")(z_m)
    z_m = ComplexReLU(name="Disentangle/LatentM/SpatialAct")(z_m)
    if use_fft_artifact_branch:
        fft_mag = _fft_magnitude_volume(ipt, name="Disentangle/FFT/Magnitude")
        for lvl in range(up_down_times):
            fft_mag = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), name=f"Disentangle/FFT/Pool_{lvl}")(fft_mag)
        fft_feat = Conv3D(artifact_filters, kernel_size=1, padding="same", activation="relu", name="Disentangle/FFT/Project")(fft_mag)
        z_m_mag = _complex_mag_feature(z_m, keepdims=True, name="Disentangle/LatentM/MagFeat")
        z_m_mag = Concatenate(axis=-1, name="Disentangle/LatentM/FFTConcat")([z_m_mag, fft_feat])
        z_m_gate = Conv3D(artifact_filters, kernel_size=1, padding="same", activation="sigmoid", name="Disentangle/LatentM/FFTFuse")(z_m_mag)
        z_m_gate = Lambda(lambda t: tf.repeat(t, repeats=2, axis=-1), name="Disentangle/LatentM/FFTGateExpand")(z_m_gate)
        z_m = Multiply(name="Disentangle/LatentM/FFTScale")([z_m, z_m_gate])

    z_a_rep, z_e, z_m = DisentangleRegularizer(
        lambda_ortho=lambda_ortho,
        lambda_invariance=lambda_invariance,
        name="Disentangle/Reg",
    )([z_a_rep, z_e, z_m])

    artifact_map = _complex_mag_feature(z_m, keepdims=True, name="Disentangle/Artifact/Mag")
    artifact_map = Conv3D(1, kernel_size=1, padding="same", activation="sigmoid", name="Disentangle/ArtifactMap")(artifact_map)
    gate_inv = Lambda(lambda t: 1.0 - tf.clip_by_value(t, 0.0, 1.0), name="Disentangle/Artifact/InverseGate")(artifact_map)
    z_e = Multiply(name="Disentangle/LatentE/Gated")([z_e, gate_inv])
    decoder_seed = Concatenate(axis=-1, name="Disentangle/Decoder/SeedConcat")([z_a_rep, z_e])
    net = ComplexConv3D(bottom_filters, kernel_size=1, name="Disentangle/Decoder/SeedProject")(decoder_seed)
    net = ComplexInstanceNorm(name="Disentangle/Decoder/SeedNorm")(net)
    net = ComplexReLU(name="Disentangle/Decoder/SeedAct")(net)

    anchor_pred = ComplexConv3D(1, kernel_size=1, name="Disentangle/AnchorPred")(z_a)
    echo_curve_pred = ComplexConv3D(1, kernel_size=1, name="Disentangle/EchoCurve/Project")(z_e)
    echo_curve_pred = _complex_mag_feature(echo_curve_pred, keepdims=True, name="Disentangle/EchoCurvePred")

    for lvl in range(up_down_times - 1, -1, -1):
        filters = (2 ** lvl) * filters_root
        scale = 2 ** (up_down_times - lvl)
        gate_lvl = _repeat_gate_hw(artifact_map, scale=scale, name=f"Disentangle/Artifact/UpsampleGate_{lvl}")
        gate_inv_lvl = Lambda(lambda t: 1.0 - tf.clip_by_value(t, 0.0, 1.0), name=f"Disentangle/Up_{lvl}/SkipInv")(gate_lvl)
        skip_gated = Multiply(name=f"Disentangle/Up_{lvl}/SkipGate")([skips[lvl], gate_inv_lvl])
        net = _upconv(net, filters, name=f"Disentangle/Up_{lvl}/UpSample")
        net = Concatenate(axis=-1, name=f"Disentangle/Up_{lvl}/Skip")([net, skip_gated])
        net = _conv_block(net, filters, base=f"Disentangle/Up_{lvl}")

    corrected = ComplexConv3D(1, kernel_size=1, name="Disentangle/OutConv")(net)
    if if_relu:
        corrected = ComplexReLU(name="Disentangle/OutAct")(corrected)
    corrected = Lambda(
        lambda xs: xs[0] + tf.zeros_like(xs[0]) * (
            tf.reduce_sum(xs[1]) + tf.reduce_sum(xs[2]) + tf.reduce_sum(xs[3])
        ),
        name="corrected",
    )([corrected, anchor_pred, echo_curve_pred, artifact_map])
    return Model(inputs=ipt, outputs=corrected, name="unet_3d_disentangle")


def build_unet3d_img_variant(
    input_shape: Tuple[int, ...],
    variant: str,
    output_channel: int = 2,
    filters_root: int = 32,
    conv_times: int = 3,
    up_down_times: int = 4,
    num_echoes: Optional[int] = None,
    attn_heads: int = 2,
    attn_key_dim: int = 16,
    anchor_echo_index: int = 0,
    latent_ratios: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    use_fft_artifact_branch: bool = True,
    lambda_ortho: float = 0.02,
    lambda_invariance: float = 0.02,
    echo1_real_head: bool = False,
    echo1_head_filters: Optional[int] = None,
    echo1_head_depth: int = 2,
    echo1_head_residual: bool = True,
    if_relu: bool = False,
    if_residule: bool = False,
) -> Model:
    """Build staged IMG variants for robust rollout.

    Supported variants:
      - baseline: legacy unet_3d
      - complex: complex convs, isotropic kernels, no attention
      - factorized: complex + (2+1)D factorized, no attention
      - attention: complex + factorized + echo attention
      - attention_phys: same architecture as attention (use with physics loss)
      - attention_phys_e1: attention_phys with echo-1 real-only decoder head/projection
      - disentangle: direct clean-image corrector with anatomy/echo/artifact branches
      - disentangle_phys: same architecture with ordinal-echo smoothness in corrected loss
    """
    v = str(variant).strip().lower()
    if v == "baseline":
        return unet_3d(
            input_shape=input_shape,
            output_channel=output_channel,
            kernel_size=3,
            filters_root=filters_root,
            conv_times=conv_times,
            up_down_times=up_down_times,
            if_relu=if_relu,
            if_residule=if_residule,
        )

    if num_echoes is None:
        if len(input_shape) < 3:
            raise ValueError(f"Expected input_shape=(H,W,E,2), got {input_shape}")
        num_echoes = int(input_shape[2])

    if v == "complex":
        return unet_3d_img(
            input_shape=input_shape,
            output_channel=output_channel,
            filters_root=filters_root,
            conv_times=conv_times,
            up_down_times=up_down_times,
            num_echoes=num_echoes,
            attn_heads=attn_heads,
            attn_key_dim=attn_key_dim,
            use_factorized=False,
            use_attention=False,
            if_relu=if_relu,
            if_residule=if_residule,
        )

    if v == "factorized":
        return unet_3d_img(
            input_shape=input_shape,
            output_channel=output_channel,
            filters_root=filters_root,
            conv_times=conv_times,
            up_down_times=up_down_times,
            num_echoes=num_echoes,
            attn_heads=attn_heads,
            attn_key_dim=attn_key_dim,
            use_factorized=True,
            use_attention=False,
            if_relu=if_relu,
            if_residule=if_residule,
        )

    if v in ("attention", "attention_phys", "attention_phys_e1"):
        model = unet_3d_img(
            input_shape=input_shape,
            output_channel=output_channel,
            filters_root=filters_root,
            conv_times=conv_times,
            up_down_times=up_down_times,
            num_echoes=num_echoes,
            attn_heads=attn_heads,
            attn_key_dim=attn_key_dim,
            use_factorized=True,
            use_attention=True,
            echo1_real_head=bool(echo1_real_head and v == "attention_phys_e1"),
            echo1_index=int(anchor_echo_index),
            echo1_head_filters=echo1_head_filters,
            echo1_head_depth=int(echo1_head_depth),
            echo1_head_residual=bool(echo1_head_residual),
            if_relu=if_relu,
            if_residule=if_residule,
        )
        if v == "attention_phys_e1":
            out = EchoRealOnlyProjection(
                echo_index=int(anchor_echo_index),
                name="IMGUNet/Echo1RealOnly",
            )(model.output)
            model = Model(inputs=model.input, outputs=out, name="unet_3d_img_attention_phys_e1")
        return model

    if v in DISENTANGLE_VARIANTS:
        return build_unet3d_disentangle_variant(
            input_shape=input_shape,
            variant=v,
            output_channel=output_channel,
            filters_root=filters_root,
            conv_times=conv_times,
            up_down_times=up_down_times,
            num_echoes=num_echoes,
            attn_heads=attn_heads,
            attn_key_dim=attn_key_dim,
            anchor_echo_index=anchor_echo_index,
            latent_ratios=latent_ratios,
            use_fft_artifact_branch=use_fft_artifact_branch,
            lambda_ortho=lambda_ortho,
            lambda_invariance=lambda_invariance,
            if_relu=if_relu,
            if_residule=if_residule,
        )

    raise ValueError(
        f"Unsupported IMG variant: {variant}. "
        "Expected one of baseline|complex|factorized|attention|attention_phys|attention_phys_e1|disentangle|disentangle_phys."
    )


def compile_img_model_with_variant(
    model: tf.keras.Model,
    learning_rate: float,
    loss_type: str,
    variant: str,
    lambda_mag: float = 0.0,
    lambda_phys: float = 0.0,
    te_ms: Optional[List[float]] = None,
    use_te_weighting: bool = True,
    physics_mask_mode: str = "echo1_true",
    physics_mask_threshold: float = 0.03,
    physics_expect_provided_mask: bool = False,
    psnr_max_val: float = 1.0,
    echo1_real_only_enabled: bool = False,
    echo1_index: int = 0,
    lambda_e1_real: float = 0.0,
    lambda_e1_mag: float = 0.0,
    lambda_e1_imag_zero: float = 0.0,
) -> None:
    """Compile IMG model with stage-appropriate loss and shared metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=1.0)
    loss_type = str(loss_type).lower()
    if loss_type not in ("l1", "l2"):
        raise ValueError(f"Unsupported lossType: {loss_type}. Expected l1 or l2.")
    psnr_max_val = float(psnr_max_val)
    v = str(variant).lower()
    e1_idx = int(echo1_index)

    def _split_true(y_true: tf.Tensor) -> tf.Tensor:
        if physics_expect_provided_mask:
            y_true = y_true[..., :2]
        return tf.cast(y_true, tf.float32)

    def _mag(t: tf.Tensor) -> tf.Tensor:
        return img_output_magnitude(tf.cast(t, tf.float32))

    def _psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))
        y_pred_mag = _mag(y_pred)
        if y_true_mag.shape.rank == 3:
            y_true_mag = y_true_mag[..., None]
            y_pred_mag = y_pred_mag[..., None]
        return tf.reduce_mean(tf.image.psnr(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))
        y_pred_mag = _mag(y_pred)
        if y_true_mag.shape.rank == 3:
            y_true_mag = y_true_mag[..., None]
            y_pred_mag = y_pred_mag[..., None]
        return tf.reduce_mean(tf.image.ssim(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _snr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))
        y_pred_mag = _mag(y_pred)
        y_true_flat = tf.reshape(y_true_mag, [-1])
        y_pred_flat = tf.reshape(y_pred_mag, [-1])
        num = tf.norm(y_true_flat)
        den = tf.norm(y_true_flat - y_pred_flat)
        return 20.0 * (tf.math.log(num / (den + 1e-12)) / tf.math.log(tf.constant(10.0, dtype=y_true_mag.dtype)))

    def _phase(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return img_phase_cosine_error(
            y_true,
            y_pred,
            mask_threshold=float(physics_mask_threshold),
            expect_provided_mask=bool(physics_expect_provided_mask),
        )

    def _echo1_psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))[:, :, :, e1_idx : e1_idx + 1]
        y_pred_mag = _mag(y_pred)[:, :, :, e1_idx : e1_idx + 1]
        return tf.reduce_mean(tf.image.psnr(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _echo1_ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))[:, :, :, e1_idx : e1_idx + 1]
        y_pred_mag = _mag(y_pred)[:, :, :, e1_idx : e1_idx + 1]
        return tf.reduce_mean(tf.image.ssim(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _echo1_snr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))[:, :, :, e1_idx : e1_idx + 1]
        y_pred_mag = _mag(y_pred)[:, :, :, e1_idx : e1_idx + 1]
        y_true_flat = tf.reshape(y_true_mag, [-1])
        y_pred_flat = tf.reshape(y_pred_mag, [-1])
        num = tf.norm(y_true_flat)
        den = tf.norm(y_true_flat - y_pred_flat)
        return 20.0 * (tf.math.log(num / (den + 1e-12)) / tf.math.log(tf.constant(10.0, dtype=y_true_mag.dtype)))

    def _echo2_10_psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))[:, :, :, e1_idx + 1 :]
        y_pred_mag = _mag(y_pred)[:, :, :, e1_idx + 1 :]
        return tf.reduce_mean(tf.image.psnr(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _echo2_10_ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))[:, :, :, e1_idx + 1 :]
        y_pred_mag = _mag(y_pred)[:, :, :, e1_idx + 1 :]
        return tf.reduce_mean(tf.image.ssim(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _echo2_10_snr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(_split_true(y_true))[:, :, :, e1_idx + 1 :]
        y_pred_mag = _mag(y_pred)[:, :, :, e1_idx + 1 :]
        y_true_flat = tf.reshape(y_true_mag, [-1])
        y_pred_flat = tf.reshape(y_pred_mag, [-1])
        num = tf.norm(y_true_flat)
        den = tf.norm(y_true_flat - y_pred_flat)
        return 20.0 * (tf.math.log(num / (den + 1e-12)) / tf.math.log(tf.constant(10.0, dtype=y_true_mag.dtype)))

    def _echo1_imag_energy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        del y_true
        pred = tf.cast(y_pred, tf.float32)
        pr, pi = tf.split(pred, num_or_size_splits=2, axis=-1)
        pr = tf.reshape(pr[:, :, :, e1_idx : e1_idx + 1, :], [-1])
        pi = tf.reshape(pi[:, :, :, e1_idx : e1_idx + 1, :], [-1])
        return tf.norm(pi) / (tf.norm(pr) + 1e-8)

    def _echo2_10_phase(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return img_phase_cosine_error_for_echoes(
            y_true,
            y_pred,
            start_echo=e1_idx + 1,
            end_echo=None,
            mask_threshold=float(physics_mask_threshold),
            expect_provided_mask=bool(physics_expect_provided_mask),
        )

    _psnr.__name__ = "psnr"
    _ssim.__name__ = "ssim"
    _snr.__name__ = "snr"
    _phase.__name__ = "phase_cosine_error"
    _echo1_psnr.__name__ = "echo1_psnr"
    _echo1_ssim.__name__ = "echo1_ssim"
    _echo1_snr.__name__ = "echo1_snr"
    _echo2_10_psnr.__name__ = "echo2_10_psnr"
    _echo2_10_ssim.__name__ = "echo2_10_ssim"
    _echo2_10_snr.__name__ = "echo2_10_snr"
    _echo1_imag_energy.__name__ = "echo1_imag_energy"
    _echo2_10_phase.__name__ = "echo2_10_phase_cosine_error"
    metrics = [_psnr, _ssim, _snr, _phase]
    if v == "attention_phys_e1":
        metrics.extend([
            _echo1_psnr,
            _echo1_ssim,
            _echo1_snr,
            _echo2_10_psnr,
            _echo2_10_ssim,
            _echo2_10_snr,
            _echo1_imag_energy,
            _echo2_10_phase,
        ])

    if v in ("attention_phys", "attention_phys_e1"):
        loss_fn = make_img_loss(
            lambda_mag=float(lambda_mag),
            lambda_phys=float(lambda_phys),
            loss_type=loss_type,
            te_ms=te_ms,
            use_te_weighting=bool(use_te_weighting),
            mask_mode=str(physics_mask_mode),
            mask_threshold=float(physics_mask_threshold),
            expect_provided_mask=bool(physics_expect_provided_mask),
            echo1_real_only=bool(echo1_real_only_enabled),
            echo1_index=e1_idx,
            lambda_e1_real=float(lambda_e1_real),
            lambda_e1_mag=float(lambda_e1_mag),
            lambda_e1_imag_zero=float(lambda_e1_imag_zero),
        )
    else:
        loss_fn = "mae" if loss_type == "l1" else "mse"
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)


def make_disentangle_img_loss(
    loss_type: str = "l2",
    lambda_mag: float = 0.2,
    lambda_phase: float = 0.05,
    lambda_phys: float = 0.0,
    echo_positions: Optional[List[float]] = None,
    mask_threshold: float = 0.03,
) -> Callable:
    """Complex image loss with magnitude, phase, and ordinal-echo smoothness terms."""
    loss_type = str(loss_type).lower()
    if loss_type not in ("l1", "l2"):
        raise ValueError(f"Unsupported loss_type={loss_type}. Expected l1 or l2.")
    complex_fn = tf.abs if loss_type == "l1" else tf.square
    lam_m = max(0.0, float(lambda_mag))
    lam_phase = max(0.0, float(lambda_phase))
    lam_phys = max(0.0, float(lambda_phys))

    positions = None
    if echo_positions is not None:
        try:
            pos = [float(v) for v in echo_positions]
        except (TypeError, ValueError) as exc:
            raise ValueError("train.disentangle.echo_positions contains a non-numeric value.") from exc
        if len(pos) < 3:
            raise ValueError("train.disentangle.echo_positions must contain at least 3 positions.")
        if any(curr <= prev for prev, curr in zip(pos[:-1], pos[1:])):
            raise ValueError("train.disentangle.echo_positions must be strictly increasing.")
        positions = tf.constant(pos, dtype=tf.float32)

    def _mag(t: tf.Tensor) -> tf.Tensor:
        xr, xi = tf.split(t, num_or_size_splits=2, axis=-1)
        return tf.sqrt(tf.square(xr) + tf.square(xi) + 1e-8)

    def _phase_cos_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        tr, ti = tf.split(y_true, num_or_size_splits=2, axis=-1)
        pr, pi = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        tmag = tf.sqrt(tf.square(tr) + tf.square(ti) + 1e-8)
        pmag = tf.sqrt(tf.square(pr) + tf.square(pi) + 1e-8)
        cos = (tr * pr + ti * pi) / (tmag * pmag + 1e-8)
        echo1 = tmag[..., 0:1, :]
        scale = tf.reduce_max(echo1, axis=(1, 2), keepdims=True)
        mask = tf.cast(echo1 >= tf.maximum(scale * float(mask_threshold), 1e-8), cos.dtype)
        mask = tf.repeat(mask, repeats=tf.shape(cos)[3], axis=3)
        return tf.reduce_sum((1.0 - cos) * mask) / (tf.reduce_sum(mask) + 1e-8)

    def _phys_loss(y_pred: tf.Tensor) -> tf.Tensor:
        mag = tf.squeeze(_mag(y_pred), axis=-1)
        log_mag = tf.math.log(mag + 1e-8)
        diff1 = log_mag[..., 1:] - log_mag[..., :-1]
        if positions is not None:
            pos32 = tf.cast(positions, log_mag.dtype)
            dpos = tf.maximum(pos32[1:] - pos32[:-1], tf.cast(1e-8, log_mag.dtype))
            slope = diff1 / dpos[None, None, None, :]
            dmid = tf.maximum(0.5 * (pos32[2:] - pos32[:-2]), tf.cast(1e-8, log_mag.dtype))
            diff2 = (slope[..., 1:] - slope[..., :-1]) / dmid[None, None, None, :]
        else:
            diff2 = diff1[..., 1:] - diff1[..., :-1]
        return tf.reduce_mean(tf.square(diff2))

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        l_complex = tf.reduce_mean(complex_fn(y_pred - y_true))
        l_mag = tf.reduce_mean(complex_fn(_mag(y_pred) - _mag(y_true)))
        l_phase = _phase_cos_loss(y_true, y_pred)
        l_phys = _phys_loss(y_pred) if lam_phys > 0.0 else tf.cast(0.0, y_pred.dtype)
        return l_complex + lam_m * l_mag + lam_phase * l_phase + lam_phys * l_phys

    loss.__name__ = f"disentangle_img_{loss_type}"
    return loss


def compile_disentangle_model(
    model: tf.keras.Model,
    learning_rate: float,
    loss_type: str,
    variant: str,
    lambda_mag: float = 0.2,
    lambda_phase: float = 0.05,
    lambda_phys: float = 0.0,
    echo_positions: Optional[List[float]] = None,
    lambda_anchor: float = 0.1,
    lambda_artifact: float = 0.1,
    lambda_echo_curve: float = 0.1,
    psnr_max_val: float = 1.0,
) -> tf.keras.Model:
    """Compile disentangle train model while keeping the base model deployable."""
    v = str(variant).strip().lower()
    if v not in DISENTANGLE_VARIANTS:
        raise ValueError(
            f"Unsupported disentangle variant: {variant}. "
            f"Expected one of {'|'.join(DISENTANGLE_VARIANTS)}."
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=1.0)
    corrected_loss = make_disentangle_img_loss(
        loss_type=loss_type,
        lambda_mag=float(lambda_mag),
        lambda_phase=float(lambda_phase),
        lambda_phys=float(lambda_phys) if v == "disentangle_phys" else 0.0,
        echo_positions=echo_positions,
    )

    def _mag(t: tf.Tensor) -> tf.Tensor:
        return img_output_magnitude(t)

    def _psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(y_true)
        y_pred_mag = _mag(y_pred)
        if y_true_mag.shape.rank == 3:
            y_true_mag = y_true_mag[..., None]
            y_pred_mag = y_pred_mag[..., None]
        return tf.reduce_mean(tf.image.psnr(y_pred_mag, y_true_mag, max_val=float(psnr_max_val)))

    def _ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(y_true)
        y_pred_mag = _mag(y_pred)
        if y_true_mag.shape.rank == 3:
            y_true_mag = y_true_mag[..., None]
            y_pred_mag = y_pred_mag[..., None]
        return tf.reduce_mean(tf.image.ssim(y_pred_mag, y_true_mag, max_val=float(psnr_max_val)))

    def _snr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(y_true)
        y_pred_mag = _mag(y_pred)
        y_true_flat = tf.reshape(y_true_mag, [-1])
        y_pred_flat = tf.reshape(y_pred_mag, [-1])
        num = tf.norm(y_true_flat)
        den = tf.norm(y_true_flat - y_pred_flat)
        return 20.0 * (tf.math.log(num / (den + 1e-12)) / tf.math.log(tf.constant(10.0, dtype=y_true_mag.dtype)))

    def _phase(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return img_phase_cosine_error(y_true, y_pred)

    _psnr.__name__ = "psnr"
    _ssim.__name__ = "ssim"
    _snr.__name__ = "snr"
    _phase.__name__ = "phase_cosine_error"
    metrics = [_psnr, _ssim, _snr, _phase]

    model.compile(optimizer=optimizer, loss=corrected_loss, metrics=metrics)

    anchor_output = Lambda(lambda t: t, name="anchor_echo")(model.get_layer("Disentangle/AnchorPred").output)
    echo_curve_output = Lambda(lambda t: t, name="echo_curve")(model.get_layer("Disentangle/EchoCurvePred").output)
    artifact_output = Lambda(lambda t: t, name="artifact_map")(model.get_layer("Disentangle/ArtifactMap").output)
    train_model = Model(
        inputs=model.input,
        outputs={
            "corrected": model.output,
            "anchor_echo": anchor_output,
            "echo_curve": echo_curve_output,
            "artifact_map": artifact_output,
        },
        name="unet_3d_disentangle_train",
    )
    train_model.compile(
        optimizer=optimizer,
        loss={
            "corrected": corrected_loss,
            "anchor_echo": "mse",
            "echo_curve": "mse",
            "artifact_map": "mse",
        },
        loss_weights={
            "corrected": 1.0,
            "anchor_echo": float(lambda_anchor),
            "echo_curve": float(lambda_echo_curve),
            "artifact_map": float(lambda_artifact),
        },
        metrics={"corrected": metrics},
    )
    return train_model
