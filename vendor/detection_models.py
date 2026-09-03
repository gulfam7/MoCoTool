from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Layer,
    LayerNormalization,
    MaxPool2D,
    Multiply,
    ReLU,
    Reshape,
    TimeDistributed,
)
from tensorflow.keras.models import Model


def _norm(name: str, enabled: bool) -> Layer:
    return LayerNormalization(axis=-1, name=name) if enabled else Activation("linear", name=name)


class EchoChannelGate(Layer):
    def __init__(self, channels: int, hidden: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.fc1 = Dense(self.hidden, activation="relu", name=f"{self.name}_fc1")
        self.fc2 = Dense(self.channels, activation="sigmoid", name=f"{self.name}_fc2")

    def call(self, inputs, training=None):
        summary = tf.reduce_mean(tf.abs(tf.cast(inputs, tf.float32)), axis=[1, 2])
        gates = self.fc2(self.fc1(summary))
        gates_reshaped = tf.reshape(gates, (-1, 1, 1, self.channels))
        return inputs * gates_reshaped

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "hidden": self.hidden})
        return cfg


class PairedComplexEchoGate(Layer):
    def __init__(self, channels: int, hidden: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.eps = float(eps)
        if self.channels % 2 != 0:
            raise ValueError(f"PairedComplexEchoGate requires an even channel count, got {self.channels}")
        self.echoes = self.channels // 2
        self.fc1 = Dense(self.hidden, activation="relu", name=f"{self.name}_fc1")
        self.fc2 = Dense(self.echoes, activation="sigmoid", name=f"{self.name}_fc2")

    def call(self, inputs, training=None):
        x = tf.cast(inputs, tf.float32)
        # Input layout: [Re(e1),...,Re(eN), Im(e1),...,Im(eN)]
        # Split into real and imaginary halves, then pair per echo.
        real = x[..., :self.echoes]
        imag = x[..., self.echoes:]
        mag = tf.sqrt(real * real + imag * imag + self.eps)
        summary = tf.reduce_mean(mag, axis=[1, 2])
        gates = self.fc2(self.fc1(summary))
        gates = tf.reshape(gates, (-1, 1, 1, self.echoes))
        gated_real = real * gates
        gated_imag = imag * gates
        return tf.concat([gated_real, gated_imag], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "hidden": self.hidden, "eps": self.eps})
        return cfg


class MaskedSliceAggregator(Layer):
    def __init__(
        self,
        mode: str = "topk_mean",
        topk: int = 5,
        window_size: int = 3,
        logit_floor: float = 0.0,
        min_positive: int = 3,
        fallback_logit: float = -6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = str(mode).lower()
        self.topk = int(topk)
        self.window_size = int(window_size)
        self.logit_floor = float(logit_floor)
        self.min_positive = int(min_positive)
        self.fallback_logit = float(fallback_logit)
        if self.mode not in ("topk_mean", "mean", "max", "confident_topk", "window_confident_topk"):
            raise ValueError(f"Unsupported aggregation mode: {mode}")

    def call(self, inputs, training=None):
        slice_logits, mask = inputs
        slice_logits = tf.cast(slice_logits, tf.float32)
        mask = tf.cast(mask, tf.float32)
        very_neg = tf.constant(-1e9, dtype=slice_logits.dtype)
        masked_logits = tf.where(mask > 0.5, slice_logits, very_neg)

        if self.mode == "mean":
            denom = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-8
            return tf.reduce_sum(slice_logits * mask, axis=1, keepdims=True) / denom

        if self.mode == "max":
            return tf.reduce_max(masked_logits, axis=1, keepdims=True)

        if self.mode in ("confident_topk", "window_confident_topk"):
            confident_logits = slice_logits
            if self.mode == "window_confident_topk":
                win = max(1, self.window_size)
                logits_exp = tf.expand_dims(slice_logits * mask, axis=-1)
                mask_exp = tf.expand_dims(mask, axis=-1)
                numer = tf.nn.avg_pool1d(logits_exp, ksize=win, strides=1, padding="SAME") * float(win)
                denom = tf.nn.avg_pool1d(mask_exp, ksize=win, strides=1, padding="SAME") * float(win)
                confident_logits = tf.squeeze(numer / tf.maximum(denom, 1e-8), axis=-1)
            confident_mask = tf.logical_and(mask > 0.5, confident_logits > self.logit_floor)
            filtered_logits = tf.where(confident_mask, confident_logits, very_neg)
            k = tf.minimum(tf.shape(slice_logits)[1], tf.constant(max(1, self.topk), dtype=tf.int32))
            values, _ = tf.math.top_k(filtered_logits, k=k, sorted=False)
            valid = tf.cast(values > (very_neg / 2.0), slice_logits.dtype)
            denom = tf.reduce_sum(valid, axis=1, keepdims=True) + 1e-8
            pooled = tf.reduce_sum(values * valid, axis=1, keepdims=True) / denom
            confident_count = tf.reduce_sum(tf.cast(confident_mask, slice_logits.dtype), axis=1, keepdims=True)
            fallback = tf.fill(tf.shape(pooled), tf.cast(self.fallback_logit, slice_logits.dtype))
            return tf.where(confident_count >= float(max(1, self.min_positive)), pooled, fallback)

        k = tf.minimum(tf.shape(slice_logits)[1], tf.constant(max(1, self.topk), dtype=tf.int32))
        values, _ = tf.math.top_k(masked_logits, k=k, sorted=False)
        valid = tf.cast(values > (very_neg / 2.0), slice_logits.dtype)
        denom = tf.reduce_sum(valid, axis=1, keepdims=True) + 1e-8
        return tf.reduce_sum(values * valid, axis=1, keepdims=True) / denom

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "mode": self.mode,
                "topk": self.topk,
                "window_size": self.window_size,
                "logit_floor": self.logit_floor,
                "min_positive": self.min_positive,
                "fallback_logit": self.fallback_logit,
            }
        )
        return cfg


class SqueezeLastDim(Layer):
    def call(self, inputs, training=None):
        return tf.squeeze(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


def _residual_block(x, filters: int, use_layer_norm: bool, name: str):
    shortcut = x
    if int(shortcut.shape[-1]) != int(filters):
        shortcut = Conv2D(filters, 1, padding="same", name=f"{name}_proj")(shortcut)

    net = Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    net = _norm(f"{name}_ln1", use_layer_norm)(net)
    net = ReLU(name=f"{name}_relu1")(net)
    net = Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(net)
    net = _norm(f"{name}_ln2", use_layer_norm)(net)
    net = Add(name=f"{name}_add")([net, shortcut])
    net = ReLU(name=f"{name}_relu2")(net)
    return net


def build_slice_encoder(input_shape: Tuple[int, int, int], model_cfg: Dict, input_representation: str = "magnitude") -> Model:
    gate_hidden = int(model_cfg.get("gate_hidden", 32))
    echo_gate_type = str(model_cfg.get("echo_gate_type", "channel_se")).lower()
    echo_gate_eps = float(model_cfg.get("echo_gate_eps", 1e-6))
    stem_filters = int(model_cfg.get("stem_filters", 16))
    stage_filters = list(model_cfg.get("stage_filters", [16, 32, 64, 128]))
    blocks_per_stage = int(model_cfg.get("blocks_per_stage", 2))
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.2))
    use_layer_norm = bool(model_cfg.get("use_layer_norm", True))

    inp = Input(shape=input_shape, name="SliceInput")
    channels = int(input_shape[-1])
    input_representation = str(input_representation).lower()
    if echo_gate_type == "channel_se":
        gate_layer = EchoChannelGate(channels=channels, hidden=gate_hidden, name="EchoGate")
    elif echo_gate_type == "complex_pair_se":
        if input_representation not in ("complex", "complex20"):
            raise ValueError(
                "echo_gate_type='complex_pair_se' requires complex input_representation "
                f"('complex' or 'complex20'), got {input_representation!r}"
            )
        if channels % 2 != 0:
            raise ValueError(f"echo_gate_type='complex_pair_se' requires an even channel count, got {channels}")
        gate_layer = PairedComplexEchoGate(channels=channels, hidden=gate_hidden, eps=echo_gate_eps, name="EchoGate")
    else:
        raise ValueError(f"Unsupported echo_gate_type: {echo_gate_type}")
    net = gate_layer(inp)
    net = Conv2D(stem_filters, 1, padding="same", name="StemFuse")(net)
    net = _norm("StemLN", use_layer_norm)(net)
    net = ReLU(name="StemReLU")(net)

    for stage_idx, filters in enumerate(stage_filters):
        for block_idx in range(blocks_per_stage):
            net = _residual_block(
                net,
                filters=int(filters),
                use_layer_norm=use_layer_norm,
                name=f"Stage{stage_idx + 1}_Block{block_idx + 1}",
            )
        if stage_idx < len(stage_filters) - 1:
            net = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=f"Stage{stage_idx + 1}_Pool")(net)

    net = GlobalAveragePooling2D(name="SliceGAP")(net)
    net = Dense(embedding_dim, activation="relu", name="SliceEmbeddingDense")(net)
    if dropout > 0.0:
        net = Dropout(dropout, name="SliceEmbeddingDropout")(net)
    return Model(inputs=inp, outputs=net, name="EchoAwareSliceEncoder")


def build_slice_classifier_model(
    slice_shape: Tuple[int, int, int],
    model_cfg: Dict,
    input_representation: str = "magnitude",
) -> Model:
    x_in = Input(shape=tuple(slice_shape), name="SliceX")
    encoder = build_slice_encoder(tuple(slice_shape), model_cfg=model_cfg, input_representation=input_representation)
    emb = encoder(x_in)
    slice_logit = Dense(1, activation=None, name="SliceLogit")(emb)
    p_slice = Activation("sigmoid", name="SliceProbability")(slice_logit)
    return Model(inputs=x_in, outputs=p_slice, name="EchoAwareSliceClassifier")


def build_subject_gate_model(
    slice_shape: Tuple[int, int, int],
    model_cfg: Dict,
    aggregation_cfg: Dict,
    input_representation: str = "magnitude",
) -> Model:
    x_in = Input(shape=(None,) + tuple(slice_shape), name="SubjectBagX")
    mask_in = Input(shape=(None,), dtype=tf.float32, name="SubjectBagMask")

    encoder = build_slice_encoder(
        tuple(slice_shape),
        model_cfg=model_cfg,
        input_representation=input_representation,
    )
    emb = TimeDistributed(encoder, name="TD_SliceEncoder")(x_in)
    slice_logits = TimeDistributed(Dense(1, activation=None), name="TD_SliceLogit")(emb)
    slice_logits = SqueezeLastDim(name="SliceLogitSqueeze")(slice_logits)
    p_slice = Activation("sigmoid", name="SliceProbabilities")(slice_logits)

    aggregator = MaskedSliceAggregator(
        mode=str(aggregation_cfg.get("type", "topk_mean")),
        topk=int(aggregation_cfg.get("topk", 5)),
        window_size=int(aggregation_cfg.get("window_size", 3)),
        logit_floor=float(aggregation_cfg.get("logit_floor", 0.0)),
        min_positive=int(aggregation_cfg.get("min_positive", 3)),
        fallback_logit=float(aggregation_cfg.get("fallback_logit", -6.0)),
        name="SubjectAggregator",
    )
    subject_logit = aggregator([slice_logits, mask_in])
    p_subject = Activation("sigmoid", name="SubjectProbability")(subject_logit)
    return Model(
        inputs=[x_in, mask_in],
        outputs=[p_subject, p_slice, subject_logit, slice_logits],
        name="EchoAwareSubjectGate",
    )
