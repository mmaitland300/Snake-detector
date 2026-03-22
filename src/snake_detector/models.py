from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelSpec:
    backbone: str = "inceptionv3"
    image_size: int = 150
    learning_rate: float = 1e-4
    freeze_backbone: bool = True


def _require_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.optimizers import RMSprop
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install with `pip install .[ml]`.") from exc
    return tf, layers, InceptionV3, RMSprop


def build_binary_classifier(spec: ModelSpec):
    if spec.backbone.lower() != "inceptionv3":
        raise ValueError("Only `inceptionv3` backbone is supported in v1.")

    tf, layers, InceptionV3, RMSprop = _require_tf()
    base = InceptionV3(
        input_shape=(spec.image_size, spec.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    for layer in base.layers:
        layer.trainable = not spec.freeze_backbone

    x = layers.Flatten()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(base.input, output)
    model.compile(
        optimizer=RMSprop(learning_rate=spec.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
