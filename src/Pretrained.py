#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import keras_tuner
from preprocess import MultiSpectral, SimplifiedSTFT

from Logger import TensorBoard as ConfusionMatrixCallback


image_models = {
    "MobileNet": keras.applications.MobileNet,
    "ResNet50": keras.applications.ResNet50,
    "ResNet50V2": keras.applications.ResNet50V2,
    "ResNet101": keras.applications.ResNet101,
    "ResNet101V2": keras.applications.ResNet101V2
}

def PretrainedModel(hp: keras_tuner.HyperParameters, SAMPLE_RATE: int, input_shape: tuple = (None,)):
    """
    A custom Keras model for audio classification.
    """
    image_model = hp.Choice(
        "image_model", list(image_models.keys()))
    
    image_size = (224, 224)

    if image_model in list(image_models.keys()):
        image_model = image_models[image_model](
            include_top=False,
            input_shape=image_size + (3, ),
            pooling="max",
        )
    else:
        raise ValueError(f"Unknown image model: {image_model}")

    input = keras.layers.Input(input_shape, name="input")
    
    spectrograms = {}
    for c in ["R", "G", "B"]:
        type = hp.Choice(f"Spectrogram_{c}", ["Mel", "STFT"])
        if type == "Mel":
            with hp.conditional_scope(f"Spectrogram_{c}", "Mel"):
                n_mel_bins = hp.Int(f"{c}_mel_bins", 96, 512, sampling="log", default=256)
                spectrograms[c] = keras.layers.MelSpectrogram(sampling_rate=SAMPLE_RATE, num_mel_bins=n_mel_bins)
        elif type == "STFT":
            with hp.conditional_scope(f"Spectrogram_{c}", "STFT"):
                frame_length = hp.Int(f"{c}_frame_length", 10, 250, 10, default=50)
                frame_step = hp.Int(f"{c}_frame_step", 10, 100, step=5, default=20)
                if frame_step > frame_length:
                    frame_step = frame_length
                spectrograms[c] = SimplifiedSTFT(
                    mode="log",
                    frame_length=frame_length * SAMPLE_RATE // 1000,
                    frame_step=frame_step * SAMPLE_RATE // 1000
                )
    
    preprocessing_layer = MultiSpectral(spectrograms.values(), image_size, name="Preprocessing_Layer")
    

    with keras.RematScope(
        # This is a custom scope to allow for the use of Keras' `Remat` feature
        # which allows for the reuse of intermediate tensors to save memory.

        mode="larger_than",
        output_size_threshold=1024 ** 4,
    ):
        x = preprocessing_layer(input)
    # x = keras.layers.Reshape((None, None, 3))(x)
    x = image_model(x)

    hidden_layer_count = hp.Int("hidden_layers", 1, 3, step=1, default=1)
    hidden_size = hp.Int("hidden_size", 128, 512, step=64)
    pipeline = keras.layers.Pipeline([
        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.5),

        *[keras.layers.Dense(units=hidden_size, activation="relu") for _ in range(hidden_layer_count)],

        keras.layers.Dense(units=10, activation="softmax"),
    ], name="Pretrained_Model_Pipeline")(x)

    model = keras.models.Model(inputs=input, outputs=pipeline)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float(
            "learning_rate", 1e-5, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.preprocessing_layer = preprocessing_layer

    return model


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    import pandas as pd
    import argparse
    import sklearn.model_selection

    parser = argparse.ArgumentParser(description="Audio Data Explorer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=Path.cwd() / "output",
        help="Directory containing the audio data.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="The sample rate of the processed audio files.",
        required=True
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=Path.cwd() / "logs",
        help="Directory to save the logs.",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parsed_args = parser.parse_args()

    dataset = pd.read_feather(Path(parsed_args.data_dir) / "audio.feather")
    tuner = keras_tuner.Hyperband(
        lambda x: PretrainedModel(
            x, parsed_args.sample_rate, dataset["audio"].iloc[0].shape),
        objective='val_loss',
        max_epochs=50,
        factor=3,
        hyperband_iterations=parsed_args.iterations,
        directory=Path.cwd() / "hyper_parameter_tuning",
        project_name="Pretrained",
        seed=42,
    )

    dataset["genre_id"] = dataset["genre"].astype("category").cat.codes

    print(dataset["genre_id"])

    x_train, x_test, y_train, y_test, labels_train, labels_test = \
        sklearn.model_selection.train_test_split(
            np.stack(dataset["audio"]),
            dataset["genre_id"].to_numpy(),
            dataset["genre"].to_numpy(),

            test_size=0.2,
            random_state=42,
            stratify=dataset["genre_id"].to_numpy()
        )

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=parsed_args.batch_size,
        callbacks=[
            # keras.callbacks.TensorBoard(parsed_args.log_dir),
            ConfusionMatrixCallback(
                validation_data=(x_test, y_test),
                log_dir=parsed_args.log_dir,
                batch_size=parsed_args.batch_size,
                categories=dataset["genre"].dtype,
                sample_rate=parsed_args.sample_rate,
            ),
            keras.callbacks.EarlyStopping(  # Stop `return 1` classifiers earlier
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(  # stop spiraling around optimal point
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ],
    )
