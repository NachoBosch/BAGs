from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


class Sampling(layers.Layer):
    """Reparameterization trick: z = mu + sigma * epsilon."""

    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def get_config(self):
        return super().get_config()


def _apply_norm(x, idx: int, prefix: str, norm_kind: str):
    """Apply normalization if requested. Accepts 'layer'/'layernorm' and 'batch'/'batchnorm'."""
    nk = (norm_kind or "").lower()
    if nk in ("layer", "layernorm"):
        return layers.LayerNormalization(name=f"{prefix}_ln_{idx}")(x)
    if nk in ("batch", "batchnorm"):
        return layers.BatchNormalization(name=f"{prefix}_bn_{idx}")(x)
    return x


def make_encoder(
    input_dim: int,
    hidden_dims: list[int],
    latent_dim: int,
    l2_reg: float,
    drop_rate: float,
    activation: str,
    norm_kind: str,
) -> keras.Model:
    """Build encoder: x -> (mu, log_var, z)."""
    inp = layers.Input(shape=(input_dim,), name="enc_input")
    x = inp

    for i, units in enumerate(hidden_dims):
        x = layers.Dense(units, kernel_regularizer=l2(l2_reg),
                         name=f"enc_dense_{i+1}")(x)
        x = _apply_norm(x, i + 1, "enc", norm_kind)
        x = layers.Activation(activation, name=f"enc_act_{i+1}")(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, name=f"enc_do_{i+1}")(x)

    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])

    return keras.Model(inp, [mu, log_var, z], name="encoder")


def make_decoder(
    input_dim: int,
    hidden_dims: list[int],
    latent_dim: int,
    l2_reg: float,
    drop_rate: float,
    activation: str,
    norm_kind: str,
) -> keras.Model:
    """Build decoder: z -> x_hat."""
    inp = layers.Input(shape=(latent_dim,), name="dec_input")
    x = inp

    for i, units in enumerate(reversed(hidden_dims)):
        x = layers.Dense(units, kernel_regularizer=l2(l2_reg),
                         name=f"dec_dense_{i+1}")(x)
        x = _apply_norm(x, i + 1, "dec", norm_kind)
        x = layers.Activation(activation, name=f"dec_act_{i+1}")(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, name=f"dec_do_{i+1}")(x)

    out = layers.Dense(input_dim, activation="linear", name="dec_out")(x)
    return keras.Model(inp, out, name="decoder")


class VAE(keras.Model):
    """Beta-VAE with configurable reconstruction loss and KL divergence."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model,
                 beta: float = 1.0, recon: str = "mse", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32)
        self.recon = recon

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def _recon_loss(self, x, x_hat):
        """Sum-then-mean reconstruction loss over the batch."""
        if self.recon == "mse":
            return tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
        if self.recon == "mae":
            return tf.reduce_mean(tf.reduce_sum(tf.abs(x - x_hat), axis=1))
        if self.recon == "huber":
            delta = 1.0
            diff = tf.abs(x - x_hat)
            elem = tf.where(diff <= delta,
                            0.5 * tf.square(diff),
                            delta * (diff - 0.5 * delta))
            return tf.reduce_mean(tf.reduce_sum(elem, axis=1))
        raise ValueError(f"Unknown recon loss: {self.recon}")

    def _forward_loss(self, x, training):
        """Shared forward pass + ELBO computation."""
        mu, log_var, z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        recon = self._recon_loss(x, x_hat)
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        return recon + self.beta * kl, recon, kl

    def _update_trackers(self, loss, recon, kl):
        self.total_loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        with tf.GradientTape() as tape:
            loss, recon, kl = self._forward_loss(x, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._update_trackers(loss, recon, kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        loss, recon, kl = self._forward_loss(x, training=False)
        self._update_trackers(loss, recon, kl)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        _, _, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)
