import tensorflow as tf
import numpy as np


class Loss:
    def __init__(self, params):
        self.epsilon = params["epsilon"]

        self.loss_type = params["loss_type"]
        self.output_size = params["n_features"]

        self.regularization_term_reconstruction = params["regularization_term_reconstruction"]
        self.regularization_term_gauss = params["regularization_term_gauss"]
        self.regularization_term_category = params["regularization_term_category"]

    def unlabeled_loss(self, input, encoder_out, decoder_out):
        gaussian_sample = encoder_out["gaussian_sample"]
        z_mean = encoder_out["mean"]
        z_variance = encoder_out["var"]
        encoder_logits = encoder_out["logits"]
        y = encoder_out["y"]
        probs = encoder_out["probs"]
        log_probs = encoder_out["log_probs"]

        output = decoder_out["out"]
        y_mean = decoder_out["y_mean"]
        y_variance = decoder_out["y_variance"]

        reconstruction_loss = self.regularization_term_reconstruction * self.mean_squared_error(input, output)
        gaussian_loss = self.regularization_term_gauss * self.labeled_loss(gaussian_sample,
                                                                           z_mean,
                                                                           z_variance,
                                                                           y_mean,
                                                                           y_variance)
        categorical_loss = self.regularization_term_category * -self.cross_entropy_with_logits(encoder_logits, probs)

        total_loss = reconstruction_loss + gaussian_loss + categorical_loss

        return {"total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "gaussian_loss": gaussian_loss,
                "categorical_loss": categorical_loss}

    @staticmethod
    def labeled_loss(gaussian_sample, z_mean, z_variance, y_mean, y_variance):
        loss = -0.5 \
               * tf.reduce_sum(tf.math.log(2 * np.pi) + tf.math.log(z_variance) + tf.square(gaussian_sample - z_mean) / z_variance,
                               axis=1) \
               + 0.5 \
               * tf.reduce_sum(tf.math.log(2 * np.pi) + tf.math.log(y_variance) + tf.square(gaussian_sample - y_mean) / y_variance,
                               axis=1)
        loss -= tf.math.log(0.1)
        return tf.reduce_mean(loss)

    @staticmethod
    def mean_squared_error(ground_truth, predictions):
        return tf.math.reduce_mean(tf.square(ground_truth - predictions))

    @staticmethod
    def cross_entropy_with_logits(logits, targets):
        log_prob = tf.nn.log_softmax(logits)
        return -tf.reduce_mean(tf.reduce_sum(targets * log_prob, 1))
