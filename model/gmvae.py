import tensorflow as tf

from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Model


class GMVAE(Model):
    def __init__(self, params):
        """
        z2 -> w
        y  -> z
        z1 -> x
        x  -> y

        p(y)p(z_2)p(z_1|y, z_2)p(x|z_1)
        p(z)p(w)  p(x  |z, w)  p(y|x)
        """
        super(GMVAE, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def call(self, inputs, training=None, mask=None):
        out_encoder = self.encoder(inputs, training)
        out_decoder = self.decoder(out_encoder, training)
        return out_encoder, out_decoder


class Encoder(Model):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.epsilon = params["epsilon"]

        self.batch_size = params["batch_size"]

        self.num_clusters = params["num_clusters"]

        self.fc11_hidden_dim = params["enc_fc11_hidden_dim"]
        self.fc12_hidden_dim = params["enc_fc12_hidden_dim"]

        self.fc21_hidden_dim = params["enc_fc21_hidden_dim"]
        self.fc22_hidden_dim = params["enc_fc22_hidden_dim"]
        self.gaussian_hidden_dim = params["enc_gaussian_hidden_dim"]

        self.activation = params["activation"]

        # Fully Connected Layer: Input to feature
        self.fc11 = Dense(units=self.fc11_hidden_dim, activation=self.activation)
        self.fc12 = Dense(units=self.fc12_hidden_dim, activation=self.activation)
        self.out = Dense(units=self.num_clusters)
        self.softmax = Softmax()

        # Fully Connected Layer: Learn Gaussian Distribution
        self.fc21 = Dense(units=self.fc21_hidden_dim, activation=self.activation)
        self.fc22 = Dense(units=self.fc22_hidden_dim, activation=self.activation)
        self.mean = Dense(units=self.gaussian_hidden_dim)
        self.variance = Dense(units=self.gaussian_hidden_dim, activation=tf.nn.softplus)

    # Citation: https://github.com/vithursant/VAE-Gumbel-Softmax/blob/master/vae_gumbel_softmax.py
    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        u = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(u + eps) + eps)

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
        y = tf.nn.softmax(gumbel_softmax_sample / temperature)
        if hard:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def call(self, inputs, training=None, mask=None):
        logits = self.out(self.fc12(self.fc11(inputs)))
        probs = self.softmax(logits)

        log_probs = tf.math.log(probs + self.epsilon)
        y = self.gumbel_softmax(logits, hard=True)

        concat = tf.concat([y, inputs], 1)
        z_input = self.fc22(self.fc21(concat))
        z_mean = self.mean(z_input)
        z_variance = self.variance(z_input)
        noise = tf.random.normal(tf.shape(z_mean), mean=0, stddev=1)
        gaussian_sample = z_mean + tf.sqrt(z_variance + self.epsilon) * noise

        out_dict = {"gaussian_sample": gaussian_sample, "y": y, "mean": z_mean, "var": z_variance, "logits": logits,
                    "probs": probs, "log_probs": log_probs}

        return out_dict


class Decoder(Model):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.epsilon = params["epsilon"]

        self.loss_type = params["loss_type"]

        self.batch_size = params["batch_size"]

        self.output_size = params["n_features"]
        self.gaussian_hidden_dim = params["dec_gaussian_hidden_dim"]

        self.activation = params["activation"]

        # For P(z|y)
        self.mean = Dense(units=self.gaussian_hidden_dim)
        self.variance = Dense(units=self.gaussian_hidden_dim, activation=tf.nn.softplus)

        # For P(x|z)
        self.fc1_hidden_dim = params["dec_fc1_hidden_dim"]
        self.fc2_hidden_dim = params["dec_fc2_hidden_dim"]

        self.fc1 = Dense(units=self.fc1_hidden_dim, activation=self.activation)
        self.fc2 = Dense(units=self.fc2_hidden_dim, activation=self.activation)
        self.out = Dense(units=self.output_size)

    def call(self, inputs, training=None, mask=None):
        y = inputs["y"]
        gaussian_sample = inputs["gaussian_sample"]

        mean = self.mean(y)
        variance = self.variance(y)

        logits = self.out(self.fc2(self.fc1(gaussian_sample)))

        if self.loss_type == "bce":
            return {"logits": logits, "out": tf.nn.sigmoid(logits), "y_mean": mean, "y_variance": variance}
        else:
            return {"logits": logits, "out": logits, "y_mean": mean, "y_variance": variance}
