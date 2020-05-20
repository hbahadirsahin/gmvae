import ujson

import tensorflow as tf

from dataloader.handle_input import InputHandler
from model.gmvae import GMVAE
from loss.loss_function import Loss

from utils.logger import Logger


class Pipeline(Logger):
    def __init__(self):
        self.config_path = "./config/config.json"
        self.input_path = "./data/traj.npy"

    def main(self):
        with open(self.config_path, 'r') as f:
            config = ujson.load(f)

        input_config = config["input_config"]
        model_config = config["model_config"]
        training_config = config["training_config"]

        self.logger.info("Reading input file and generating train/test splits!")
        input_handler = InputHandler(input_path=self.input_path,
                                     use_mnist=input_config["use_mnist"],
                                     split_vali=input_config["split_vali"],
                                     batch_size=input_config["batch_size"],
                                     test_size=input_config["test_size"])

        train_ds, test_ds, vali_ds = input_handler.create_tf_dataset()
        n_features = input_handler.initial_data_shape[1]
        self.logger.info("Dataset loaded!")

        model_config["batch_size"] = input_config["batch_size"]
        model_config["n_features"] = n_features
        model_config["activation"] = tf.nn.relu
        model_config["loss_type"] = "mse"

        self.logger.info("GMVAE Model is being initialized!")
        gmvae = GMVAE(model_config)
        self.logger.info("GMVAE Model has been initialized!")
        self.logger.info("Loss Object is being initialized!")
        loss = Loss(model_config)
        self.logger.info("Loss Object has been initialized!")
        self.logger.info("Optimizer is being initialized!")
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_config["learning_rate"])
        self.logger.info("Optimizer has been initialized!")

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for i in range(training_config["max_iter"]):
            train_loss.reset_states()
            val_loss.reset_states()
            test_loss.reset_states()

            train_accuracy.reset_states()
            val_accuracy.reset_states()
            test_accuracy.reset_states()

            for train_idx, train_x in enumerate(train_ds):
                if input_config["use_mnist"]:
                    self.train(train_x[0],
                               model=gmvae,
                               optimizer=optimizer,
                               loss=loss,
                               loss_metric=train_loss,
                               labels=train_x[1],
                               accuracy_metric=train_accuracy)
                else:
                    self.train(train_x, model=gmvae, optimizer=optimizer, loss=loss, loss_metric=train_loss)
                if train_idx % 1000 == 0:
                    self.logger.info("Training at Epoch {} - Batch id {}".format(i + 1, train_idx + 1))

            if input_config["split_vali"]:
                for vali_idx, vali_x in enumerate(vali_ds):
                    if input_config["use_mnist"]:
                        self.evaluate(vali_x[0],
                                      model=gmvae,
                                      loss=loss,
                                      loss_metric=val_loss,
                                      labels=vali_x[1],
                                      accuracy_metric=val_accuracy)
                    else:
                        self.evaluate(vali_x, model=gmvae, loss=loss, loss_metric=val_loss)
                    if vali_idx % 1000 == 0:
                        self.logger.info("Validation at Epoch {} - Batch id {}".format(i + 1, vali_idx + 1))

            template = "Epoch {}, Training Loss: {}, Validation Loss: {}, Train Accuracy: {}, Validation Accuracy: {}"
            if input_config["split_vali"]:
                self.logger.info(template.format(i+1,
                                                 train_loss.result(),
                                                 val_loss.result(),
                                                 train_accuracy.result(),
                                                 val_accuracy.result()))
            else:
                self.logger.info(template.format(i+1, train_loss.result(), 0, train_accuracy.result(), 0))

        for test_idx, test_x in enumerate(test_ds):
            if input_config["use_mnist"]:
                self.evaluate(test_x[0],
                              model=gmvae,
                              loss=loss,
                              loss_metric=test_loss,
                              labels=test_x[1],
                              accuracy_metric=test_accuracy)
            else:
                self.evaluate(test_x, model=gmvae, loss=loss, loss_metric=test_loss)
        self.logger.info("Test Loss: {}, Test Accuracy: {}".format(test_loss.result(), test_accuracy.result()))

    @tf.function
    def train(self, train_data, model, optimizer, loss, loss_metric, labels=None, accuracy_metric=None):
        with tf.GradientTape() as tape:
            out_encoder, out_decoder = model(train_data, training=True)
            out_loss = loss.unlabeled_loss(train_data, out_encoder, out_decoder)
        gradients = tape.gradient(out_loss["total_loss"], model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_metric(out_loss["total_loss"])
        if accuracy_metric:
            accuracy_metric(tf.reshape(labels, [labels.shape.as_list()[0], -1]),
                            tf.reshape(labels, [out_loss["predicted"].shape.as_list()[0], -1]))

    @tf.function
    def evaluate(self, eval_data, model, loss, loss_metric, labels=None, accuracy_metric=None):
        out_encoder, out_decoder = model(eval_data, training=False)
        out_loss = loss.unlabeled_loss(eval_data, out_encoder, out_decoder)
        loss_metric(out_loss["total_loss"])
        if accuracy_metric:
            accuracy_metric(tf.reshape(labels, [labels.shape.as_list()[0], -1]),
                            tf.reshape(labels, [out_loss["predicted"].shape.as_list()[0], -1]))


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.main()
