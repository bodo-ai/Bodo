import sys

import horovod.tensorflow as hvd
import numba
import numpy as np
import tensorflow as tf

import bodo

# Example adapted from https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_mnist.py


@tf.function
def training_step(mnist_model, loss, opt, images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(mnist_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value


def deep_learning(X_train, y_train, X_test, y_test):
    print(
        "[{}] X_train size is {}, y_train size is {}".format(
            bodo.get_rank(), len(X_train), len(y_train)
        )
    )
    accuracy = 0.0
    if hvd.is_initialized():  # ranks not using horovod (e.g. non-gpu ranks) skip
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.expand_dims(X_train, 3), y_train)
        )
        dataset = dataset.repeat().shuffle(10000).batch(128)

        mnist_model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
                tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        loss = tf.losses.SparseCategoricalCrossentropy()

        # Horovod: adjust learning rate based on number of GPUs.
        opt = tf.optimizers.Adam(0.001 * hvd.size())

        checkpoint_dir = "./checkpoints"
        checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

        # Horovod: adjust number of steps based on number of GPUs.
        for batch, (images, labels) in enumerate(dataset.take(400 // hvd.size())):
            loss_value = training_step(
                mnist_model, loss, opt, images, labels, batch == 0
            )

            if batch % 10 == 0 and hvd.local_rank() == 0:
                print("Step #%d\tLoss: %.6f" % (batch, loss_value))

        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting it.
        if hvd.rank() == 0:
            checkpoint.save(checkpoint_dir)

    if bodo.get_rank() == 0:
        mnist_model.compile(opt, loss, metrics=["accuracy"])
        results = mnist_model.evaluate(
            tf.expand_dims(X_test, 3), y_test, batch_size=128
        )
        print("Accuracy is", results[1])


@bodo.jit(cache=True)
def main(train_data_fname, train_targets_fname, test_data_fname, test_targets_fname):
    # See generate_mnist_data.py script in this directory to generate data
    X_train = np.fromfile(train_data_fname, np.uint8)
    X_train = X_train.reshape(60000, 28, 28)
    y_train = np.fromfile(train_targets_fname, np.int64)

    X_test = np.fromfile(test_data_fname, np.uint8)
    X_test = X_test.reshape(10000, 28, 28)
    X_test = ((X_test / 255) - 0.1307) / 0.3081
    X_test = X_test.astype(np.float32)
    y_test = np.fromfile(test_targets_fname, np.int64)
    X_test = bodo.gatherv(X_test)
    y_test = bodo.gatherv(y_test)

    # preprocessing: do image normalization in Bodo
    # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
    # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize
    # using mean=0.1307, std=0.3081
    X_train = ((X_train / 255) - 0.1307) / 0.3081
    X_train = X_train.astype(np.float32)

    bodo.dl.start("tensorflow")
    X_train = bodo.dl.prepare_data(X_train)
    y_train = bodo.dl.prepare_data(y_train)
    with bodo.objmode:
        deep_learning(X_train, y_train, X_test, y_test)
    bodo.dl.end()


if __name__ == "__main__":
    path = sys.argv[1]
    require_cache = False
    if len(sys.argv) > 2:
        require_cache = bool(sys.argv[2])
    train_data_fname = path + "/train_data.dat"
    train_targets_fname = path + "/train_targets.dat"
    test_data_fname = path + "/test_data.dat"
    test_targets_fname = path + "/test_targets.dat"
    main(train_data_fname, train_targets_fname, test_data_fname, test_targets_fname)
    if require_cache and isinstance(main, numba.core.dispatcher.Dispatcher):
        assert (
            main._cache_hits[main.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"
