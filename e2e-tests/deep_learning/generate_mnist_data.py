from torchvision import datasets


def generate_train_test_data():
    train = datasets.MNIST("data", train=True, download=True)
    test = datasets.MNIST("data", train=False)

    train.data.numpy().tofile("data/train_data.dat")
    train.targets.numpy().tofile("data/train_targets.dat")

    test.data.numpy().tofile("data/test_data.dat")
    test.targets.numpy().tofile("data/test_targets.dat")


generate_train_test_data()
