# CIFAR-100
def unpickle(file, encoding='bytes'):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding=encoding)
    return dict

if __name__ == "__main__":
    d = unpickle('./cifar-100-python/meta', 'ASCII')

    d = unpickle('./cifar-100-python/train')

    print(d[b'data'][0].shape)