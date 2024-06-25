# CIFAR-100
def unpickle(file, encoding='bytes'):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding=encoding)
    return dict

if __name__ == "__main__":
    d = unpickle('./cifar-100-python/meta', 'ASCII')
    print(d['fine_label_names'][19])
    d = unpickle('./cifar-100-python/train')

    print(d.keys())
    print(d[b'data'][0].shape)
    print(d[b'fine_labels'][0])
    print(d[b'coarse_labels'][0])