import torch
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from .dataset import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_datasets(batch_size=32, max_length=120):
    imdb, _ = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_dataset = imdb['train']
    test_dataset = imdb['test']
    vocab_set = set()
    tokenizer = tfds.deprecated.text.Tokenizer()

    for _, data in enumerate(train_dataset):
        text, label = data
        label = label.numpy()
        examples = str(text.numpy().decode())
        examples = tokenizer.tokenize(examples)
        examples = np.reshape(examples, (-1)).tolist()
        vocab_set.update(examples)

    vocab_set = list(set(vocab_set))
    encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

    def tf_encode(x):
        result = tf.py_function(lambda s: tf.constant(encoder.encode(s.numpy())), [
            x,
        ], tf.int32)
        result.set_shape([None])
        return result

    def tokenize(text, label):
        print(label)
        return {
            'text': tf_encode(text)[:max_length],
            'label': [label]
        }

    train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

    max_shape = {'text': [max_length], 'label': [1]}
    train_dataset = train_dataset.shuffle(
        buffer_size=1024, reshuffle_each_iteration=True).padded_batch(
        batch_size, padded_shapes=max_shape)
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=max_shape)

    test_dataset = tfds.as_numpy(test_dataset)
    train_dataset =tfds.as_numpy(train_dataset)

    return train_dataset, test_dataset


class IMDBDataset(Dataset):
    def __init__(self, batch_size, max_length, patch_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.max_length = max_length
        self.patch_size = patch_size

        self.d_train, self.d_test, = get_datasets(batch_size=batch_size, max_length=max_length)
        self.test_enum = iter(self.d_test)
        self.train_enum = iter(self.d_train)

    def reset_test(self):
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            batch = next(self.train_enum, None)
            if batch is None:
                self.train_enum = iter(self.d_train)
                batch = next(self.train_enum)
        else:
            batch = next(self.test_enum, None)
            if batch is None:
                self.test_enum = iter(self.d_test)
                batch = next(self.test_enum)
        x, y = batch['text'], batch['label']
        x = np.expand_dims(x, axis=1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y.squeeze()).float()
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        self._ind += 1
        return x, y
