import numpy as np
from .autograd import Tensor
import os
import pickle
import struct, gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
          return img[:, ::-1, :]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        H, W, C = img.shape
        npad = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        padded_img = np.pad(img, npad, 'constant')
        res = padded_img[self.padding + shift_x : H + shift_x + self.padding,
                         self.padding + shift_y : W + shift_y + self.padding, :]
        return res


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
        dtype="float32"
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        if self.shuffle:
          shuffle_order = np.random.permutation(len(self.dataset))
          self.ordering = np.array_split(shuffle_order, range(self.batch_size, len(self.dataset), self.batch_size))
        self.num_batch = len(self.ordering)
        self.num_iter = 0
        return self

    def __next__(self):
        if self.num_iter >= self.num_batch:
          raise StopIteration
        else:
          data_batch = self.dataset[self.ordering[self.num_iter]]
          self.num_iter += 1
        return tuple(Tensor(i, device=self.device, dtype=self.dtype) for i in data_batch)
    
    def __len__(self) -> int:
      return self.num_batch


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(label_filename, 'rb') as lbpath:
          magic, n = struct.unpack('>II',lbpath.read(8))
          self.labels = np.frombuffer(lbpath.read(),dtype=np.uint8)

        with gzip.open(image_filename, 'rb') as imgpath:
          magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
          images = np.frombuffer(imgpath.read(),dtype=np.uint8).reshape(len(self.labels), 28 * 28)
          images = images / np.max(images)
          self.imgs = np.float32(images).reshape((num, rows, cols, 1))

        self.transforms = transforms

    def __getitem__(self, index) -> object:
        img = self.imgs[index]
        if self.transforms:
          for transform in self.transforms:
            img = transform(img)
        return img, self.labels[index]

    def __len__(self) -> int:
        return self.imgs.shape[0]


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        self.p = p
        self.transforms = transforms

        if train:
          file_names = ["data_batch_" + str(i) for i in range(1, 6)]
        else:
          file_names = ["test_batch"]
        
        X = []
        y = []
        for file_name in file_names:
          with open(base_folder + "/" + file_name, mode='rb') as f:
            batch = pickle.load(f, encoding='latin1')
          features = (batch['data'] / 255).reshape((len(batch['data']), 3, 32, 32))
          labels = np.array(batch['labels'])
          X.append(features)
          y.append(labels)
        self.X = np.concatenate(X, axis=0)
        self.y = np.concatenate(y, axis=0)


    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        if isinstance(index, slice):
            img_slice = self.X[index.start:index.stop:index.step]
            y_slice = self.y[index.start:index.stop:index.step]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        elif isinstance(index, list):
            img_slice = self.X[index]
            y_slice = self.y[index]
            if self.transforms is not None and RandomCrop.random() < self.p:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        else:
            img = self.X[index]
            if self.transforms is not None and RandomCrop.random() < self.p:
                for transform in self.transforms:
                    img = transform(img)
            return (img, self.y[index])

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.y.shape[0]


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
          index = len(self.idx2word)
          self.idx2word.append(index)
          self.word2idx[word] = index
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.word2idx)



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        with open(path, 'r') as f:
            lines = f.readlines()[:max_lines]
            for l in lines:
                print(l)
                for word in l.split():
                    self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx[word])
                self.dictionary.add_word('<eos>')
                ids.append(self.dictionary.word2idx['<eos>'])
        return ids


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    data = np.array(data, dtype=dtype)
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size].reshape((batch_size, nbatch)).transpose()
    return data


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    nbatch = len(batches)
    seq_len = min(bptt, nbatch-1-i)
    data = batches[i:i+seq_len]
    target = batches[i+1:i+1+seq_len].flatten()
    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)

    