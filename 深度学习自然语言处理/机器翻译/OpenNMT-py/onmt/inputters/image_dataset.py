# -*- coding: utf-8 -*-

import os

import torch
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase

# domain specific dependencies
try:
    from PIL import Image
    from torchvision import transforms
    import cv2
except ImportError:
    Image, transforms, cv2 = None, None, None


class ImageDataReader(DataReaderBase):
    """Read image data from disk.

    Args:
        truncate (tuple[int] or NoneType): maximum img size. Use
            ``(0,0)`` or ``None`` for unlimited.
        channel_size (int): Number of channels per image.

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing any of ``PIL``, ``torchvision``, or ``cv2`` fail.
    """

    def __init__(self, truncate=None, channel_size=3):
        self._check_deps()
        self.truncate = truncate
        self.channel_size = channel_size

    @classmethod
    def from_opt(cls, opt):
        return cls(channel_size=opt.image_channel_size)

    @classmethod
    def _check_deps(cls):
        if any([Image is None, transforms is None, cv2 is None]):
            cls._raise_missing_dep(
                "PIL", "torchvision", "cv2")

    def read(self, images, side, img_dir=None):
        """Read data into dicts.

        Args:
            images (str or Iterable[str]): Sequence of image paths or
                path to file containing audio paths.
                In either case, the filenames may be relative to ``src_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            img_dir (str): Location of source image files. See ``images``.

        Yields:
            a dictionary containing image data, path and index for each line.
        """
        if isinstance(images, str):
            images = DataReaderBase._read_file(images)

        for i, filename in enumerate(images):
            filename = filename.decode("utf-8").strip()
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                img_path = filename

            assert os.path.exists(img_path), \
                'img path %s not found' % filename

            if self.channel_size == 1:
                img = transforms.ToTensor()(
                    Image.fromarray(cv2.imread(img_path, 0)))
            else:
                img = transforms.ToTensor()(Image.open(img_path))
            if self.truncate and self.truncate != (0, 0):
                if not (img.size(1) <= self.truncate[0]
                        and img.size(2) <= self.truncate[1]):
                    continue
            yield {side: img, side + '_path': filename, 'indices': i}


def img_sort_key(ex):
    """Sort using the size of the image: (width, height)."""
    return ex.src.size(2), ex.src.size(1)


def batch_img(data, vocab):
    """Pad and batch a sequence of images."""
    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.zeros(len(data), c, h, w).fill_(1)
    for i, img in enumerate(data):
        imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
    return imgs


def image_fields(**kwargs):
    img = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=batch_img, sequential=False)
    return img
