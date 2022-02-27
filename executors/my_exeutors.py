import os
from sqlite3 import paramstyle
from typing import Dict
import torchvision 

import numpy as np
from jina import Executor, requests
from docarray import DocumentArray, Document

from image_helpers import utils


DATA_DIR = "./data/flag_imgs/*.jpg"


class MyMeans(Executor):
    """
    Executor with basic mean height (note: not actually mean of pixels)
    """
    def __init__(self, parameter=True, **kwargs):
        super().__init__(**kwargs)
        self.parameter = parameter

    @requests(on='/means')
    def means(self, docs: 'DocumentArray', **kwargs):
        docs.apply(utils.preproc)
        model = torchvision.models.resnet50(pretrained=True)
        docs.embed(model, device="cpu", to_numpy=True)
        heights = [np.mean(doc.tensor) for doc in docs]
        format_txt = f"mean of means for pixel height: {np.mean(heights)}"
        return DocumentArray(Document(text=format_txt))        

class MyIndexer(Executor):
    """
    Executor with basic exact search using cosine distance
    """

    def __init__(self, parameter=0, **kwargs):
        super().__init__(**kwargs)
        self._docs = DocumentArray()
        self.parameter = parameter

    @requests(on='/index')
    def index(self, docs: 'DocumentArray', **kwargs):
        """Extend self._docs
        :param docs: DocumentArray containing Documents
        :param kwargs: other keyword arguments
        """
        docs.apply(utils.preproc)
        model = torchvision.models.resnet50(pretrained=True)
        docs.embed(model, device="cpu", to_numpy=True)
        self._docs.extend(docs)

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        # construct left/right document arrays
        right_da = self._docs
        left_da = DocumentArray(docs[0]) # query

        # match document arrays by use of embedding (feature vector similarity)
        left_da.match(right_da, limit=parameters["limit"])

        # return AFTER matching
        return DocumentArray(left_da)
     
    @requests(on='/status')
    def status(self, **kwargs):
        """
        Display status of object
        """
        return {"internal_parameter": self.parameter}

    def close(self):
        """
        Stores the DocumentArray to disk
        """
        self._docs.save("./")

class MyConverter(Executor):
    """
    Convert DocumentArrays removing tensor and reshaping tensor as image
    """

    @requests
    def convert(self, docs: 'DocumentArray', **kwargs):
        """
        Remove tensor and reshape documents as squared images
        :param docs: documents to modify
        :param kwargs: other keyword arguments
        """
        for doc in docs:
            doc.convert_image_tensor_to_uri()
            doc.pop('tensor')