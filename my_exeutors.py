import os
from typing import Dict

import numpy as np
from jina import Executor, requests
from docarray import DocumentArray


class MyExec(Executor):
    @requests(
        on=['/test']
    )
    def foo(self, docs: 'DocumentArray', **kwargs):
        print(f'Calling foo')
        return docs


class MyIndexer(Executor):
    """
    Executor with basic exact search using cosine distance
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._docs = DocumentArray()

    @requests(on='/index')
    def index(self, docs: 'DocumentArray', **kwargs):
        """Extend self._docs
        :param docs: DocumentArray containing Documents
        :param kwargs: other keyword arguments
        """
        self._docs.extend(docs)

    @requests(on=['/match'])
    def search(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        """Append best matches to each document in docs
        :param docs: documents that are searched
        :param parameters: dictionary of pairs (parameter,value)
        :param kwargs: other keyword arguments
        """
        docs[0].match(
            self._docs,
            metric='cosine',
            normalization=(1, 0)
        )

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