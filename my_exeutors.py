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

    def __init__(self, parameter=0, **kwargs):
        super().__init__(**kwargs)
        self._docs = DocumentArray()
        self.result = DocumentArray()
        self.parameter = parameter

    @requests(on='/index')
    def index(self, docs: 'DocumentArray', **kwargs):
        """Extend self._docs
        :param docs: DocumentArray containing Documents
        :param kwargs: other keyword arguments
        """
        self._docs.extend(docs)

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        # construct left/right document arrays
        right_da = self._docs
        left_da = DocumentArray(docs[0]) # query

        # match document arrays by use of embedding (feature vector similarity)
        left_da.match(right_da, limit=parameters["limit"])

        # AFTER matching, persist state of query as result
        self.result.extend(left_da)
     
    @requests(on='/status')
    def status(self, **kwargs):
        """
        Display status of object
        """
        return {"internal_parameter": self.parameter}

    @requests(on='/returner')
    def returner(self, **kwargs):
        """
        Return result stored as class attribute
        """
        return self.result

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