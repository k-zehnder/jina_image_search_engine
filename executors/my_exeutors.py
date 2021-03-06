from typing import Dict
import numpy as np
from imutils import paths

from jina import Executor, requests
from docarray import DocumentArray, Document

from image_helpers import utils


DATA_DIR = "./data/flag_imgs/left/*.jpg"
DATA_DIR_RIGHT = "./data/flag_imgs/right/*.jpg" 
DATA_DIR_AUGMENTED_RIGHT = "./data/flag_imgs/augmented_right/*.jpg" 


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
        """
        :param docs: DocumentArray containing Documents
        :param kwargs: other keyword arguments
        """
        self._docs.extend(utils.prepare_docs(docs))

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        # construct left/right document arrays
        right_da = self._docs
        left_da = DocumentArray(docs[0]) # query

        # match query to docs we have already indexed at "/index"
        left_da.match(right_da, limit=parameters["limit"])

        utils.print_match_results(left_da)
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

class MySimulator(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @requests(on='/evaluate')
    def evaluate(self, **kwargs):
        # read
        left_da = DocumentArray.from_files(DATA_DIR)
        right_da = DocumentArray.from_files(DATA_DIR_AUGMENTED_RIGHT)

        # prepare
        left_da = utils.prepare_docs(left_da)
        right_da = utils.prepare_docs(right_da)
        
        # match
        left_da.match(right_da, limit=9)
        groundtruth = DocumentArray(
            Document(uri=d.uri, matches=[Document(uri=d.uri.replace('left', 'augmented_right'))]) for d in left_da)

       # evaluate
        for k in range(1, 6):
            print(f'precision@{k}',
                left_da.evaluate(
                    groundtruth,
                    hash_fn=lambda d: d.uri,
                    metric='precision_at_k',
                    k=k,
                    max_rel=1))

class MyMeans(Executor):
    """
    Executor with basic mean height (note: not actually mean of pixels)
    """
    def __init__(self, parameter=True, **kwargs):
        super().__init__(**kwargs)
        self.parameter = parameter

    @requests(on='/means')
    def means(self, docs: 'DocumentArray', **kwargs):
        heights = [np.mean(doc.tensor) for doc in utils.prepare_docs(docs)]
        format_txt = f"mean of means for pixel height: {np.mean(heights):.2f}"
        return DocumentArray(Document(text=format_txt))        

