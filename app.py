import os
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
import cv2
import imutils

from image_helpers.resultsmontage import ResultsMontage
from image_helpers.utils import print_response_parameters, print_match_results, preproc, show_montage, generate_docs
from my_exeutors import MyMeans, MyIndexer


DATA_DIR = "./data/flag_imgs/*.jpg"


def print_mean_results(resp):
    print(resp.to_dict()["data"][0]["text"])

f = (
    Flow(port_expose=8080, protocol='http')
    .add(uses=MyIndexer)
    .add(uses=MyMeans)
)

def main(task: str) -> None: # args
    indexing_documents = generate_docs(DATA_DIR)
    query = DocumentArray(indexing_documents[0])

    with f:
        # index()
        f.post("/index", inputs=indexing_documents)
        
        # search(limit)
        res = f.post("/search", parameters={'limit': 9}, inputs=query)
        f.post("/status", inputs=[])

        means = f.post("/means", inputs=DocumentArray(indexing_documents), on_done=print_mean_results)
        # print("means", means[0].text)

    show_montage(query, res)


if __name__ == "__main__":
    main("test")