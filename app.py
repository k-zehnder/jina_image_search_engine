import os
import glob
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
import cv2
import imutils


from image_helpers.resultsmontage import ResultsMontage
from image_helpers.utils import print_response_parameters, print_match_results, show_montage
from executors.my_exeutors import MyMeans, MyIndexer


DATA_DIR = "./data/flag_imgs/*.jpg"


def print_mean_results(resp):
    print(resp.to_dict()["data"][0]["text"])

f = (
    Flow(port_expose=8080, protocol='http')
    .add(uses=MyIndexer)
    .add(uses=MyMeans)
)


def my_input(DATA_DIR):
    image_uris = glob.glob(DATA_DIR)
    for image_uri in image_uris:
        yield Document(uri=image_uri)

def main() -> None:
    with f:
        returned_query = f.post("/index", inputs=my_input(DATA_DIR))
        
        res = f.post("/search", parameters={'limit': 9}, inputs=returned_query)

        f.post("/status", inputs=[])

        f.post("/means", inputs=my_input(DATA_DIR), on_done=print_mean_results)

    show_montage(returned_query, res)


if __name__ == "__main__":
    main()
