import os
import glob
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
import cv2
import imutils


from image_helpers.resultsmontage import ResultsMontage
from image_helpers.utils import print_response_parameters, print_match_results, print_mean_results, show_montage, my_input, preprocess_img
from executors.my_exeutors import MyMeans, MyIndexer


DATA_DIR = "./data/flag_imgs/*.jpg"


f = (
    Flow(port_expose=8080, protocol='http')
    .add(uses=MyIndexer)
    .add(uses=MyMeans)
)

def main() -> None:
    # query should be in flow, but for clarity in client
    query = preprocess_img("france_6.jpg") 

    with f:
        f.post("/index", inputs=my_input(DATA_DIR))
        
        res = f.post("/search", parameters={'limit': 9}, inputs=query)

        f.post("/status", inputs=[])

        f.post("/means", inputs=my_input(DATA_DIR), on_done=print_mean_results)

    show_montage(query, res)

if __name__ == "__main__":
    main()

