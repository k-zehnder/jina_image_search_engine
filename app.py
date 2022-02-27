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
    try:
        with f:
            f.post("/index", inputs=my_input(DATA_DIR))
            
            query = preprocess_img("/home/inthrustwetrust71/Desktop/jina_image_search_engine/data/flag_imgs/france_6.jpg")
            print(query)

            # res = f.post("/search", parameters={'limit': 9}, inputs=query)

            # f.post("/status", inputs=[])

            # f.post("/means", inputs=my_input(DATA_DIR), on_done=print_mean_results)

        # show_montage(query, res)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

