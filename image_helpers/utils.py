import os
import glob
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
import cv2
import imutils

from .resultsmontage import ResultsMontage


def my_input(DATA_DIR):
    image_uris = glob.glob(DATA_DIR)
    for image_uri in image_uris:
        yield Document(uri=image_uri)

def print_mean_results(resp):
    print(resp.to_dict()["data"][0]["text"])

def print_response_parameters(resp):
    print(f'{resp.to_dict()["parameters"]}')

def print_match_results(resp):
    # resp is <jina.types.request.data.DataRequest>

    data = resp.to_dict()["data"]
    for d in data:
        for m in d["matches"]:
            print(f"query_uri: {d['uri']}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")

def show_montage(query, res):
    # load the query image, display it
    query_cv2 = cv2.imread(query[0].uri)
    cv2.imshow("Query", imutils.resize(query_cv2, width=200))
    cv2.waitKey(0)
    
    res = res.to_dict()[0]
    montage = ResultsMontage((240, 320), 5, 20)
    for i, m in enumerate(res["matches"]):
        result = cv2.imread(m["uri"]) 
        score = m['scores']['cosine']['value']
        montage.addResult(result, text=f"#{i+1} > {score:.2f}")

        # show the output image of results
        cv2.imshow("Results", imutils.resize(montage.montage, height=500))
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# TODO: functionalize me
# print(f"query_uri: {query[0].uri}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")