import os
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
import cv2
import imutils

from .resultsmontage import ResultsMontage


DATA_DIR = "./data/flag_imgs/*.jpg"


def generate_docs(DATA_DIR, num_docs=10000):
    indexing_documents = DocumentArray.from_files(DATA_DIR, size=num_docs)
    indexing_documents.apply(preproc)
    model = torchvision.models.resnet50(pretrained=True)
    indexing_documents.embed(model, device="cpu", to_numpy=True)
    return indexing_documents

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

# Convert to tensor, normalize so they're all similar enough
def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_shape((80, 60))  # ensure all images right size (dataset image size _should_ be (80, 60))
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis for the PyTorch model later

# TODO: functionalize me
# print(f"query_uri: {query[0].uri}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")