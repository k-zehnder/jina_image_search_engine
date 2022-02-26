import os
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
import cv2
import imutils

from image_helpers.resultsmontage import ResultsMontage
from image_helpers.utils import print_response_parameters, print_match_results, preproc
from my_exeutors import MyMeans, MyIndexer



indexing_documents = DocumentArray.from_files("./data/flag_imgs/*.jpg", size=10000)
indexing_documents.apply(preproc)
model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
indexing_documents.embed(model, device="cpu", to_numpy=True)

f = (
    Flow(port_expose=8080, protocol='http')
    .add(uses=MyIndexer)
    .add(uses=MyMeans)
)


query = DocumentArray(indexing_documents[0])

with f:
    f.post("/index", inputs=indexing_documents)
    res = f.post("/search", parameters={'limit': 9}, inputs=query)
    f.post("/status", inputs=[])

    means = f.post("/means", inputs=DocumentArray(indexing_documents))
    print("means", means[0].text)

    # load the query image, display it, describe it
    print("[INFO] describing query...")
    query_cv2 = cv2.imread(query[0].uri)
    print(query_cv2)
    cv2.imshow("Query", imutils.resize(query_cv2, width=200))
    cv2.waitKey(0)
    
    res = res.to_dict()[0]
    montage = ResultsMontage((240, 320), 5, 20)
    for i, m in enumerate(res["matches"]):
        print(f"query_uri: {query[0].uri}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")
        result = cv2.imread(m["uri"]) 
        score = m['scores']['cosine']['value']
        montage.addResult(result, text=f"#{i+1} > {score:.2f})")

        # show the output image of results
        cv2.imshow("Results", imutils.resize(montage.montage, height=200))
        cv2.waitKey(0)
