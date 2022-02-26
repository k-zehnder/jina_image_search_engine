# USAGE
# python search.py --index index.csv --dataset ../ukbench --relevant ../ukbench/relevant.json \
# 	--query ../ukbench/ukbench00980.jpg

# import the necessary packages
from __future__ import print_function
import os
from resultsmontage import ResultsMontage
import argparse
import imutils
import json
import cv2
import torchvision
from shutil import rmtree
from docarray import DocumentArray, Document
from my_exeutors import MyMeans, MyIndexer


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--index", required=True, help="Path to where the features index will be stored")
# ap.add_argument("-q", "--query", required=True, help="Path to the query image")
# ap.add_argument("-d", "--dataset", required=True, help="Path to the original dataset directory")
# args = vars(ap.parse_args())


workspace = './workspace'
os.environ['JINA_WORKSPACE'] = workspace

if os.path.exists(workspace):
    print(f'Workspace at {workspace} exists. Will delete')
    rmtree(workspace)

# Convert to tensor, normalize so they're all similar enough
def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_shape((80, 60))  # ensure all images right size (dataset image size _should_ be (80, 60))
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis for the PyTorch model later

indexing_documents = DocumentArray.from_files("./data/flag_imgs/*.jpg", size=10000)
indexing_documents.apply(preproc)
model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
indexing_documents.embed(model, device="cpu", to_numpy=True)

if __name__ == "__main__":
    indexing_documents = DocumentArray.from_files("./data/flag_imgs/*.jpg", size=10000)
    indexing_documents.apply(preproc)
    model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
    indexing_documents.embed(model, device="cpu", to_numpy=True)

    m = MyMeans()
    means = m.means(indexing_documents)
    print(means[0].text)

    idxer = MyIndexer()
    idxer.index(indexing_documents)

    q = indexing_documents[0]
    q = DocumentArray(q)

    res = idxer.search(q, parameters={'limit': 9})
    res = res.to_dict()[0]

    # load the query image, display it, describe it
    print("[INFO] describing query...")
    query = cv2.imread(q[0].uri)
    print(query)
    cv2.imshow("Query", imutils.resize(query, width=200))
    cv2.waitKey(0)

    montage = ResultsMontage((240, 320), 5, 20)
    for i, m in enumerate(res["matches"]):
        print(f"query_uri: {q[0].uri}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")
        result = cv2.imread(m["uri"])
        montage.addResult(result, text=f"#{i+1}")

        # show the output image of results
        cv2.imshow("Results", imutils.resize(montage.montage, height=200))
        cv2.waitKey(0)

    # for m in res["matches"]:
    #     print(f"query_uri: {q[0].uri}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")q