import os
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
from my_exeutors import MyMeans, MyIndexer
from resultsmontage import ResultsMontage
import cv2
import imutils

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

f = (
    Flow(port_expose=8080, protocol='http')
    .add(uses=MyIndexer)
    .add(uses=MyMeans)
)

def print_response_parameters(resp):
    print(f' {resp.to_dict()["parameters"]}')

def print_match_results(resp):
    # resp is <jina.types.request.data.DataRequest>

    data = resp.to_dict()["data"]
    for d in data:
        for m in d["matches"]:
            print(f"query_uri: {d['uri']}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")


query = DocumentArray(indexing_documents[0])

with f:
    f.post("/index", inputs=indexing_documents)
    res = f.post("/search", parameters={'limit': 9}, inputs=query)
    # f.post("/status", inputs=[])

    # means = f.post("/means", inputs=DocumentArray(indexing_documents))
    # print("means", means[0].text)

    # load the query image, display it, describe it
    print("[INFO] describing query...")
    query = cv2.imread(query[0].uri)
    print(query)
    cv2.imshow("Query", imutils.resize(query, width=200))
    cv2.waitKey(0)
    
    res = res.to_dict()[0]
    montage = ResultsMontage((240, 320), 5, 20)
    for i, m in enumerate(res["matches"]):
        print(f"query_uri: query, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")
        result = cv2.imread(m["uri"])
        montage.addResult(result, text=f"#{i+1}")

        # show the output image of results
        cv2.imshow("Results", imutils.resize(montage.montage, height=200))
        cv2.waitKey(0)
