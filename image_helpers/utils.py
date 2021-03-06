import glob
import torchvision
from docarray import DocumentArray, Document
import cv2
import imutils
from imutils import paths

from .resultsmontage import ResultsMontage


# Convert to tensor, normalize so they're all similar enough
def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_shape((80, 60))  # ensure all images right size (dataset image size _should_ be (80, 60))
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis for the PyTorch model later

def my_input(DATA_DIR):
    image_uris = glob.glob(DATA_DIR)
    for image_uri in image_uris:
        yield Document(uri=image_uri)

def prepare_docs(docs):
    docs.apply(preproc)
    model = torchvision.models.resnet50(pretrained=True)
    docs.embed(model, device="cpu", to_numpy=True)
    return docs

def preprocess_img(image):
    image_path = f"./data/flag_imgs/left/{image}"
    return prepare_docs(DocumentArray(Document(uri=image_path)))

def print_mean_results(resp):
    print(resp.to_dict()["data"][0]["text"])

def print_response_parameters(resp):
    print(f'{resp.to_dict()["parameters"]}')

def print_match_results(da):
    # had trouble putting as on_done parameter for /search, so put inside flow. must address this.
    data = da.to_dict()
    for d in data:
        for m in d["matches"]:
            print(f"query_uri: {d['uri']}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']:.2f}")

def show_montage(query, res):
    # load the query image, display it
    query_cv2 = cv2.imread(query[0].uri)
    cv2.imshow("Query", imutils.resize(query_cv2, width=200))

    # get paths which are same country as query
    imagePaths = sorted(list(paths.list_images("./data/flag_imgs")))
    wantedPaths = [path for path in imagePaths if "france" in path]

    # build montage
    montage = ResultsMontage((240, 320), 5, 20)
    for (i, match) in enumerate(res.to_dict()[0]["matches"]):
        result = cv2.imread(match["uri"]) 
        score = match['scores']['cosine']['value']
        
        montage.addResult(result, text=f"#{i+1} > {score:.2f}",
        highlight=match["uri"] in wantedPaths)

        cv2.imshow("Results", imutils.resize(montage.montage, height=500))
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
