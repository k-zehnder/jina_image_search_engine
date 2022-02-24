import os
import torchvision
from jina import Flow
from shutil import rmtree
from docarray import DocumentArray, Document
from my_exeutors import MyExec, MyIndexer


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
    .add(uses=MyExec)
)

query = indexing_documents[0]
with f:
    f.post("/test", inputs=indexing_documents)
    f.post("/index", inputs=indexing_documents)
    f.post("/match", inputs=query)

    # errors if mess up dimensions with brackets
    # f.post("/match", inputs=[indexing_documents])
    # f.post("/close", inputs=[])
