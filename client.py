from docarray.array.mixins import delitem
from jina.clients import Client
from docarray import DocumentArray, Document
from jina import Flow
import os
import torchvision

# First let’s define a client:
client = Client(host='localhost', protocol='http', port=8080)
client.show_progress = True

# Convert to tensor, normalize so they're all similar enough
def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_shape((80, 60))  # ensure all images right size (dataset image size _should_ be (80, 60))
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis for the PyTorch model later

indexing_documents = DocumentArray.from_files("./flag_imgs/*.jpg", size=10000)
indexing_documents.apply(preproc)
model = torchvision.models.resnet50(pretrained=True)  # load ResNet50

indexing_documents.embed(model, device="cpu")

# Then let’s index the set of images we want to search:
indexed_docs = client.post('/index', inputs=indexing_documents)
print(f'Indexed Documents: {len(indexed_docs)}')


# Then let’s search for the closest image to our query image:
query_doc = indexing_documents[-1]
queried_docs = client.post("/search", inputs=[query_doc])
matches = queried_docs[0].matches
print(f'matches: {matches}')
print(f'query: {query_doc.uri}')
print(f'first: {queried_docs[0].uri}')

print("--")

for idx, match in enumerate(matches):
    score = match.scores['cosine'].value
    print(f'> {idx:>2d}({score:.2f}). {match.uri}')
