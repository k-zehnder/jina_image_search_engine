import os
import pandas as pd
import torchvision
from jina import Flow, Document, DocumentArray


os.environ['JINA_LOG_LEVEL'] = 'DEBUG'

fname = "./data/pqdata/my_details.csv"
df = pd.read_csv(fname, warn_bad_lines=True, error_bad_lines=False)
df = df.dropna()

def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_shape((80, 60))  # ensure all images right size (dataset image size _should_ be (80, 60))
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis for the PyTorch model later

def get_product_docs():
    da = DocumentArray()
    for index, row in df.iterrows(): 
        doc_uri = f'{row["Name"]}'        
        doc = Document(uri=doc_uri, tags=dict(row))
        da.append(doc)
    
    da.apply(preproc)
    model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
    da.embed(model, device='cpu', to_numpy=True)
    return da

# main
docs = get_product_docs()
print(len(docs))

f = Flow().add(
            uses="jinahub://DocCache/v0.1", 
            name="deduplicator",
            install_requirements=True
        ).add(
            uses='jinahub://PQLiteIndexer/latest',
            name="indexer",
            uses_with={
                'dim': 1000,
                'metric': 'cosine',
                'columns': [
                    ('Name', 'str'),
                    ('Country', 'str'),
                    ('Width', 'str'),
                    ('Type', 'str')
                ],
            },
            uses_metas={"workspace": "./workspace"},
            install_requirements=True
        )

with f:
    f.index(inputs=docs, show_progress=True)

    resp = f.search(inputs=docs[0], 
                    return_results=True, 
                    parameters={
                        'filter': {
                            'Country': {'$eq': "canada"},
                            'Width' : {'$gte' : 150}
                        },
                        'limit': 15
                    })
    for d in resp:
        for m in d.matches:
            score = m.scores['cosine'].value
            print(f"query_uri: {d.uri}, match_uri: {m.uri}, scores: {m.scores['cosine'].value:.2f}")