from docarray import DocumentArray, Document


def print_response_parameters(resp):
    print(f' {resp.to_dict()["parameters"]}')

def print_match_results(resp):
    # resp is <jina.types.request.data.DataRequest>

    data = resp.to_dict()["data"]
    for d in data:
        for m in d["matches"]:
            print(f"query_uri: {d['uri']}, match_uri: {m['uri']}, scores: {m['scores']['cosine']['value']}")


# Convert to tensor, normalize so they're all similar enough
def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_shape((80, 60))  # ensure all images right size (dataset image size _should_ be (80, 60))
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis for the PyTorch model later