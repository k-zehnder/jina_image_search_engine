from bing_image_downloader.downloader import download

query_string = 'ireland flag'

download(query_string, limit=200,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)