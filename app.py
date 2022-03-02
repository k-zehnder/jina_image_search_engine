from jina import Flow
import imutils

import typer
from rich.console import Console
from rich.table import Table

from image_helpers.utils import print_mean_results, show_montage, my_input, preprocess_img
from executors.my_exeutors import MyMeans, MyIndexer


DATA_DIR = "./data/flag_imgs/left/*.jpg"
DATA_DIR_RIGHT = "./data/flag_imgs/right/*.jpg" 
DATA_DIR_AUGMENTED_RIGHT = "./data/flag_imgs/augmented_right/*.jpg" 


console = Console()
app = typer.Typer()

f = (
    Flow(port_expose=8080, protocol='http')
    .add(uses=MyIndexer)
    .add(uses=MyMeans)
)

@app.command()
def main() -> None:
    # query preprocessing should be in flow, but for clarity in client
    query = preprocess_img("france_6.jpg") 
    console.print("[bold green]Preparing data[/bold green]!", " 🚩")
    
    with f:
        f.post("/index", inputs=my_input(DATA_DIR))
        
        res = f.post("/search", parameters={'limit': 9}, inputs=query)

        f.post("/evaluate")

        f.post("/status", inputs=[])

        f.post("/means", on_done=print_mean_results,  inputs=my_input(DATA_DIR))

    show_montage(query, res)

if __name__ == "__main__":
    app()
    