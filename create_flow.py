from jina import Flow
from docarray import DocumentArray
from docarray import DocumentArray, Document
from jina import Executor, requests

# for docker
#  sudo docker-compose -f docker-compose.yml up -d --build

class MyExec(Executor):
    @requests(
        on=['/test']
    )
    def foo(self, **kwargs):
        print(f'Calling foo')

f = (
    Flow(port_expose=8080, protocol='http')
    .add(
        name='encoder',
        uses='jinahub+docker://CLIPImageEncoder',
        replicas=2
    )
    .add(
        name='indexer',
        uses='jinahub+docker://PQLiteIndexer',
        uses_with={'dim': 512},
        shards=2,
    )
    #.add(uses=MyExec)
)


# f.to_docker_compose_yaml()

# print(dir(f))
# f.save_config("flow.yml")

