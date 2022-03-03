from jina import Flow

# for docker
#  sudo docker-compose -f docker-compose.yml up -d --build


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
    # .add(uses=MyIndexer) # this doesnt work yet
    # .add(uses=MyExec) # this doesnt work yet
)


f.to_docker_compose_yaml()


