DATE_TAG := $(shell date +%Y-%m-%d-%H-%M)
VERSION ?= $(DATE_TAG)
IMAGE_NAME = mskkote/open-data-embedder
PORT = 8080

build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

run:
	docker run --rm -p $(PORT):8080 $(IMAGE_NAME)

push:
	docker push $(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	docker push $(IMAGE_NAME):latest

test:
	curl -X POST http://localhost:$(PORT)/embed \
	  -H "Content-Type: application/json" \
	  -d '{"texts": ["пример текста", "вторая строка"]}' | jq

health:
	curl http://localhost:$(PORT)/healthz | jq

local:
	uvicorn src.main:app --host 0.0.0.0 --port $(PORT) --reload
