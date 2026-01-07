DATE_TAG := $(shell date +%Y-%m-%d-%H-%M)
VERSION ?= $(DATE_TAG)

# Supported models
MODELS := \
    jinaai/jina-embeddings-v3 \
    intfloat/multilingual-e5-base \
    BAAI/bge-m3 \
    sentence-transformers/LaBSE

# Default model
MODEL ?= BAAI/bge-m3

# Convert model name to tag-friendly format (lowercase, replace / and special chars with -)
MODEL_TAG := $(shell echo "$(MODEL)" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

# Image naming
IMAGE_BASE = mskkote/embedder
IMAGE_NAME = $(IMAGE_BASE)-$(MODEL_TAG)
PORT = 8080

.PHONY: build run push test health dev build-all push-all list-models info

build:
	@echo "Building $(IMAGE_NAME):$(VERSION) with model $(MODEL)"
	docker build \
		--build-arg MODEL_NAME="$(MODEL)" \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		.

run:
	docker run --rm -p $(PORT):8080 $(IMAGE_NAME):latest

push:
	docker push $(IMAGE_NAME):$(VERSION)
	docker push $(IMAGE_NAME):latest

# Build all models
build-all:
	@for model in $(MODELS); do \
		echo "Building $$model..."; \
		$(MAKE) build MODEL=$$model || exit 1; \
	done
	@echo "All models built successfully!"

# Push all models
push-all:
	@for model in $(MODELS); do \
		echo "Pushing $$model..."; \
		$(MAKE) push MODEL=$$model || exit 1; \
	done
	@echo "All models pushed successfully!"

# List all supported models
list-models:
	@echo "Supported models:"
	@for model in $(MODELS); do \
		echo "  - $$model"; \
	done

# Show build information
info:
	@echo "Model: $(MODEL)"
	@echo "Model Tag: $(MODEL_TAG)"
	@echo "Image Name: $(IMAGE_NAME):$(VERSION)"
	@echo "Image Name (latest): $(IMAGE_NAME):latest"
	@echo "Date Tag: $(DATE_TAG)"

test:
	curl -X POST http://localhost:$(PORT)/embed \
	  -H "Content-Type: application/json" \
	  -d '{"texts": ["text example", "second string"]}' | jq

health:
	curl http://localhost:$(PORT)/healthz | jq

dev:
	MODEL_NAME="$(MODEL)" uvicorn src.main:app --host 0.0.0.0 --port $(PORT) --reload
