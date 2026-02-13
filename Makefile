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

.PHONY: build run push test test-sparse test-hybrid health dev build-all push-all list-models info \
       build-gpu run-gpu compose-gpu compose-gpu-down build-all-gpu push-gpu push-all-gpu

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
	@echo "Pushing all tags for $(IMAGE_NAME)"
	@docker images --format "{{.Repository}}:{{.Tag}}" | grep "^$(IMAGE_NAME):" | while read image; do \
		echo "Pushing $$image"; \
		docker push $$image || true; \
	done

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

test-sparse:
	curl -X POST http://localhost:$(PORT)/embed \
	  -H "Content-Type: application/json" \
	  -d '{"texts": ["text example", "second string"], "mode": "sparse"}' | jq

test-hybrid:
	curl -X POST http://localhost:$(PORT)/embed \
	  -H "Content-Type: application/json" \
	  -d '{"texts": ["text example", "second string"], "mode": "hybrid"}' | jq

health:
	curl http://localhost:$(PORT)/healthz | jq

dev:
	MODEL_NAME="$(MODEL)" uvicorn src.main:app --host 0.0.0.0 --port $(PORT) --reload

# ── GPU targets ──────────────────────────────────────────────────────

build-gpu:
	@echo "Building GPU $(IMAGE_NAME)-gpu:$(VERSION) with model $(MODEL)"
	docker build \
		--build-arg MODEL_NAME="$(MODEL)" \
		-f Dockerfile.gpu \
		-t $(IMAGE_NAME)-gpu:$(VERSION) \
		-t $(IMAGE_NAME)-gpu:latest \
		.

run-gpu:
	docker run --rm --gpus all -p $(PORT):8080 $(IMAGE_NAME)-gpu:latest

compose-gpu:
	MODEL_NAME="$(MODEL)" docker compose -f docker-compose.gpu.yml up --build

compose-gpu-down:
	docker compose -f docker-compose.gpu.yml down

build-all-gpu:
	@for model in $(MODELS); do \
		echo "Building GPU $$model..."; \
		$(MAKE) build-gpu MODEL=$$model || exit 1; \
	done
	@echo "All GPU models built successfully!"

push-gpu:
	@echo "Pushing all tags for $(IMAGE_NAME)-gpu"
	@docker images --format "{{.Repository}}:{{.Tag}}" | grep "^$(IMAGE_NAME)-gpu:" | while read image; do \
		echo "Pushing $$image"; \
		docker push $$image || true; \
	done

push-all-gpu:
	@for model in $(MODELS); do \
		echo "Pushing GPU $$model..."; \
		$(MAKE) push-gpu MODEL=$$model || exit 1; \
	done
	@echo "All GPU models pushed successfully!"
