IMAGE_NAME=vector-search-api
PORT=8000

docker-build:
	docker build --tag ${IMAGE_NAME} .

docker-run:
	docker run \
	--rm \
	-it \
	-p ${PORT}:5000 \
	-v $(PWD):/app \
	${IMAGE_NAME}

port-usage:
	lsof -i tcp:${PORT}

