IMAGE = flii_osm
CONTAINER_NAME = flii_osm_run

build:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --env-file=.env \
		-v ./flii_outputs:/app/flii_outputs \
		-v ./src:/app/src \
		-it --name $(CONTAINER_NAME) \
		--entrypoint python $(IMAGE) src/task.py --skip_cleanup

shell:
	docker run --env-file=.env \
		-v ./flii_outputs:/app/flii_outputs \
		-v ./src:/app/src \
		-it --name $(CONTAINER_NAME) \
		--entrypoint bash $(IMAGE)

attach:
	docker exec -it $(CONTAINER_NAME) bash

stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

cleanup:
	isort ./src/*.py
	black ./src/*.py
	flake8 ./src/*.py
	mypy ./src/*.py
