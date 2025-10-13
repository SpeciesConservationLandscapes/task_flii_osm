IMAGE=flii_osm


build:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --env-file=.env -v ./flii_outputs:/flii_outputs -v ./src:/app/src --rm -it --memory=128g --memory-swap=256g --entrypoint python $(IMAGE) src/task.py --skip_cleanup

shell:
	docker run --env-file=.env -v ./flii_outputs:/flii_outputs -v ./src:/app/src --rm -it --memory=128g --memory-swap=256g --entrypoint bash $(IMAGE)

cleanup:
	isort ./src/*.py
	black ./src/*.py
	flake8 ./src/*.py
	mypy ./src/*.py