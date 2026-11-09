.PHONY: install synth train stream demo app test lint notebook clean

CONFIG ?= configs/synthetic.yaml
MODEL  ?= ae

install:
	pip install -e ".[app,dev]"

synth:
	iotids synth --out data/synthetic

train:
	iotids train --config $(CONFIG)

stream:
	iotids stream --config $(CONFIG) --model $(MODEL)

demo: synth
	iotids all --config configs/synthetic.yaml --model $(MODEL)

real:
	iotids all --config configs/default.yaml --model $(MODEL)

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check src tests app

notebook:
	python scripts/make_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_quickstart.ipynb --ExecutePreprocessor.timeout=1200

clean:
	rm -rf runs/ .pytest_cache **/__pycache__
