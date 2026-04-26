PYTHON ?= python3
PYTHONPATH := src

.PHONY: test doctor smoke bench tools stats voice

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

doctor:
	. .venv/bin/activate 2>/dev/null || true; PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatly doctor

smoke:
	. .venv/bin/activate 2>/dev/null || true; PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatly once --text "what time is it?"

bench:
	. .venv/bin/activate 2>/dev/null || true; PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatly bench --repeat 3 --text "Answer in one short sentence: are you online?"

tools:
	. .venv/bin/activate 2>/dev/null || true; PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatly tools

stats:
	. .venv/bin/activate 2>/dev/null || true; PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatly stats

voice:
	./scripts/run_voice_default.sh
