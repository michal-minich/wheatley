PYTHON ?= python3
PYTHONPATH := src
WHEATLY := PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatly

.PHONY: test doctor smoke bench tools stats voice

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

doctor:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLY) doctor

smoke:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLY) once --text "what time is it?"

bench:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLY) bench --repeat 3 --text "Answer in one short sentence: are you online?"

tools:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLY) tools

stats:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLY) stats

voice:
	./scripts/start_wheatly.sh
