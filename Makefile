PYTHON ?= python3
PYTHONPATH := src
WHEATLEY := PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m wheatley

.PHONY: test doctor smoke bench tools stats voice stt-server

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

doctor:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLEY) doctor

smoke:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLEY) once --text "what time is it?"

bench:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLEY) bench --repeat 3 --text "Answer in one short sentence: are you online?"

tools:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLEY) tools

stats:
	. .venv/bin/activate 2>/dev/null || true; $(WHEATLEY) stats

voice:
	./scripts/start_wheatley.sh

stt-server:
	./scripts/start_janka_stt_server.sh
