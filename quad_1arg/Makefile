
bench: bench.py __pythran__/bench.py util.*.so
	python bench.py

util.*.so:
	pythran util.py

__pythran__/bench.py: bench.py
	transonic bench.py
	touch __pythran__/bench.py