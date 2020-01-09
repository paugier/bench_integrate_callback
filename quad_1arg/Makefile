
bench: bench.py __pythran__/bench.py util.*.so
	python bench.py

util.*.so:
	pythran util.py -Ofast -march=native

__pythran__/bench.py: bench.py
	transonic bench.py -af "-march=native -Ofast"
	touch __pythran__/bench.py

clean:
	rm -rf __pythran__
	rm -rf *.so