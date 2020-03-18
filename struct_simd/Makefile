
ifeq ($(PYTHON),)
PYTHON := python
endif

all: bench.py __pythran__/bench.py tmp_result_julia.txt
	$(PYTHON) bench.py

clean:
	rm -f tmp* __pythran__/bench.py

cleanall: clean
	rm -rf __pythran__

build:
	transonic bench.py -af "-march=native -DUSE_XSIMD -Ofast"

__pythran__/bench.py: bench.py
	transonic bench.py -af "-march=native -DUSE_XSIMD -Ofast"

tmp_result_julia.txt:
	julia bench.jl > tmp_result_julia.txt
