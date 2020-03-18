
ifeq ($(PYTHON),)
PYTHON := python
endif

all: __pythran__/bench.py tmp_result_julia.txt
	$(PYTHON) bench.py

clean:
	rm -f tmp* __pythran__/bench.py

cleanall: clean
	rm -rf __pythran__

build:
	transonic bench.py -af "-march=native -DUSE_XSIMD -Ofast"

__pythran__/bench.py: bench.py
	transonic bench.py -af "-march=native -DUSE_XSIMD -Ofast"
	@touch __pythran__/bench.py

tmp_result_julia.txt: bench.jl
	julia bench.jl | tee tmp_result_julia.txt
