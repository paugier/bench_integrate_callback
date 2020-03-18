```raw
julia bench.jl | tee tmp_result_julia.txt
Julia without simd 0.163 µs
Julia with simd    0.123 µs
ratio (without simd)/(with simd): 1.33
python bench.py
Pythran: 0.29 µs
Julia:   0.12 µs
ratio Pythran/Julia: 2.39
```