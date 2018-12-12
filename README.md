# "Generic Reduction"
A generic parallel redcution using a single block (implemented in CUDA).

## Compilation

    # run from repo dir
    nvcc -o out/generic-reducation generic-reduction.cu

## TODOs
- [x] push the integer version
- [ ] write a shared memory version
- [ ] write a generic kernel (using C++ templates and functors)
- [ ] write an efficient segmented reduction
