# ggml-simple

1. Simply put `add_subdirectory(ggml-simple)` to end of `examples/CMakeLists.txt` in llama.cpp project.  
```
    ...
    add_subdirectory(tokenize)
    add_subdirectory(train-text-from-scratch)
    add_subdirectory(ggml-simple) <-- HERE!
endif()
```
2. Build llama.cpp project using `CMake`:  
```
$ cmake -B bld
```
if you have CUDA enabled device, you can build CUDA version:
```
$ cmake -B bld -DGGML_CUDA=on
```
3. `$ make ggml-simple` in `bld` directory.
4. Launch `./bin/ggml-simple`!
```shell
$ ./bin/ggml-simple
main: using Metal backend
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M2
ggml_metal_init: picking default device: Apple M2
ggml_metal_init: using embedded metal library
ggml_metal_init: GPU name:   Apple M2
ggml_metal_init: GPU family: MTLGPUFamilyApple8  (1008)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction support   = true
ggml_metal_init: simdgroup matrix mul. support = true
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 17179.89 MB
main: compute buffer size: 0.0625 KB
mul mat (4 x 3) (transposed result):
[ 60.00 110.00 54.00 29.00
 55.00 90.00 126.00 28.00
 50.00 54.00 42.00 64.00 ]
ggml_metal_free: deallocating
```
