# GPU Acceleration Structures

## Build instructions
Built on windows using:
* MSVC (19.29.30040)
* CUDA (11.6.55)

```bash
    nvcc src/main.cu -o build/main
```

## Commandline Arguments

```bash
    -i, --input [FILE]
        Which .obj file to use, e.g.    -i sample_models/large_70.obj
```