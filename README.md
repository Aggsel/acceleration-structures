# GPU Acceleration Structures

## Build instructions
Built on windows using:
* MSVC (19.29)
* CUDA (11.6)

Although will likely still work on older versions.

```bash
    nvcc src/main.cu -o build/main
```

## Command line arguments

```bash
    -i , --input <filepath>
        Which .obj file to use.
        Defaults to: sample_models/large_70.obj

    -o, --image-output <filepath>
        What name to save the resulting render as (ppm format).
        Defaults to: output.ppm

    -spp, --samples-per-pixel <int>
        When rendering the image, how many samples per pixels should be used.
        Defaults to: 30

    -iw, --image-width <int>
        Resulting image width.
        Defaults to: 512

    -ih, --image-height <int>
        Resulting image height.
        Defaults to: 512

    --max-depth
        Specify maximum number of per ray bounces allowed during raytracing.
        Defaults to: 5

    -bvh <int>
        Specify which acceleration structure to use.
        Defaults to: 1
            0 - No acceleration structure, brute force ray triangle intersections.
            1 - Karras (2012), LBVH.
```