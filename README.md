# GPU Acceleration Structures

## Requirements
Requires a Nvidia graphics card and CUDA drivers. The program does not use any API features introduced in newer versions of CUDA, so it'll likely work even on somewhat older versions. It has however only been tested on CUDA 11.6.

## Build instructions
Built on windows using:
* MSVC (19.29)
* CUDA (11.6)

Although will likely still work on older versions. Requires an Nvidia GPU.

```bash
    nvcc src/main.cu -Xcompiler "/O2" -o build/main
```

Include debugging symbols by passing the `Zi` flag to MSVC via the NVCC `-Xcomplier` argument (obviously only applicable on windows when compiling using MSVC):

```bash
    nvcc.exe -Xcompiler "/Zi" -o build/main src/main.cu
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

    --max-depth <int>
        Specify maximum number of per ray bounces allowed during raytracing.
        Defaults to: 5

    -bvh <int>
        Specify which acceleration structure to use.
        Defaults to: 1
            0 - No acceleration structure, brute force ray triangle intersections.
            1 - Karras (2012), LBVH.
            2 - Wald (2007), SAH Binning.

    -r, --render <int>
        Whether or not to render the scene.
        Defaults to: 1
            0 - Do not render.
            1 - Render normal (lambertian diffuse).
            2 - Render BVH traversal heatmap.
    
    -x <float>, -y <float>, -z <float>
        Specify world position of the camera by supplying either x, y and/or z coordinates.
        Defaults to: 0.0

    --normalize <int>
        When rendering BVH traversal heatmap, you sometimes want to normalize the 
        output image with regards to some arbitrary max value. This is useful when
        rendering an animation and the entire image sequence pixel values should be 
        normalized according to the maximum steps traversed throughout the animation 
        (as opposed to per frame normalization, which could result in flickering).
            Defaults to: The maximum number of steps traversed during the frame rendered.
```

## Benchmarking

### Dependencies
Requires Python >= 3.6. The algorithms can be benchmarked by running the script ```benchmark.py```. Before running any benchmarks however, install any missing dependencies:
```bash
    python -m pip install -r requirements.txt
```

### Running the benchmarks
The benchmarking script assumes that the binary is located at ```build/main.exe```.
The benchmarks can then be run from the projects root directory:
```bash
    python Benchmarking/benchmark.py
```
The default benchmarking configuration expects 6 .obj files to be present in `sample_models/mcguire` directory. These models can be retreived from [this page.](https://casual-effects.com/data/) They are the following:

* salle_de_bain.obj
* dragon.obj
* sibenik.obj
* sponza.obj
* vokselia_spawn.obj
* san_miguel.obj

The benchmarking script will output traversal data into the folder traversal_steps with the naming scheme `traversal_frame_<frame>_<scene>-<bvh type>_<Render type>.txt` or `traversal_frame_99_san_miguel-LBVH_RenderType.HEATMAP.txt`. The other script in the Benchmarking folder, `traversal_stats.py` will process these files, group them together based on scene and acceleration structure and output various statistics about the number of steps traversed, such as min, max, mean and standard deviation. This script is also meant to be ran from the root directory as such:
```bash
    python Benchmarking/traversal_stats.py
```
Depending on the amount of data to be processed, this can take a while and there's no indication of progress while the script is running.

### Animations
Additionally, the benchmarking file can be used to create camera animations. By default the animation tool will interpolate the camera position between two positions and render out a .ppm image sequence for each frame. The image sequence will then be encoded to a h264 .mp4 file, this step requires [ffmpeg](https://ffmpeg.org/) to be installed and present in your ```PATH``` environment variable.

To reduce flickering while rendering heatmap animations, the animation tool should be used together with the ```--normalize``` flag (see section [command line arguments](#command-line-arguments) for more information).