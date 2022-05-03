import subprocess, datetime, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from enum import Enum

class BVH(Enum):
    NONE = 0
    LBVH = 1
    SAH = 2

class RenderType(Enum):
    NONE = 0
    NORMAL = 1
    HEATMAP = 2

executable_path = "build/main.exe"
benchmark_output_dir = "Benchmark Results/Renders/"
model_dir = "sample_models"
model = "sponza.obj"
benchmark_results = []
benchmark_results_header = ["BVH", "Model", "Construction Time (ms)", "Construction Time (us)", "Render Time (ms)", "Render Time (us)", "SPP", "Rays Emitted", "Traversed Nodes (total)", "Traversed Nodes (min)", "Traversed Nodes (max)", "Triangle Count"]

def benchmark(number_of_times = 1):
    def inner(func):
        def wrapper(*args, **kwargs):
            for i in range(number_of_times):
                print(f"Benchmark {func.__name__}: {i / number_of_times*100}%")
                func(*args, **kwargs)
            print(f"Benchmark {func.__name__}: 100.0%")
        return wrapper
    return inner

def extract_timing(input_line : str) -> Tuple[int, int]:
    result_ms = input_line.split("\t")[1]
    result_us = input_line.split("\t")[-2]
    return result_ms, result_us

def add_benchmark_result(bvh : BVH, model, construction_time_us, render_time_us, spp, rays_emitted, traversed_nodes, traversed_nodes_min, traversed_nodes_max, tri_count):
    row = [ bvh.name,
            model,
            int(construction_time_us)/1000.0,
            construction_time_us,
            int(render_time_us)/1000.0,
            render_time_us,
            spp,
            rays_emitted,
            traversed_nodes,
            traversed_nodes_min,
            traversed_nodes_max,
            tri_count]

    benchmark_results.append(row)

def run_single_benchmark(   bvh = BVH.LBVH,
                            r_type : RenderType = RenderType.NORMAL,
                            model_dir = model_dir,
                            model = model,
                            spp = 30,
                            max_depth = 5,
                            image_size = (512, 512)):
    timestamp = datetime.datetime.now()
    filename = f"{benchmark_output_dir}Output-{bvh.name}-{r_type.name} {timestamp.strftime('%Y-%m-%d %H%M%S')}.ppm"
    output = subprocess.check_output([executable_path, 
                                "-o", filename, 
                                "-i", f"{model_dir}/{model}", 
                                "-bvh", str(bvh.value), 
                                "-r", str(r_type.value),
                                "-spp", str(spp),
                                "--max-depth", str(max_depth),
                                "-iw", str(image_size[0]),
                                "-ih", str(image_size[1])])

    output = output.decode("utf-8")
    output_lines = output.splitlines()

    triangle_count = 0
    construction_time = 0
    rendering_time = 0

    for line in output_lines:
        if(line.find("triangles") != -1):
            triangle_count = int(line[line.find("= ")+2:-1])
        if(line.find("\t\t") != -1):
            construction_time = line[0:line.find("\t\t")]
            rendering_time = line[line.find("\t\t"):-1]

    with open("output.txt") as f:
        lines = f.readlines()
    #Strip newlines, parse to int and filter empty rows.
    lines_int = [int(line.rstrip("\n")) for line in lines if line != "\n"]
    arr = np.array(lines_int)
    median = np.median(arr)
    min = np.min(arr)
    max = np.max(arr)
    std = np.std(arr)
    total_traversed = np.sum(arr)

    add_benchmark_result(   bvh,
                            model,
                            construction_time,
                            rendering_time,
                            spp,
                            0,  #rays emitted
                            total_traversed,
                            min,
                            max,
                            triangle_count)

def run_benchmark( r_type : RenderType = RenderType.NORMAL,
                    model_dir = model_dir,
                    model = model,
                    spp = 30,
                    max_depth = 5,
                    image_size = (512, 512)) -> None:
    bvh = BVH.LBVH
    run_single_benchmark(bvh, r_type, model_dir, model, spp, max_depth, image_size)
    bvh = BVH.SAH
    run_single_benchmark(bvh, r_type, model_dir, model, spp, max_depth, image_size)

@benchmark(10)
def sponza_1():
    run_benchmark(r_type = RenderType.HEATMAP, model="sponza.obj")

@benchmark(10)
def sponza_2():
    run_benchmark(r_type = RenderType.HEATMAP, model="sponza.obj")

def main():
    os.makedirs(benchmark_output_dir, exist_ok=True)

    sponza_1()

    df = pd.DataFrame(benchmark_results)
    df.columns = benchmark_results_header
    df.to_csv("Benchmark_Results.tsv", sep="\t")

if __name__ == "__main__":
    main()