import subprocess, datetime, os
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

def run_benchmark( r_type : RenderType = RenderType.NORMAL,
                    model_dir = model_dir,
                    model = model,
                    spp = 30,
                    max_depth = 5,
                    image_size = (512, 512)) -> None:
    bvh = BVH.LBVH
    timestamp = datetime.datetime.now()
    filename = f"{benchmark_output_dir}Output-{bvh.name}-{r_type.name} {timestamp.strftime('%Y-%m-%d %H%M%S')}.ppm"

    result = {"BVH" : bvh, "RenderType": r_type}
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

    output_lines = output[output.find("Unit\r\n")+6:-1].splitlines()
    output_lines = output_lines[0:-1]

    construction_timing = extract_timing(output_lines[0])
    rendering_timing = extract_timing(output_lines[1])
    result["construction_ms"] = construction_timing[0]
    result["construction_us"] = construction_timing[1]
    result["rendering_ms"] = rendering_timing[0]
    result["rendering_us"] = rendering_timing[1]
    benchmark_results.append(result)

    bvh = BVH.SAH
    filename = f"{benchmark_output_dir}Output-{bvh.name}-{r_type.name} {timestamp.strftime('%Y-%m-%d %H%M%S')}.ppm"
    result = {"BVH" : bvh, "RenderType": r_type}
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

    output_lines = output[output.find("Unit\r\n")+6:-1].splitlines()
    output_lines = output_lines[0:-1]

    construction_timing = extract_timing(output_lines[0])
    rendering_timing = extract_timing(output_lines[1])
    result["construction_ms"] = construction_timing[0]
    result["construction_us"] = construction_timing[1]
    result["rendering_ms"] = rendering_timing[0]
    result["rendering_us"] = rendering_timing[1]
    benchmark_results.append(result)

@benchmark(10)
def sponza_1():
    run_benchmark(spp = 1, max_depth = 1, model="sponza.obj")

@benchmark(10)
def sponza_2():
    run_benchmark(spp = 200, max_depth = 1, model="sponza.obj")

def main():
    os.makedirs(benchmark_output_dir, exist_ok=True)

    sponza_1()
    sponza_2()

    for result in benchmark_results:
        print(result)

if __name__ == "__main__":
    main()