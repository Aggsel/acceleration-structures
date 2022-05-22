import subprocess, datetime, os
import pandas as pd
import numpy as np
from typing import Dict
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
benchmark_results_header = ["BVH",
                            "Model",
                            "Construction Time (ms)",
                            "Construction Time (us)",
                            "Render Time (ms)",
                            "Render Time (us)",
                            "SPP",
                            "Rays Emitted",
                            "Traversed Nodes (total)",
                            "Traversed Nodes (min)",
                            "Traversed Nodes (max)",
                            "Traversed (mean)",
                            "Traversed (std)",
                            "Triangle Count",
                            "Comment"]

def save_benchmark_results(name = "Benchmark_Results.tsv"):
    if(len(benchmark_results) == 0):
        print("Attempting save, but no benchmark results found.")
        return
    df = pd.DataFrame(benchmark_results)
    df.columns = benchmark_results_header
    df.to_csv("Benchmark_Results.tsv", sep="\t")

# Benchmark decorator, allows us to run the same function multiple 
# times, prints progress while doing so.
def benchmark(number_of_times = 1):
    def inner(func):
        def wrapper(*args, **kwargs):
            for i in range(number_of_times):
                print(f"Benchmark {func.__name__}: {i / number_of_times*100}%")
                func(*args, **kwargs)
            print(f"Benchmark {func.__name__}: 100.0%")
        return wrapper
    return inner

def add_benchmark_result(bvh : BVH,
                            model, 
                            construction_time_us, 
                            render_time_us, 
                            spp, 
                            rays_emitted, 
                            traversed_nodes, 
                            traversed_nodes_min, 
                            traversed_nodes_max, 
                            traversed_nodes_median, 
                            traversed_nodes_std, 
                            tri_count,
                            comment = ""):
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
            traversed_nodes_median,
            traversed_nodes_std,
            tri_count,
            comment]

    benchmark_results.append(row)

def run_single_benchmark(   bvh = BVH.LBVH,
                            r_type : RenderType = RenderType.NORMAL,
                            model_dir = model_dir,
                            model = model,
                            spp = 30,
                            max_depth = 5,
                            image_size = (512, 512),
                            filename = "",
                            include_in_benchmark = True,
                            pos = (0,0,0),
                            custom_normalize = -1,
                            comment=""):
    timestamp = datetime.datetime.now()
    if(filename == ""):
        filename = f"{benchmark_output_dir}Output-{bvh.name}-{r_type.name} {timestamp.strftime('%Y-%m-%d %H%M%S')}.ppm"
    output = subprocess.check_output([executable_path, 
                                "-o", filename, 
                                "-i", f"{model_dir}/{model}", 
                                "-bvh", str(bvh.value), 
                                "-r", str(r_type.value),
                                "-spp", str(spp),
                                "--max-depth", str(max_depth),
                                "-iw", str(image_size[0]),
                                "-ih", str(image_size[1]),
                                "-x", str(pos[0]),
                                "-y", str(pos[1]),
                                "-z", str(pos[2]),
                                "--normalize", str(custom_normalize)])

    if(not include_in_benchmark):
        return

    output = output.decode("utf-8")
    output_lines = output.splitlines()

    triangle_count = 0
    construction_time = 0
    rendering_time = 0

    for line in output_lines:
        if(line.find("triangles") != -1):
            triangle_count = int(line[line.find("= ")+1:])
        if(line.find("\t\t") != -1):
            construction_time = line[0:line.find("\t\t")]
            rendering_time = line[line.find("\t\t"):-1]

    # BUG: This will read stale data from an old benchmark if rendertype is normal.
    with open("traversal_steps.txt") as f:
        lines = f.readlines()
    #Strip newlines, parse to int and filter empty rows.
    lines_int = [int(line.rstrip("\n")) for line in lines if line != "\n"]
    arr = np.array(lines_int)
    median = np.mean(arr)
    min = np.min(arr)
    max = np.max(arr)
    std = np.std(arr)
    total_traversed = np.sum(arr)

    add_benchmark_result(   bvh,
                            model,
                            construction_time,
                            rendering_time,
                            spp,
                            0,  #TODO: rays emitted
                            total_traversed,
                            min,
                            max,
                            median,
                            std,
                            triangle_count,
                            comment)
    
    save_benchmark_results()

def run_benchmark( r_type : RenderType = RenderType.NORMAL,
                    model_dir = model_dir,
                    model = model,
                    spp = 30,
                    max_depth = 5,
                    image_size = (512, 512)) -> None:
    run_single_benchmark(BVH.LBVH, r_type, model_dir, model, spp, max_depth, image_size)
    run_single_benchmark(BVH.SAH, r_type, model_dir, model, spp, max_depth, image_size)

@benchmark(10)
def traversal_benchmark(model):
    run_benchmark(r_type = RenderType.HEATMAP, model=model)

def lerp_v3(v1, v2, t):
    x = v2[0] * t + v1[0] * (1.0-t)
    y = v2[1] * t + v1[1] * (1.0-t)
    z = v2[2] * t + v1[2] * (1.0-t)
    return x,y,z

# This is really inefficient as we're rerunning the program each time.
# It's however the easiest way at the moment to implement animation rendering without
# adding support in the actual program.
def animate(model, 
            output_dir, 
            bvh : BVH, 
            r_type, 
            origin, 
            target, 
            frames, 
            spp = 1, 
            max_depth = 1, 
            image_size = (512, 512), 
            output_filename="Animation", 
            include_in_benchmark=True, 
            custom_normalize = -1, 
            comment=""):

    os.makedirs(output_dir, exist_ok=True)

    joined_benchmarks = []
    for frame in range(frames):
        t = frame / float(frames)
        print(f"Rendering frame {frame} ({t*100}%)")
        pos = lerp_v3(origin, target, t)

        filename = f"{output_dir}/animation-{frame:04d}.ppm"
        run_single_benchmark(bvh, r_type, model_dir, model, spp, max_depth, image_size, filename, include_in_benchmark, pos = pos, custom_normalize=custom_normalize, comment=comment)
        if(not include_in_benchmark):
            continue

        with open("traversal_steps.txt") as f:
            lines = f.readlines()
        #Strip newlines, parse to int and filter empty rows.
        lines_int = [int(line.rstrip("\n")) for line in lines if line != "\n"]
        joined_benchmarks.append(lines_int)

        os.makedirs("traversal_steps", exist_ok=True)
        with open(f"traversal_steps/traversal_frame_{frame}_{output_filename}_{r_type}.txt", "w") as file:
            file.writelines(lines)
    
    joined_benchmarks = np.array(joined_benchmarks)
    # TODO: Log the total results as well.
    print(f"\nSum:{np.sum(joined_benchmarks)}\nMin:{np.min(joined_benchmarks)}\nMax:{np.max(joined_benchmarks)}\nAvg:{np.average(joined_benchmarks)}\nStd:{np.std(joined_benchmarks)}\n")

    return_code = subprocess.call([
        "ffmpeg",
        "-framerate", "25",
        "-i", f"{output_dir}/animation-%04d.ppm",
        "-segment_format_options", "movflags=+faststart",
        "-vf", "format=yuv420p",
        "-b:v", "10000k",
        f"{output_filename}.mp4"
        ])
    if(return_code != 0):
        print(f"\n\nERROR: FFMPEG returned with error code {return_code}.\n\n")

def main():
    os.makedirs(benchmark_output_dir, exist_ok=True)
    
    frames = 100

    configurations = {
        "salle_de_bain" : {
            "model" : "mcguire/salle_de_bain.obj",
            "output_dir" : "Render/salle_de_bain/Heatmap",
            "render_type" : RenderType.HEATMAP,
            "origin" : (0, 8, 20),
            "target" : (0, 8, 0),
            "frames" : frames,
            "output_filename" : "salle_de_bain",
            "custom_normalize" : -1, #931
            "comment" : "Animation"
        },
        "dragon" : {
            "model" : "mcguire/dragon.obj",
            "output_dir" : "Render/dragon/Heatmap",
            "render_type" : RenderType.HEATMAP,
            "origin" : (0, 0, 1),
            "target" : (0, 0, 0.4),
            "frames" : frames,
            "output_filename" : "dragon",
            "custom_normalize" : -1, #417
            "comment" : "Animation"
        },
        "sibenik" : {
            "model" : "mcguire/sibenik.obj",
            "output_dir" : "Render/sibenik/Heatmap",
            "render_type" : RenderType.HEATMAP,
            "origin" : (0, -10, 6),
            "target" : (0, -10, -1),
            "frames" : frames,
            "output_filename" : "sibenik",
            "custom_normalize" : -1, #283
            "comment" : "Animation"
        },
        "sponza" : {
            "model" : "mcguire/sponza.obj",
            "output_dir" : "Render/sponza/Heatmap",
            "render_type" : RenderType.HEATMAP,
            "origin" : (0, 1.5, 4),
            "target" : (0, 1.5, 0),
            "frames" : frames,
            "output_filename" : "sponza",
            "custom_normalize" : -1, #495
            "comment" : "Animation"
        },
        "vokselia_spawn" : {
            "model" : "mcguire/vokselia_spawn.obj",
            "output_dir" : "Render/vokselia_spawn/Heatmap",
            "render_type" : RenderType.HEATMAP,
            "origin" : (0.01,0.3,1),
            "target" : (0.01,0.3,-0.5),
            "frames" : frames,
            "output_filename" : "vokselia_spawn",
            "custom_normalize" : -1, #1024
            "comment" : "Animation"
        },
        "san_miguel" : {
            "model" : "mcguire/san_miguel.obj",
            "output_dir" : "Render/san_miguel/Heatmap",
            "render_type" : RenderType.HEATMAP,
            "origin" : (4.5, 0, 6),
            "target" : (4.5, 0, 2),
            "frames" : frames,
            "output_filename" : "san_miguel",
            "custom_normalize" : -1, #3447
            "comment" : "Animation"
        }
    }

    for benchmark_name, config in configurations.items():
        animate(    config["model"],
                    f"{config['output_dir']}/LBVH",
                    BVH.LBVH,
                    r_type = config["render_type"],
                    origin = config["origin"],
                    target = config["target"],
                    frames = config["frames"],
                    output_filename=f"{config['output_filename']}-LBVH",
                    custom_normalize=config["custom_normalize"],
                    comment=config["comment"])
        animate(    config["model"],
                    f"{config['output_dir']}/SAH",
                    BVH.SAH,
                    r_type = config["render_type"],
                    origin = config["origin"],
                    target = config["target"],
                    frames = config["frames"],
                    output_filename=f"{config['output_filename']}-SAH",
                    custom_normalize=config["custom_normalize"],
                    comment=config["comment"])

if __name__ == "__main__":
    main()