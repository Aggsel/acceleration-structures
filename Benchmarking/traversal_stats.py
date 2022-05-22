import numpy as np
from os import walk

traversal_steps_dir = "traversal_steps"

def main():
    for (dirpath, dirnames, filenames) in walk(traversal_steps_dir):
        combined = {}
        for file in filenames:
            bvh = file.split("-")[-1].split("_")[0]
            scene = "-".join(file.split("-")[0].split("_")[3:])
            if(scene not in combined):
                combined[scene] = {}
            if(bvh not in combined[scene]):
                combined[scene][bvh] = []
            with open(f"{traversal_steps_dir}/{file}") as f:
                lines = f.readlines()
            lines_int = [int(line.rstrip("\n")) for line in lines if line != "\n"]
            combined[scene][bvh].extend(lines_int)

        print(f"scene\tbvh\tmean\tmin\tmax\tstd")
        for scene in combined.keys():
            for bvh in combined[scene].keys():
                arr = np.array(combined[scene][bvh])
                print(f"{scene}\t{bvh}\t{np.mean(arr)}\t{np.min(arr)}\t{np.max(arr)}\t{np.std(arr)}")

if __name__ == "__main__":
    main()