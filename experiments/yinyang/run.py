

# import nevergrad as ng

# print(f"{ng.__version__=}")

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run", help="The script to run (without file extension).", type=str, default="yy"
    )
    parser.add_argument(
        "-w", "--wafer", help="The wafer to run on.", type=int, default=69
    )
    parser.add_argument(
        "-f", "--fpga", help="The desired FPGA.", type=int, default=3
    )
    parser.add_argument(
        "-t", "--time", help="Maximum run time.", type=str, default="00:30:00"
    )
    parser.add_argument(
        "-n", "--ntasks", help="Number of processes.", type=int, default=1
    )
    args = parser.parse_args()

    return args


def run_job():
    import time
    from subprocess import run
    from pathlib import Path

    
    log_dir: Path = Path.home()/"data"/args.run/f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}"
    log_dir.mkdir(parents=True)

    args_str = f"--wafer={args.wafer} --fpga={args.fpga} --dir={log_dir}"

    # calib_str = f"singularity exec --app dls /containers/stable/latest python calibrate.py  -w {args.wafer} -f {args.fpga} -t {args.run} -n True"

    bs = f"""#!/bin/bash
#SBATCH --partition=cube
#SBATCH --job-name=mackwood.job
#SBATCH --output={log_dir}/out.txt
#SBATCH --error={log_dir}/err.txt
#SBATCH --time={args.time}
#SBATCH --mem=12000
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=owen.mackwood@hochschule-stralsund.de
#SBATCH --ntasks={args.ntasks}
#SBATCH --wafer={args.wafer}
#SBATCH --fpga-without-aout={args.fpga}

source ~/venvs/ng/bin/activate

singularity exec --app dls /containers/stable/latest python {args.run}.py  {args_str}
"""

    with open("run.sh", "wt") as f:
        f.write(bs)

    print("Running sbatch")
    run(["sbatch", "run.sh"], shell=False)


args = parse_arguments()

if __name__ == "__main__":
    run_job()
