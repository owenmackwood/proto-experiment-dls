

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
    parser.add_argument(
        "-s", "--subdir", help="Subdirectory inside main directory to store results.", type=str, default=None
    )
    args = parser.parse_args()

    return args


def run_job(args):
    import time
    from subprocess import run
    from pathlib import Path

    log_dir: Path = Path.home()/"data"/args.run
    if args.subdir is not None:
        log_dir /= args.subdir
    log_dir /= f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}"
    log_dir.mkdir(parents=True)

    args_str = f"--wafer={args.wafer} --fpga={args.fpga} --dir={log_dir}"

    calib_str = f"python calibrate.py  -w {args.wafer} -f {args.fpga} -t {args.run} -n True"
    run_str = f"source ~/venvs/ng/bin/activate && python {args.run}.py  {args_str}"

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

singularity exec --app dls /containers/stable/latest bash -c "{run_str}"
"""

    with open("run.sh", "wt") as f:
        f.write(bs)

    run(["sbatch", "run.sh"], shell=False)


if __name__ == "__main__":
    args = parse_arguments()
    run_job(args)
