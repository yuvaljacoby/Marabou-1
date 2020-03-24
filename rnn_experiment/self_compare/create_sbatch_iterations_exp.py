import datetime
import os
import sys

BASE_FOLDER = "/home/yuval/projects/Marabou/"
if os.path.exists("/cs/usr/yuvalja/projects/Marabou"):
    BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"

OUT_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/out/")
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)


def create_sbatch(models_folder, output_folder):
    print("*" * 100)
    print("creating sbatch")
    print("*" * 100)
    os.makedirs(output_folder, exist_ok=1)
    for model in os.listdir(models_folder):
        exp_time = str(datetime.now()).replace(" ", "-")
        with open(os.path.join(output_folder, "run_iterations_exp_" + model + ".sh"), "w") as slurm_file:
            exp = "iterations".format()
            model_name = model[:model.rfind('.')]
            slurm_file.write('#!/bin/bash\n')
            slurm_file.write('#SBATCH --job-name={}_{}_{}\n'.format(model_name, exp, exp_time))
            # slurm_file.write(f'#SBATCH --job-name={model}_{exp}_{exp_time}\n')
            slurm_file.write('#SBATCH --cpus-per-task=2\n')
            # slurm_file.write(f'#SBATCH --output={model}_{job_output_rel_path}\n')
            slurm_file.write('#SBATCH --output={}\n'.format(os.path.join(OUT_FOLDER, model_name)))
            # slurm_file.write(f'#SBATCH --partition={partition}\n')
            slurm_file.write('#SBATCH --time=24:00:00\n')
            slurm_file.write('#SBATCH --mem-per-cpu=300\n')
            slurm_file.write('#SBATCH --mail-type=BEGIN,END,FAIL\n')
            slurm_file.write('#SBATCH --mail-user=yuvalja@cs.huji.ac.il\n')
            slurm_file.write('#SBATCH -w, --nodelist=hm-47\n')
            slurm_file.write('export LD_LIBRARY_PATH=/cs/usr/yuvalja/projects/Marabou\n')
            slurm_file.write('export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou\n')
            slurm_file.write('source /cs/labs/guykatz/yuvalja/tensorflow/bin/activate.csh\n')
            slurm_file.write('python3 rnn_experiment/self_compare/IterationsExperiment.py {} {}\n'.format("exp", model))


if __name__ == '__main__':
    create_sbatch(sys.argv[1], sys.argv[2])
