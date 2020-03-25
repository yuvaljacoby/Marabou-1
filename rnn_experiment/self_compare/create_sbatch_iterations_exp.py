from datetime import datetime
import os
import sys
import shutil

BASE_FOLDER = "/home/yuval/projects/Marabou/"
if os.path.exists("/cs/usr/yuvalja/projects/Marabou"):
    BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"

OUT_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/out/")
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)

def check_if_model_in_dir(model_name: str, output_folder: str):
    if not output_folder:
        return False
    for f in os.listdir(output_folder):
        if model_name in f or model_name[:model_name.rfind('.')] in f:
            return True
    return False

def create_sbatch(models_folder, output_folder, cache_folder=''):
    print("*" * 100)
    print("creating sbatch {}".format('using cache {}'.format(cache_folder) if cache_folder else ''))
    print("*" * 100)

    if cache_folder:
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=1)
    for model in os.listdir(models_folder):
        exp_time = str(datetime.now()).replace(" ", "-")

        if check_if_model_in_dir(model, cache_folder):
            continue
        with open(os.path.join(output_folder, "run_iterations_exp_" + model + ".sh"), "w") as slurm_file:
            exp = "iterations".format()
            model_name = model[:model.rfind('.')]
            slurm_file.write('#!/bin/bash\n')
            slurm_file.write('#SBATCH --job-name={}_{}_{}\n'.format(model_name, exp, exp_time))
            # slurm_file.write(f'#SBATCH --job-name={model}_{exp}_{exp_time}\n')
            slurm_file.write('#SBATCH --cpus-per-task=2\n')
            # slurm_file.write(f'#SBATCH --output={model}_{job_output_rel_path}\n')
            slurm_file.write('#SBATCH --output={}.out\n'.format(os.path.join(OUT_FOLDER, model_name)))
            # slurm_file.write(f'#SBATCH --partition={partition}\n')
            slurm_file.write('#SBATCH --time=24:00:00\n')
            slurm_file.write('#SBATCH --mem-per-cp  u=300\n')
            slurm_file.write('#SBATCH --mail-type=END,FAIL\n')
            slurm_file.write('#SBATCH --mail-user=yuvalja@cs.huji.ac.il\n')
            slurm_file.write('#SBATCH -w, --nodelist=hm-47\n')
            slurm_file.write('export LD_LIBRARY_PATH=/cs/usr/yuvalja/projects/Marabou\n')
            slurm_file.write('export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou\n')
            slurm_file.write('source /cs/labs/guykatz/yuvalja/tensorflow/bin/activate.csh\n')
            slurm_file.write('python3 rnn_experiment/self_compare/IterationsExperiment.py {} {}\n'.format("exp", model))


if __name__ == '__main__':
    create_sbatch(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else '')
