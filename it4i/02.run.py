import os
import shutil
import subprocess
import time

program = "../../../main/__init__.py"
path = "./results"

variants = {
    "ga_p50_g100_r25_b64": "-a ga -p 50 -g 200 -r 25 -b 64",
    "ga_p50_g100_r11_b64": "-a ga -p 50 -g 200 -r 11 -b 64",
    "ga_p50_g100_r5_b64": "-a ga -p 50 -g 200 -r 5 -b 64",

    "qiga_p50_g100_r25_b64": "-a qiga -p 50 -g 200 -r 25 -b 64",
    "qiga_p50_g100_r11_b64": "-a qiga -p 50 -g 200 -r 11 -b 64",
    "qiga_p50_g100_r5_b64": "-a qiga -p 50 -g 200 -r 5 -b 64",

    "nsga2_p50_g100_r25_b64": "-a nsga2 -p 50 -g 200 -r 25 -b 64",
    "nsga2_p50_g100_r11_b64": "-a nsga2 -p 50 -g 200 -r 11 -b 64",
    "nsga2_p50_g100_r5_b64": "-a nsga2 -p 50 -g 200 -r 5 -b 64",

    "gnsga2_p50_g100_r25_b64": "-a gnsga2 -p 50 -g 200 -r 25 -b 64",
    "gnsga2_p50_g100_r11_b64": "-a gnsga2 -p 50 -g 200 -r 11 -b 64",
    "gnsga2_p50_g100_r5_b64": "-a gnsga2 -p 50 -g 200 -r 5 -b 64"
}

test = {
    "gnsga2_p5_g5_r5_b64": "-a gnsga2 -p 5 -g 5 -r 5 -b 64",
    "ga_p5_g5_r25_b64": "-a ga -p 5 -g 5 -r 25 -b 64",
    "qiga_p5_g5_r25_b64": "-a qiga -p 5 -g 5 -r 25 -b 64",
    "nsga2_p5_g5_r11_b64": "-a nsga2 -p 5 -g 5 -r 11 -b 64"
}


def make_cmd(folder, p, a):
    text = "#!/bin/bash\n\n" + "ml Python/3.9.5-GCCcore-10.3.0\n" + "python " + p + " " + a + "\n"
    file = "run.sh"
    with open(file, 'w') as f:
        f.write(text)
    subprocess.call(['chmod', '+x', file])


if not os.path.isdir(path):
    os.mkdir(path)

for name, args in variants.items():
    result = subprocess.run(['sh', './my_jobs_list.sh'], stdout=subprocess.PIPE)
    while len(result.stdout) > 5:
        time.sleep(10)
        result = subprocess.run(['sh', './my_jobs_list.sh'], stdout=subprocess.PIPE)

    folder = os.path.join(path, name)
    os.mkdir(folder)
    shutil.copyfile("submit.sh", os.path.join(folder, "submit.sh"))
    shutil.copyfile("###QSUB.SH", os.path.join(folder, "###QSUB.SH"))
    os.chdir(folder)
    make_cmd(folder, program, args)
    subprocess.call(['chmod', '+x', 'submit.sh'])
    subprocess.call(['chmod', '+x', '###QSUB.SH'])
#    subprocess.call(['./###QSUB.SH'])
    os.chdir("../../")

