import os
import shutil
import subprocess

program = "../../main/__init__.py"
path = "./results"
variants = {
    "ga_p50_g100_r25_b64": "-a ga -p 50 -g 100 -r 25 -b 64",
    "ga_p50_g100_r11_b64": "-a ga -p 50 -g 100 -r 11 -b 64",
    "ga_p50_g100_r5_b64": "-a ga -p 50 -g 100 -r 5 -b 64",

    "qiga_p50_g100_r25_b64": "-a qiga -p 50 -g 100 -r 25 -b 64",
    "qiga_p50_g100_r11_b64": "-a qiga -p 50 -g 100 -r 11 -b 64",
    "qiga_p50_g100_r5_b64": "-a qiga -p 50 -g 100 -r 5 -b 64",

    "nsga2_p50_g100_r25_b64": "-a nsga2 -p 50 -g 100 -r 25 -b 64",
    "nsga2_p50_g100_r11_b64": "-a nsga2 -p 50 -g 100 -r 11 -b 64",
    "nsga2_p50_g100_r5_b64": "-a nsga2 -p 50 -g 100 -r 5 -b 64",

    "gnsga2_p50_g100_r25_b64": "-a gnsga2 -p 50 -g 100 -r 25 -b 64",
    "gnsga2_p50_g100_r11_b64": "-a gnsga2 -p 50 -g 100 -r 11 -b 64",
    "gnsga2_p50_g100_r5_b64": "-a gnsga2 -p 50 -g 100 -r 5 -b 64"
}

test = {
    "gnsga2_p5_g5_r5_b64": "-a gnsga2 -p 5 -g 5 -r 5 -b 64",
    "ga_p5_g5_r25_b64": "-a ga -p 5 -g 5 -r 25 -b 64",
    "qiga_p5_g5_r25_b64": "-a qiga -p 5 -g 5 -r 25 -b 64",
    "nsga2_p5_g5_r11_b64": "-a nsga2 -p 5 -g 5 -r 11 -b 64"
}

if not os.path.isdir(path):
    os.mkdir(path)

# TODO test to variants
for name, args in test:
    folder = os.path.join(path, name)
    os.mkdir(folder)
    shutil.copyfile("submit.sh", folder)
    shutil.copyfile("###QSUB.SH", folder)
    os.chdir(folder)
    subprocess.call(['./###QSUB.SH', program, args])
    os.chdir("../../")



