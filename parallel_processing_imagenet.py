import subprocess
import sys
import logging

model_name = sys.argv[1]
logging.basicConfig(filename='std.log', format='%(name)s - %(levelname)s - %(message)s')
noOfThreads = [40, 41]
jobQ = []
# for t in range(0, noOfThreads):
for t in noOfThreads:
    # p = subprocess.Popen(["python3", "inference_imagenet_overall_2.py", model_name, str(t), str(noOfThreads)])
    p = subprocess.Popen(["python3", "imagenet_final_FI.py", model_name, str(t)])
    jobQ.append(p)
allDone = 0
while allDone == 0:
    allDone = 1
    for eachP in jobQ:
        if eachP.poll() is None:
            allDone = 0
            break

print("Done")
