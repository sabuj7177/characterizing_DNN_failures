import subprocess
import sys

model_name = sys.argv[1]
noOfThreads = 20
jobQ = []
for t in range(0, noOfThreads):
    # p = subprocess.Popen(["python3", "inference_cifar100_overall.py", model_name, str(t)])
    p = subprocess.Popen(["python3", "cifar100_final_FI.py", model_name, str(t)])
    jobQ.append(p)
allDone = 0
while allDone == 0:
    allDone = 1
    for eachP in jobQ:
        if eachP.poll() is None:
            allDone = 0
            break

print("Done")
