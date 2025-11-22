from memory_profiler import memory_usage
import time
# import sys, os, time
# log_path = "/data/xueleyan/MultiBench/log.txt"
# os.makedirs(os.path.dirname(log_path), exist_ok=True)
# sys.stdout = open(log_path, "a", buffering=1)   #

def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


def all_in_one_train(trainprocess, trainmodules):
    starttime = time.time()
    
    # 只调用一次 trainprocess，同时拿到 acc 和 memory
    mem_usage, acc = memory_usage(proc=trainprocess, retval=True, max_usage=True, interval=0.1, timeout=None)
    
    endtime = time.time()

    print("Training Time: " + str(endtime - starttime))
    print("Training Peak Mem: " + str(mem_usage))  # mem_usage 是最大内存使用
    print("Training Params: " + str(getallparams(trainmodules)))
    
    return acc,mem_usage,getallparams(trainmodules)

def all_in_one_test(testprocess, testmodules):
    teststart = time.time()
    acc= testprocess()
    testend = time.time()
    print("Inference Time: "+str(testend-teststart))
    print("Inference Params: "+str(getallparams(testmodules)))
    return acc
