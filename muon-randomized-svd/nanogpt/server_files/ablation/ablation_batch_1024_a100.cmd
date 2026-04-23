universe   = vanilla
getenv     = true
executable = ablation_batch_1024_a100.sh

log    = /home/xcheng328/nanogpt/$(Cluster).$(Process).log
output = /home/xcheng328/nanogpt/$(Cluster).$(Process).out
error  = /home/xcheng328/nanogpt/$(Cluster).$(Process).err

request_cpus   = 4
request_gpus   = 1
request_memory = 100GB

# TODO: set the hostname(s) of your 1x A100 80GB node(s).
requirements = (Machine == "isye-hpc0458.isye.gatech.edu")

queue
