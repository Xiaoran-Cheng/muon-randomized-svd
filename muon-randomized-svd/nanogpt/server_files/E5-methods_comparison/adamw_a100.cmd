universe   = vanilla
getenv     = true
executable = adamw_a100.sh

log    = /home/xcheng328/nanogpt/$(Cluster).$(Process).log
output = /home/xcheng328/nanogpt/$(Cluster).$(Process).out
error  = /home/xcheng328/nanogpt/$(Cluster).$(Process).err

request_cpus   = 4
request_gpus   = 1
request_memory = 100GB

requirements = (Machine == "isye-hpc0458.isye.gatech.edu")

queue
