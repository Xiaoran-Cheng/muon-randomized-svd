universe   = vanilla
getenv     = true
executable = ablation_batch_1024_GT.sh

log    = /home/xcheng328/nanogpt/$(Cluster).$(Process).log
output = /home/xcheng328/nanogpt/$(Cluster).$(Process).out
error  = /home/xcheng328/nanogpt/$(Cluster).$(Process).err

request_cpus   = 4
request_gpus   = 1
request_memory = 100GB

# A100 80GB nodes
requirements = (Machine == "isye-hpc0452.isye.gatech.edu") || \
               (Machine == "isye-hpc0456.isye.gatech.edu") || \
               (Machine == "isye-hpc0458.isye.gatech.edu") || \
               (Machine == "isye-hpc0459.isye.gatech.edu") || \
               (Machine == "isye-hpc0460.isye.gatech.edu")

queue
