universe   = vanilla
getenv     = true
executable = smoke_test.sh

log    = /home/xcheng328/nanogpt/$(Cluster).$(Process).log
output = /home/xcheng328/nanogpt/$(Cluster).$(Process).out
error  = /home/xcheng328/nanogpt/$(Cluster).$(Process).err

request_cpus   = 2
request_gpus   = 1
request_memory = 100GB

# Either of the two A100 machines is fine for the smoke test.
requirements = (Machine == "isye-hpc0456.isye.gatech.edu") || \
               (Machine == "isye-hpc0458.isye.gatech.edu")

queue
