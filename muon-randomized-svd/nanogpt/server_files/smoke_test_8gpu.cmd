universe   = vanilla
getenv     = true
executable = smoke_test_8gpu.sh

log    = /home/xcheng328/nanogpt/$(Cluster).$(Process).log
output = /home/xcheng328/nanogpt/$(Cluster).$(Process).out
error  = /home/xcheng328/nanogpt/$(Cluster).$(Process).err

request_cpus   = 16
request_gpus   = 8
request_memory = 200GB

# TODO: set this to the hostname(s) of the 8x A6000 machine(s) in your cluster.
# Example pattern from smoke_test.cmd: (Machine == "isye-hpc0xxx.isye.gatech.edu")
requirements = (TARGET.CUDACapability >= 8.6) && (TARGET.CUDADeviceName =?= "NVIDIA RTX A6000")

queue
