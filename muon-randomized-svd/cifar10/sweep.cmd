universe   = vanilla
getenv     = true
executable = sweep.sh

# Usage: first create the sweep on the login node:
#   wandb sweep sweep_e2_rank.yaml
# then paste the sweep ID below and submit:
#   condor_submit sweep.cmd
arguments  = xjc5161-penn-state/uncategorized/cn17sri5 50

log    = /home/xcheng328/cifar10/$(Cluster).$(Process).log
output = /home/xcheng328/cifar10/$(Cluster).$(Process).out
error  = /home/xcheng328/cifar10/$(Cluster).$(Process).err

request_cpus   = 1
request_gpus   = 1
request_memory = 30GB

requirements = (Machine == "isye-hpc0452.isye.gatech.edu") || \
               (Machine == "isye-hpc0456.isye.gatech.edu") || \
               (Machine == "isye-hpc0458.isye.gatech.edu") || \
               (Machine == "isye-hpc0459.isye.gatech.edu") || \
               (Machine == "isye-hpc0460.isye.gatech.edu") || \
               (Machine == "isye-syang605.isye.gatech.edu") || \
               (Machine == "isye-yuding2.isye.gatech.edu")

queue 4
