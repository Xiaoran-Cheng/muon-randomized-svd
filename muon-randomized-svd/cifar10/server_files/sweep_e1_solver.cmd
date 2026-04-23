universe   = vanilla
getenv     = true
executable = sweep.sh

# Usage:
#   wandb sweep sweep_e1_solver.yaml   # creates sweep, prints SWEEP_ID
#   replace <SWEEP_ID> below with the printed ID
#   condor_submit sweep_e1_solver.cmd
arguments  = xjc5161-penn-state/e1-solver/3y1q6yy9 1

log    = /home/xcheng328/cifar10/$(Cluster).$(Process).log
output = /home/xcheng328/cifar10/$(Cluster).$(Process).out
error  = /home/xcheng328/cifar10/$(Cluster).$(Process).err

request_cpus   = 1
request_gpus   = 1
request_memory = 30GB

requirements = (Machine == "isye-hpc0456.isye.gatech.edu")

notification = Always
notify_user = xjc5161@psu.edu

queue 4
