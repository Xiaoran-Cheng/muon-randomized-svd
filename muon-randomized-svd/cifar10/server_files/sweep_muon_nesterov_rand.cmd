universe   = vanilla
getenv     = true
executable = sweep.sh

# Usage:
#   wandb sweep sweep_muon_nesterov_rand.yaml   # creates sweep, prints SWEEP_ID
#   replace <SWEEP_ID> below with the printed ID
#   condor_submit sweep_muon_nesterov_rand.cmd
# (Old completed sweep: xjc5161-penn-state/uncategorized/nnn9t5sv — kept for reference)
arguments  = xjc5161-penn-state/muon-nesterov-rand/sadl8tfm 100

log    = /home/xcheng328/cifar10/$(Cluster).$(Process).log
output = /home/xcheng328/cifar10/$(Cluster).$(Process).out
error  = /home/xcheng328/cifar10/$(Cluster).$(Process).err

request_cpus   = 1
request_gpus   = 1
request_memory = 30GB

requirements = (Machine == "isye-hpc0456.isye.gatech.edu") || \
               (Machine == "isye-hpc0458.isye.gatech.edu")

notification = Always
notify_user = xjc5161@psu.edu

queue 6
