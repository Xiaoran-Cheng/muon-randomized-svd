universe   = vanilla
getenv     = true
executable = sweep.sh

# Usage:
#   wandb sweep sweep_muon_polyak_full.yaml    # creates sweep, prints SWEEP_ID
#   replace <SWEEP_ID> below with the printed ID
#   condor_submit sweep_muon_polyak_full.cmd
arguments  = xjc5161-penn-state/muon-polyak-full-nanogpt/<SWEEP_ID> 50

log    = /home/xcheng328/nanogpt/$(Cluster).$(Process).log
output = /home/xcheng328/nanogpt/$(Cluster).$(Process).out
error  = /home/xcheng328/nanogpt/$(Cluster).$(Process).err

request_cpus   = 8
request_gpus   = 8
request_memory = 200GB

# TODO: update machine list to 8×H100 nodes on your cluster.
requirements = (Machine == "isye-hpc0456.isye.gatech.edu") || \
               (Machine == "isye-hpc0458.isye.gatech.edu")

queue 4
