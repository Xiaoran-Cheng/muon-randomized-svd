universe   = vanilla
getenv     = true
executable = final_muon_polyak_rand.sh

log    = /home/xcheng328/cifar10/$(Cluster).$(Process).log
output = /home/xcheng328/cifar10/$(Cluster).$(Process).out
error  = /home/xcheng328/cifar10/$(Cluster).$(Process).err

request_cpus   = 1
request_gpus   = 1
request_memory = 30GB

requirements = (Machine == "isye-hpc0452.isye.gatech.edu") || \
               (Machine == "isye-hpc0459.isye.gatech.edu") || \
               (Machine == "isye-hpc0460.isye.gatech.edu") || \
               (Machine == "isye-syang605.isye.gatech.edu") || \
               (Machine == "isye-yuding2.isye.gatech.edu")

notification = Always
notify_user = xjc5161@psu.edu

queue
