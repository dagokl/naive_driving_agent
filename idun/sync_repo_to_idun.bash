ntnu_user='dagbo'

local_repo_root=$(git rev-parse --show-toplevel)

rsync_opts=(-avz --exclude 'wandb/' --exclude 'models/')

# Use jump host if not on NTNU network
if ! nc -w 1 -z idun.hpc.ntnu.no 22 &> /dev/null; then
    echo "Direct connection failed; using jump host via login.stud.ntnu.no"
    rsync_opts+=(-e "ssh -J ${ntnu_user}@login.stud.ntnu.no")
fi

rsync "${rsync_opts[@]}" $local_repo_root "${ntnu_user}@idun.hpc.ntnu.no:/cluster/home/${ntnu_user}"
