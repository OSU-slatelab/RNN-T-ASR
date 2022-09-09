NODES=8
for i in {1..60}
do
    rm -vrf sync/shared
    rm -vrf slurm-*
    sbatch job_submit.sh
    NSL=$(find . -type f -name 'slurm-*' | wc -l)
    while [ $NSL != $NODES ]
    do
        sleep 1
        NSL=$(find . -type f -name 'slurm-*' | wc -l)
    done
    sleep 1h
done
