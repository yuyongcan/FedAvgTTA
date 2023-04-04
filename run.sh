nohup python main.py  --dataset cifar100  --algorithm cotta --server_lr 5e-5 --server_update_mode BN --gpu 1 &

nohup python main.py  --dataset cifar100  --algorithm cotta --server_lr 1e-5 --server_update_mode BN --gpu 2 &

nohup python main.py  --dataset cifar100  --algorithm cotta --server_lr 5e-4 --server_update_mode BN --gpu 3 &

nohup python main.py  --dataset cifar100  --algorithm cotta --server_lr 1e-4 --server_update_mode BN --gpu 4 &
