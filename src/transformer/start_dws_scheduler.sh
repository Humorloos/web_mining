# Script for starting dws scheduler, can be executed from $HOME
tmux new-session -d -s scheduler "$HOME/miniconda3/bin/python $CEPH/web_mining/src/transformer/launch_dws.py"
