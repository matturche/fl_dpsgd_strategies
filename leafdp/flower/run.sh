#!/bin/bash

# Set the path to the project folder, replace the brackets by your own path
export BASEPATH='/home/mmoreau/Desktop/git/lab/leaf-fl-dp/'

# Loading script arguments
NBCLIENTS="${1:-2}" # Nb of clients launched by the script (default to 2)
NBMINCLIENTS="${2:-2}" # Nb min of clients before launching round (default to 2)
NBFITCLIENTS="${3:-2}" # Nb of clients sampled for the round (default to 2)
CENTRALIZED="${4:-1}" # Wether we centralize the model or have one for each client (default to True)
DATASET="${5:-femnist}" # Name of leaf dataset used
BATCHSIZE="${6:-16}" # Batch size
VBATCHSIZE="${7:-16}" # Virtual batch size
NBROUNDS="${8:-3}" # Nb of rounds (default to 3)
LR="${9:-0.0001}" # Learning rate
DP="${10:-0}" # Wether Differential Privacy is used or not (default to False)
NM="${11:-1.2}" # Noise multiplier for the Privacy Engine
MGN="${12:-1.0}" # Max grad norm for the Privacy Engine
EPS="${13:-0.0}" # Target epsilon for the privacy budget
STRAT="${14:-vanilla}" # Strategy used to respect target epsilon
SEED="${15:-42}" # Seed for generating clients


python server.py -s $SEED -r $NBROUNDS -nbc $NBCLIENTS -d $DATASET -b $BATCHSIZE -fc $NBFITCLIENTS -ac $NBMINCLIENTS --centralized $CENTRALIZED -dp $DP --tepsilon $EPS --strategy $STRAT &
sleep 30 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect
for ((nb=0; nb<$NBCLIENTS; nb++))
do
    python client.py -c $nb -s $SEED -nbc $NBCLIENTS -d $DATASET -b $BATCHSIZE -vb $VBATCHSIZE -lr $LR -dp $DP -nm $NM -mgn $MGN --centralized $CENTRALIZED --tepsilon $EPS --strategy $STRAT &
    sleep 1
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3`
sleep 86400