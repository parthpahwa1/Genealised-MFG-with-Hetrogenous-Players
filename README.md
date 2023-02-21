# MFG
Dominant and multi type agent MFG




## Instructions for Ubuntu

### Requirements

Atleast 

- `python==3.6.1`


```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

- `gym==0.9.2`


```shell
pip install gym
```

- `scikit-learn==0.22.0`


```shell
sudo pip install scikit-learn
```


- `tensorflow 2`


[Check Documentation](https://www.tensorflow.org/install).


- `libboost libraries`


```shell
sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
```
 

### Build the MAgent framework 

```shell
cd ./examples/battle_model
./build.sh
```

Similarly change directory and build for multigather and predatorprey folders for those testbeds. 

### Training and Testing

```
cd ./multibattle/mfrl
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
python3 train_battle.py --algo GenQ_MFG
```
