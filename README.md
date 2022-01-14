# GAN

![image](https://user-images.githubusercontent.com/45096827/149458161-c0a5582d-f6f2-471d-a380-b4ceef6ac746.png)


# 1.Installation

```
nvidia-docker run --name GAN -it -v /your/project/path/:/project/path/ --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

apt update

apt install -y libgl1-mesa-glx

pip uninstall opencv-python
pip install opencv-python
```
***

# 2.Training with Testing

```
cd /project/path
python train.py
```
***

# 3.Problems

## 3.1 Mode Collapsing
There is a problem that the generator focuses only on reducing loss without following the distribution of data <br>
(생성자가 학습 데이터의 분포를 학습하는 것이 아니라 로스를 줄이는데만 집중하는 문제)

![image](https://user-images.githubusercontent.com/45096827/149459039-15aab7eb-03e9-4b28-a0c3-774f1fbe77ea.png) <br>

## 3.2 Nash Equilibrium
If the balance between the generator and the discriminator is not balanced, learning does not work well.<br>
(생성기와 판별기 사이에 균형이 맞지 않을 경우 학습이 잘 되지 않는 문제)

![Screenshot from 2022-01-14 11-15-55](https://user-images.githubusercontent.com/45096827/149458564-d85887e6-5c83-4841-ab30-df064a8558cb.png) <br>
