# rl-homework
Open AI gym의 Taxi를 DQN으로 구현한 소스코드이다


## 사전 설치 필요한 pacakge 
```
 pip3 pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0hl/cpu--extra-index-url https://download.pytorch.org/w 
 pip3 install tensorboard
 pip3 install gymnasium
 pip3 install numpy 
```

## 파일 설명 
- taxi.py
  - 실제 구현된 코드
  - Agent ,  ReplayBuffer, DQN(network)로 구현 되어 있다. .
  - learn_taxi.py와 play_taxi.py 는 모두 taxi.py를 import 하고 있다.
     
- learn_taxi.py
  -  모델을 학습 시키는 파일
     ```
      python3 learn_taxi.py
     ```
- play_taxi.py
  - 학습된 모델을 로드하여 UI로 실행할 수 있는 파일
  - 파일에서 생성된 모델path를 변경한 후 실행한다. 
    ```
     python3 play_taxi.py
    ```
- logs/tb_taxi
  - tensorboard log파일
  - tensorboard를 실행한 후 http://localhost:6006/ 로 접속하면  학습에 대한 metric 데이터를 볼 수 있다. 
     ```
     tensorboard --logdir=logs/tb_taxi
    ```
- #### models/pytorch_1702476656.pt
  - 학습된 모델파일
  - python3 play_taxi.py 파일에서 model_path를 수정하여 실행할 수 있다. 


 

