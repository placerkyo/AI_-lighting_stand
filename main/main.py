import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
import cv2
import os
from concurrent import futures
from pydub import AudioSegment
from pydub.playback import play
import pygame.mixer
 
#print('start')
def main():
    ###########################################################################################################
    ########顔検出用のネットワーク構築##########
    rng = np.random.RandomState(1234)
    random_state = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_imag = 40
   
   
    class Conv(nn.Module):
        def __init__(self, filter_shape, function=lambda x: x, stride=(1, 1), padding=0):
            super().__init__()
            fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
            fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]
   
            self.W = nn.Parameter(torch.tensor(rng.uniform(
                            -np.sqrt(6/fan_in),
                            np.sqrt(6/fan_in),
                            size=filter_shape
                        ).astype('float32')))
 
            self.b = nn.Parameter(torch.tensor(np.zeros((filter_shape[0]), dtype='float32')))
            self.function = function
            self.stride = stride
            self.padding = padding
           
        def forward(self, x):
            u = F.conv2d(x, self.W, bias=self.b, stride=self.stride, padding=self.padding)
            return self.function(u)
   
   
   
    class Pooling(nn.Module):
        def __init__(self, ksize=(2, 2), stride=(2, 2), padding=0):
            super().__init__()
            self.ksize = ksize
            self.stride = stride
            self.padding = padding
   
        def forward(self, x):
            return F.avg_pool2d(x, kernel_size=self.ksize, stride=self.stride, padding=self.padding)
   
   
    class Flatten(nn.Module):
        def __init__(self):
            super().__init__()
   
        def forward(self, x):
            return x.view(x.size()[0], -1)
   
   
    class Dense(nn.Module):
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            super().__init__()    
            self.W = nn.Parameter(torch.tensor(rng.uniform(
                            -np.sqrt(6/in_dim),
                            np.sqrt(6/in_dim),
                            size=(in_dim, out_dim)
                        ).astype('float32')))
   
            self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype('float32')))
            self.function = function
   
        def forward(self, x):
            return self.function(torch.matmul(x, self.W) + self.b)
   
    class Activation(nn.Module):
        def __init__(self, function=lambda x: x):
            super().__init__()
            self.function = function
   
        def __call__(self, x):
            return self.function(x)
   
    class Dropout(nn.Module):
        def __init__(self, dropout_ratio=0.5):
            super().__init__()
            self.dropout_ratio = dropout_ratio
            self.mask = None
   
        def forward(self, x):
            if self.training:
                self.mask = torch.rand(*x.size()) > self.dropout_ratio
                return x * self.mask.to(x.device)
            else:
                return x * (1.0 - self.dropout_ratio)
   
   
    ########角度検出のためのネットワーク##########
    conv_net_dir = nn.Sequential(
        Conv((32, 1, 8, 8)),     # 画像の大きさ：40x40x1 -> 33x33x32
        Dropout(),
        Pooling((3, 3),(3, 3)),                  # 33x33x32 -> 11x11x32
        Conv((64, 32, 6, 6)),          # 11x11x32 -> 6x6x64
        Conv((128, 64, 3, 3)),           # 6x6x64 -> 4x4x128
        Activation(F.relu),
        Pooling((2, 2)),                 # 4x4x128 -> 2x2x128
        Flatten(),
        Dense(2*2*128, 256, F.relu),
        Dense(256, 3)
    )
 
    ########個人検出のためのネットワーク##########
    conv_net_persons = nn.Sequential(
        Conv((32, 1, 8, 8)),     # 画像の大きさ：40x40x1 -> 33x33x32
        Dropout(),
        Pooling((3, 3),(3, 3)),                  # 33x33x32 -> 11x11x32
        Conv((64, 32, 6, 6)),          # 11x11x32 -> 6x6x64
        Conv((128, 64, 3, 3)),           # 6x6x64 -> 4x4x128
        Activation(F.relu),
        Pooling((2, 2)),                 # 4x4x128 -> 2x2x128
        Flatten(),
        Dense(2*2*128, 256, F.relu),
        Dense(256, 10)
    )
 
   
    ##########データの変換##############
    class test_dataset(torch.utils.data.Dataset):
        def __init__(self, x_test):
            self.x_test = x_test.reshape(-1, 1, size_imag, size_imag).astype('float32') / 255
 
        def __len__(self):
            return self.x_test.shape[0]
 
        def __getitem__(self, idx):
            return torch.tensor(self.x_test[idx], dtype=torch.float)
 
    ###########学習済みモデルの反映##########
    n_epochs = 5
    lr = 0.001
 
    conv_net_dir.to(device)
    conv_net_dir.load_state_dict(torch.load("/home/sozo/main/Identification_dir.pth", map_location=torch.device('cpu')))
    conv_net_dir.eval()
    #dir_dic = {0:'front', 1:'right', 2:'left'}
    dir_dic = {0:0, 1:1, 2:2}
 
    conv_net_persons.to(device)
    conv_net_persons.load_state_dict(torch.load("/home/sozo/main/Identification_persons.pth", map_location=torch.device('cpu')))
    conv_net_persons.eval()
    #name_dic = {0:'kuramoti', 1:'kuwahara', 2:'sasakura', 3:'satou', 4:'tokoro', 5:'maekawa'}
    name_dic = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
    music_dic = {0:'03 Teenager Forever.mp3',1:'05 白日.mp3',2:'06 幕間.mp3',3:'07 飛行艇.mp3',4:'Birth_of_Life.mp3',5:'Refreshing_Morning.mp3'}
 
    #############openCVのモデル##################
    PATH = '/home/sozo/haarcascades/'
    detectorPaths = {
        #"face1": "haarcascade_frontalface_default.xml",    #マスクダメ
        #"face2": "haarcascade_frontalface_alt.xml",        #マスクありでも若干可能
        "face3": "haarcascade_frontalface_alt2.xml",       #マスクありほぼ行ける
        #"face4": "haarcascade_frontalface_alt_tree.xml",    #マスクダメ
        #"eyes": "haarcascade_eye.xml",
        #"lefteye": "haarcascade_lefteye_2splits.xml",
        #"righteye": "haarcascade_eye.xml",
        #"smile": "haarcascade_righteye_2splits.xml",
    }
 
    detectors = {}
    for (name, path) in detectorPaths.items():
        path = PATH + path
        detectors[name] = cv2.CascadeClassifier(path)
 
    ##############画像データの変換###################
    def transform_graytodataloader(img_face):
            img_face = cv2.resize(img_face, dsize=(size_imag,size_imag))
            data = np.array(img_face)           # numpy 形式に変換する
            data = data.reshape([-1, size_imag, size_imag])
            data = test_dataset(data)
            dataloader_test = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
            return dataloader_test
 
    ##############方向検出の関数#####################
    def Indentification_dir(dataloader_test):
            for face in dataloader_test:
                face = face.to(device)
                y_face = conv_net_dir.forward(face)
                pred = y_face.argmax(1).tolist()
            dir_ = dir_dic[pred[0]]
            return dir_
 
    ##############個人検出の関数#####################
    def Indentification_persons(dataloader_test):
            for face in dataloader_test:
                face = face.to(device)
                y_face = conv_net_persons.forward(face)
                pred = y_face.argmax(1).tolist()
            dir_ = name_dic[pred[0]]
            return dir_
 
    ########################################################################################################
    #############その他の機能#########################
    ###############LEDライト#########################
    import gpiod
    def LED_onoff(x):     #x=0でoff、x=1でon
        chip = gpiod.chip(0)
        pin = 26
        gpiod_pin = chip.get_line(pin)
        config = gpiod.line_request()
        config.consumer = "Blink"
        config.request_type = gpiod.line_request.DIRECTION_OUTPUT
        gpiod_pin.request(config)
        gpiod_pin.set_value(x)
        return -1
 
    ################モーター##############################
    from buildhat import Motor
    motor_a = Motor('A')
    def Motor(degrees):  #°表示
        #motor_a = Motor('A')
        motor_a.run_for_degrees(degrees)
        return -1
 
    ###############カメラの設定######################
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # 横幅を設定
    cap.set(4,480) # 縦幅を設定
 
 
 
    #############################################################################################################
    #################mainプログラム#####################
    #########フラグ###########
    LED_onoff(0)
    fl_LED = 0    #LEDがついているかどうか
    fl_person = 0   #人が写っていない期間
    fl_dir = 0    #カメラの向き(0:front,1:right,2:left)
    fl_dir_seq = 0   #同じ方向を連続している期間
    pre_dir_ = 0   #一つ前の推定された方向
    fl_name = -1    #検出した人
    fl_music_onoff = 0
 

    pygame.mixer.init()
    while True:
        #print(fl_name)
        # フレーム毎にキャプチャする
        ret, img = cap.read()     #retは画像情報が入っているかのflag(True/False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #顔検出の負荷軽減のために、キャプチャした画像をモノクロにする
       
        # openCVによる顔検出
        faces = detectors["face3"].detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))   #検出されていたら各画素、numpy、されていなければ()、tupple
       
        #LEDライト
        if len(faces) == 0:
            fl_person += 1
            if fl_LED==1 and fl_person>=20:   #20期間顔が写っていなければLEDを切る
                LED_onoff(0)
                fl_LED = 0
                fl_name = -1
                if fl_music_onoff == 1:
                    pygame.mixer.music.stop()
                    fl_music_onoff = 0
                    fl_name = -1
            elif fl_LED==0:
                fl_person = 0
        else :
            fl_person = 0
            if fl_LED==0:
                LED_onoff(1)
                fl_LED = 1
 
        #人物識別
        for (x,y,w,h) in faces:
            img_face = gray[y:y+h,x:x+w]
            dataloader_test = transform_graytodataloader(img_face)
            dir_ = Indentification_dir(dataloader_test)
            name_ = Indentification_persons(dataloader_test)
           
            #カメラの方向
            if pre_dir_ == dir_: fl_dir_seq += 1
            else : fl_dir_seq = 0
            pre_dir_ = dir_
               
            if fl_dir == 0:
                if (fl_dir_seq >= 2) and (dir_ == 1):
                    Motor(50)
                    fl_dir = 1
                elif (fl_dir_seq >= 2) and (dir_ == 2):
                    Motor(-50)
                    fl_dir = 2
            elif fl_dir == 1:
                if (fl_dir_seq >= 2) and (dir_==0 or dir_==2):
                    Motor(-50)
                    fl_dir = 0
            else:
                if (fl_dir_seq >= 2) and (dir_==0 or dir_==1):
                    Motor(50)
                    fl_dir = 0
                   
            #人によって音楽を変える
            PATH = '/home/sozo/main/'
            fl_name = name_
            if fl_name != -1:
                if fl_music_onoff == 0:
                    now_name = fl_name
                    filename = PATH + music_dic[now_name]
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play(-1)
                    fl_music_onoff = 1
                elif fl_music_onoff == 1 and now_name!=fl_name:
                    now_name = fl_name
                    filename =  PATH + music_dic[now_name]
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play(-1)
                
 
        # imshow関数で結果を表示する
        cv2.imshow('video',img)
 
        # ESCが押されたら終了する
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            LED_onoff(0)
            if fl_dir == 1:
                Motor(50)
            elif fl_dir == 2:
                Motor(-50)
            break
    return -1


if __name__ == '__main__':
    main()
 
 
 
 
cap.release()
cv2.destroyAllWindows()


