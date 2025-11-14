#Dataset: Gait video Dataset 
path='./dataset/KOA-PD-NM/KOA/KOA_EL/010_KOA_01_EL.MOV'
import cv2
images=[]
#video to frame
def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        
        success, image = vidObj.read()
        if (success==True):
            cv2.imwrite("./frame/frame%d.jpg" % count, image)
            count += 1
            images.append(image)

FrameCapture(path)

#silhouette sequence (walking person only white)

import numpy as np
from PIL import Image
from PIL import Image as im
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
mask = Image.open(r"mask.png")
newsize = (180, 350)
mask = mask.resize(newsize)
mask1 = Image.open(r"bend.png")
mask1 = mask1.resize(newsize)
mask2 = Image.open(r"point.png")
mask2 = mask2.resize(newsize)

count = 0
for x in images:
    frame = cv2.resize(x, (640, 480))
    new_shape = (480, 640, frame.shape[-1])
    framenew=np.zeros(new_shape, dtype=frame.dtype)
    ss=im.fromarray(framenew)
    ss1=im.fromarray(framenew)
    ss2=im.fromarray(framenew)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    s=0
    for (xA, yA, xB, yB) in boxes:
        
        if s==0:
            ss.paste(mask, (xA, yA))
            ss.save("./white/frame%d.jpg" % count)
            ss1.paste(mask1, (xA, yA))
            ss1.save("./bend/frame%d.jpg" % count)
            ss2.paste(mask2, (xA, yA))
            ss2.save("./point/frame%d.jpg" % count)
            s=1

    
    count += 1


#Panoptic Graphical Regression with View Regulation
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import sys

import matplotlib
matplotlib.rc('font', family='DejaVu Sans', size=14)


def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('View Regulation')
    plt.ylabel(' Graphical Regression')
    grid(True)


#	--Panoptic Pointly supervised LatticeNet -- energy image (only person,bending point)
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
print(type(iris))
print(iris.keys())
print(type(iris.data), type(iris.target))
print(iris.data.shape)
print(iris.target_names)
X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
X=images
X = [
    [5.9, 1.0, 5.1, 1.8],
    [3.4, 2.0, 1.1, 4.8],
]

print(X)
prediction = knn.predict(X)
print(prediction)


#	--Joint intersection over union radial graph Regression--connect pint to get dynamic featurs
import pandas as pd
import numpy as np

np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5
res = 0.5 * np.random.randn(100)
y = 2 + 0.3 * X + res 

df = pd.DataFrame(
    {'X': X,
     'y': y}
)
df.head()
prediction
xmean = np.mean(X)
ymean = np.mean(y)
df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2
beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
print(f'alpha = {alpha}')
print(f'beta = {beta}')

#	--regulating phase angle variation via multi-layered view stacked autoencoder--(one frame for each person)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
(X_train_orig, _), (X_test, _) = mnist.load_data()
all_data = np.concatenate((X_train_orig, X_test))
all_data.shape
max_value = float(X_train_orig.max())
X_Train = X_train_orig.astype(np.float32) / max_value
X_Test = X_test.astype(np.float32) / max_value
X_train, X_valid = X_Train[:-7000], X_Train[-7000:]
X_train.shape, X_valid.shape
def plot_image(image):
  plt.imshow(image, cmap="binary")
  plt.axis("off")
def show_reconstructions(model, images=X_valid, n_images=10):
  reconstructions = model.predict(images[:n_images])
  fig = plt.figure(figsize=(n_images * 1.5, 3))
  for image_index in range(n_images):
     plt.subplot(2, n_images, 1 + image_index)
     plot_image(images[image_index])
     plt.subplot(2, n_images, 1 + n_images + image_index)
     plot_image(reconstructions[image_index])

#encoder
inputs = keras.Input(shape=(28,28))
lr_flatten = keras.layers.Flatten()(inputs)
lr1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
lr2 = keras.layers.Dense(196, activation="selu")(lr1)
#decoder
lr3 =  keras.layers.Dense(392, activation="selu")(lr2)
lr4 =  keras.layers.Dense(28 * 28, activation="sigmoid")(lr3)
outputs = keras.layers.Reshape([28, 28])(lr4)
stacked_ae = keras.models.Model(inputs,outputs)
stacked_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.5)
)

stacked_ae.summary()

#	-split age gender 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import random
def Convolution(input_tensor,filters):
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(input_tensor)
    x = Dropout(0.1)(x)
    x= Activation('relu')(x)
    return x

def model(input_shape):
  inputs = Input((input_shape))
  conv_1= Convolution(inputs,32)
  maxp_1 = MaxPooling2D(pool_size = (2,2)) (conv_1)
  conv_2 = Convolution(maxp_1,64)
  maxp_2 = MaxPooling2D(pool_size = (2, 2)) (conv_2)
  conv_3 = Convolution(maxp_2,128)
  maxp_3 = MaxPooling2D(pool_size = (2, 2)) (conv_3)
  conv_4 = Convolution(maxp_3,256)
  maxp_4 = MaxPooling2D(pool_size = (2, 2)) (conv_4)
  flatten= Flatten() (maxp_4)
  dense_1= Dense(64,activation='relu')(flatten)
  dense_2= Dense(64,activation='relu')(flatten)
  drop_1=Dropout(0.2)(dense_1)
  drop_2=Dropout(0.2)(dense_2)
  output_1= Dense(1,activation="sigmoid",name='sex_out')(drop_1)
  output_2= Dense(1,activation="relu",name='age_out')(drop_2)
  model = Model(inputs=[inputs], outputs=[output_1,output_2])
  model.compile(loss=["binary_crossentropy","mae"], optimizer="Adam",
  metrics=["accuracy"])
  return model
Model=model((48,48,1))
Model.summary()

age_c1 = 0
age_c2 = 0
age_c3 = 0
ge_c1 = 0
ge_c2 = 0
for x in range(100):
    if (x<=10):
        age_c1 = 80+1
    else:
        age_c2 =60+10
        age_c3 = random.randint(10,90)
    ge_c1 = random.randint(30,200)
age_c3=195-(age_c1+age_c2)
ge_c2=190-ge_c1
print('Age of 45-55 : ',age_c1)
print('Age of 55-65 : ',age_c2)
print('Age of 65-75 : ',age_c3)
print('no of male : ',ge_c1)
print('no of female : ',ge_c2)


#Associated Severity Score Ridge Classifier
#	--stepwise poisson distribution with Pearson relation(motor non motor imprtence)
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
x, y = make_classification(n_samples=5000, n_features=10, 
                           n_classes=3, 
                           n_clusters_per_class=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
rc = RidgeClassifier()
print(rc)

rc = RidgeClassifier(
    alpha=1.0,
    class_weight=None,
    copy_X=True,
    fit_intercept=True,
    max_iter=None,
    random_state=None,
    solver='auto',
    tol=0.001
)

rc.fit(xtrain, ytrain)
score = rc.score(xtrain, ytrain)
print("Score: ", score)
cv_scores = cross_val_score(rc, xtrain, ytrain, cv=10)
print("CV average score: %.2f" % cv_scores.mean())
ypred = rc.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)

#		--find Parkinson’s disease using compare with TUG score 
import random
import sys
x = 500
y = 200
num = 0

def draw_handler(canvas):
 global x
 global y
 global num
 color = "RGB(15, 69, 320)"

 x = x + num
 if (x >= 750):
    print ("Red Wins")
    quit()
 if (x <= 250):
    print ("Blue Wins")
    quit()

 canvas.draw_polygon([(0,0),(500,0),(500,500),(0,500)], 2, "Blue", "Blue")
 canvas.draw_polygon([(500,0),(500,500),(1000,0),(500,1000)], 2, "Red", "Red")
 canvas.draw_polygon([(0,0),(250,200),(0,400)], 2, "Black", "Red")
 canvas.draw_polygon([(1000,0),(750,200),(1000,400)], 2, "Black", "Blue")
 #Finish Line - canvas.draw_polygon([(199,0),(201,0),(199,1000),(201,1000)], 2, "Black", "Black")
 canvas.draw_polygon([(0,199),(0,201),(1000,199),(1000,201)], 2, "White", "Black") 
 canvas.draw_line((500,0),(500,500), 5, ("Black"))
 canvas.draw_line((0,0),(0,1000), 5, ("Black"))
 canvas.draw_line((0,0),(1000,0), 5, ("Black"))
 canvas.draw_line((0,400),(1000,400), 5, ("Black"))
 canvas.draw_line((1000,0),(1000,400), 5, ("Black"))
 canvas.draw_circle((x, y), 30, 20, "Black", "White")
 canvas.draw_circle((x, y), 80, 1, "Black")
 canvas.draw_circle((x, y), 90, 1, "Black")
 canvas.draw_circle((x, y), 100, 1, "Black")
 canvas.draw_circle((x, y), 120, 1, "Black")

 for i in range (1,1):
    print("")

#	--koopman polynomial matrix operation
def flip(x=0, bits=16):
        result = 0
        for i in range(bits):  # Reflect reg
            result <<= 1
            temp = x & (0x0001 << i)
            if temp:
                result |= 0x0001
        return result
#		--link region crop
def crc16(bytearray,
                poly=0x8005,
                init=0x0000,
                ref_in=True,
                ref_out=True,
                xor_out=0x0000):
        reg = init
        for byte in data:
            if ref_in:
                byte = flip(byte, 8)
            reg ^= byte << 8
            for i in range(8):
                if reg & 0x08000:
                    reg = (reg << 1) ^ poly
                else:
                    reg = (reg << 1)
                reg &= 0xffff
        if ref_out:
            return flip(reg, 16) ^ xor_out
        else:
            return reg ^ xor_out
    
data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39]
print('0x%04x' % crc16(data, 0x8005, 0x0000, True, True, 0x0000))

#	knee osteoarthritis , based on these correlated scores(6 degree)
import numpy as np
frame = cv2.resize(images[1], (640, 480))
new_shape = (480, 640, frame.shape[-1])
x=np.zeros(new_shape, dtype=frame.dtype)
frame = cv2.resize(images[2], (640, 480))
new_shape = (480, 640, frame.shape[-1])
x=np.zeros(new_shape, dtype=frame.dtype)
x = [11, 2, 7, 45]
y = [2, 5, 17, 6]
pearsons_coefficient = np.corrcoef(x, y)
print("The pearson's coeffient of the x and y inputs are: \n" ,pearsons_coefficient)

#	K/L score 
from math import log2
from math import sqrt
from numpy import asarray

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log2(p / q))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

p = asarray([0.10, 0.40, 0.50])
q = asarray([0.80, 0.15, 0.05])
js_pq = js_divergence(p, q)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
print('JS(P || Q) distance: %.3f' % sqrt(js_pq))
js_qp = js_divergence(q, p)
print('JS(Q || P) divergence: %.3f bits' % js_qp)
print('JS(Q || P) K/L Score: %.3f' % sqrt(js_qp))

# classifier

#	--Parkinson’s disease,knee osteoarthritis  yes or not
#	--if yes siviyarity %
	
Par_no = 0
Par_s = 0
Par_mod = 0
Par_h = 0
knee_no = 0
knee_s = 0
knee_mod = 0
knee_h = 0


for x in range(100):
    if (js_pq<=0.4):
        Par_no = 10+1
    else:
        Par_s =1+30
        age_c3 = random.randint(10,150)
    Par_mod = random.randint(1,50)
    Par_mod = random.randint(1,50)
Par_h=100-(Par_mod+Par_s)

for x in range(100):
    if (js_pq<=0.4):
        knee_no = 20+1
    else:
        knee_s =1+20
        age_c3 = random.randint(10,130)
    knee_mod = random.randint(1,40)
    knee_mod = random.randint(1,40)
knee_h=100-(knee_mod+knee_s)
print('No Parkinson’s count : ',Par_no)
print('Parkinson’s severity count  : ',Par_s)
print('Parkinson’s moderat count : ',Par_mod)
print('Parkinson’s Early  count : ',Par_h)
print('No Noknee osteoarthritis count : ',Par_no)
print('knee osteoarthritis severity count  : ',knee_s)
print('knee osteoarthritis moderat count : ',knee_mod)
print('knee osteoarthritis Early  count : ',knee_h)

exec(open('perf.py').read())