from cProfile import label
import glob
import os
from tempfile import tempdir
from tkinter import SE
from unittest import result
import numpy as np
import cv2
import torch
import torchvision
from torchvision import  transforms
from PIL import Image

class ChangeDetectionMetric(object):
    def __init__(self, numClass=2):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def P_R_F1_IoU(self):
        hist = self.confusionMatrix
        p = hist[0,0]/( hist[0,0] + hist[0,1] + 1e-50)
        r = hist[0,0]/( hist[0,0] + hist[1,0] + 1e-50)
        f1 = 2*p*r/(p+r+1e-50)
        iou = hist[0,0]/( hist[0,0] + hist[0,1] + hist[1,0] + 1e-50)
        return p,r,f1,iou
        


    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return np.flip(np.flip(confusionMatrix,axis=0),axis=1)



    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def voc_rand_crop(feature1, feature2, label, height, width):
    """随机裁剪特征和标签图像。"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature1, (height, width))
    feature1 = torchvision.transforms.functional.crop(feature1, *rect)
    feature2 = torchvision.transforms.functional.crop(feature2, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature1, feature2, label


def metric_net(net,mode='test',crop_size=224):
    #metric
    metric = ChangeDetectionMetric(2) 
    #net
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #data
    list_image1 = glob.glob(os.path.join('..','input','sysucd',mode,mode,'time1','*.png'))
    list_image2 = glob.glob(os.path.join('..','input','sysucd',mode,mode,'time2','*.png'))
    list_label = glob.glob(os.path.join('..','input','sysucd',mode,mode,'label','*.png'))
    list_image1.sort()
    list_image2.sort()
    list_label.sort()
    #data transform
    transform0 = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for idx in range(len(list_label)):
        img1_path = list_image1[idx]
        img1 = transform(Image.open(img1_path))
        img2_path = list_image2[idx]
        img2 = transform(Image.open(img2_path))
        label_path = list_label[idx]
        label = transform0(Image.open(label_path))
        img1,img2,label = voc_rand_crop(img1, img2, label, crop_size, crop_size)
        img = torch.cat([img1,img2],dim=0)
        label = label.reshape(crop_size,crop_size)
        img_input=torch.unsqueeze(img, 0)

        pre=net(img_input.to(device))
        preint=torch.squeeze(pre, 0)
        predict = preint.argmax(dim=0)

        imgPredict = predict.cpu().clone()  # 可直接换成预测图片
        imgLabel = label.cpu().clone() # 可直接换成标注图片
        
        hist = metric.addBatch(imgPredict, imgLabel)
        if(idx%1000==0):
            print(idx)
            print(hist)
    temp=metric.P_R_F1_IoU()
    pa= metric.pixelAccuracy()
    P=temp[0]
    R=temp[1]
    F1=temp[2]
    IoU=temp[3]
    print("pA=",pa)
    print("P=",P)
    print("R=",R)
    print("F1=",F1)
    print("IoU=",IoU)
    

    


if __name__ =="__main__":
    metric = ChangeDetectionMetric()
    pred = np.array([[1,1,1],
                     [0,1,1],
                     [0,0,0]])
    label = np.array([[0,0,0],
                    [1,1,1],
                    [0,0,0]])
    metric.addBatch(pred, label)
    # pred = np.array([[0,1,1],
    #                  [0,1,1],
    #                  [0,0,0]])
    # label = np.array([[0,0,0],
    #                 [1,1,1],
    #                 [0,0,0]])
    # metric.addBatch(pred, label)
    # pred = np.array([[0,0,1],
    #                  [0,0,0],
    #                  [0,0,0]])
    # label = np.array([[0,0,0],
    #                 [1,1,1],
    #                 [0,0,0]])
    # metric.addBatch(pred, label)
    p,r,f1,iou = metric.P_R_F1_IoU()
    print("p=",p)
    print("r=",r)
    print("f1=",f1)
    print("iou=",iou)

















        # def Frequency_Weighted_Intersection_over_Union(self):
        # """
        # FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        # """

        # freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        # iu = np.diag(self.confusion_matrix) / (
        #         np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #         np.diag(self.confusion_matrix))
        # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        # return FWIoU