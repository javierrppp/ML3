import pickle
import ensemble
import feature
import math
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def normalize():
    positive_features = []
    negative_features = []
    for i in range(500):
        n = '%03d' % i
        filepath = './datasets/original/face/face_' + n + '.jpg'
        savepath = './datasets/pre-processing/face/face_' + n + '.jpg'
        image = Image.open(filepath).convert('L')
        image.thumbnail((24,24))
        image.save(savepath)
        mat = np.array(image)
        NPD = feature.NPDFeature(mat)
        fea = NPD.extract()
        positive_features.append(fea)
    np.save("./datasets/nonface_features.npy" ,positive_features)

    for i in range(500):
        n = '%03d' % i
        filepath = './datasets/original/nonface/nonface_' + n + '.jpg'
        savepath = './datasets/pre-processing/nonface/nonface_' + n + '.jpg'
        image = Image.open(filepath).convert('L')
        image.thumbnail((24,24))
        image.save(savepath)
        mat = np.array(image)
        NPD = feature.NPDFeature(mat)
        fea = NPD.extract()
        negative_features.append(fea)
    np.save("./datasets/face_features.npy" ,negative_features)

def generateDateset(face_features,nonface_features):
    
    positive_label = [np.ones(1) for i in range(face_features.shape[0])]
    negative_label = [-np.ones(1) for i in range(nonface_features.shape[0])]

    positive_samples = np.concatenate((face_features, positive_label), axis=1)
    negative_samples = np.concatenate((nonface_features, negative_label), axis=1)

    features_dataset = np.concatenate((positive_samples, negative_samples), axis=0)

    np.random.shuffle(features_dataset)
    X = features_dataset[:,:features_dataset.shape[1]-1]
    y = features_dataset[:,features_dataset.shape[1]-1:]
    
    np.save("X.npy", X)
    np.save("y.npy", y)

if __name__ == "__main__":
    #normalize()
    #face_features = np.load("./datasets/face_features.npy")
    #nonface_features = np.load("./datasets/nonface_features.npy")
    #generateDateset(face_features, nonface_features)

    X = np.load("X.npy")
    y = np.load("y.npy")
    mode = DecisionTreeClassifier(criterion='gini')
    adaBoost = ensemble.AdaBoostClassifier(mode, 10)
    xTrain, xValidation, yTrain, yValidation = train_test_split(X, y, test_size=0.5, random_state=42)
    m = adaBoost.n_weakers_limit
    for i in range(m):
        mode = adaBoost.fit(xTrain, yTrain)
        xTest = adaBoost.predict(xValidation)
        errorRate=0;
        for j in range(xValidation.shape[0]):
            if xTest[j] != yValidation[j]:
                errorRate =errorRate+adaBoost.weight[j]
        if errorRate>0.5:
            break
        alpha = math.log((1-errorRate)/errorRate)/2
        z=0
        for k in range(adaBoost.weight.shape[0]):
            z=z+adaBoost.weight[k]*math.exp(-alpha*yTrain[k]*xTest[k])

        for k in range(adaBoost.weight.shape[0]):
            adaBoost.weight[k]=adaBoost.weight[k]*math.exp(-alpha*yTrain[k]*xTest[k])/z
        adaBoost.weak_classifier_list.append(mode)
        adaBoost.alphas.append(alpha)
    h = adaBoost.predict_scores(xValidation)
   
    labels=[-1,1]
    target_names = ['face','nonface']
    result = classification_report(yValidation,h,labels,target_names)
    print(result)
    report_path = "report.txt"
    with open(report_path, "wb") as f:
        f.write(result)
