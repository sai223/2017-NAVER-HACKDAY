import numpy as np
import pickle
#import nest
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import KFold

new_man = []
man_file = open("value_pickle.txt", 'rb')
label_file = open("name_pickle.txt", 'rb')
entire_file = []
labels = []

for i in range(12800):
    new_man = pickle.load(man_file)
    label = pickle.load(label_file)
    #print(new_man)
    entire_file.append(new_man)
    labels.append(label)

entire_file = np.array(entire_file)
print(np.shape(entire_file))
#print(new_man)

labels = np.array(labels)
classifier = NearestNeighbors(n_neighbors=10)
classifier.fit(entire_file)
predict = classifier.kneighbors([entire_file[0]], return_distance=True)
print(predict)
print(labels[predict[1]])

#kf = KFold(len(entire_file), n_folds=3, shuffle=True)
#for training in entire_file:
#    classifier.fit(training, entire_file)
    #prediction = classifier.predict(features[testing])
