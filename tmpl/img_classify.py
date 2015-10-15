import os
import sys
import numpy as np
import pandas as pd
from skimage.transform import resize
import skimage.io

sys.path.append('/home/heatonli/caffe-master-201506026/python/')
import caffe


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls._instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance

class imgClassify():
    __metaclass__ = Singleton

    def __init__(self):
        self.root_dir = '/home/heatonli/caffe-master-201506026/models/bvlc_reference_caffenet/'
        self.imagefile = self.root_dir + 'cat.jpg'

        pretrained = self.root_dir + 'bvlc_reference_caffenet.caffemodel'
        model_file = self.root_dir + 'deploy.prototxt'

        mean, channel_swap = None, None

        # Make the classifier using bvlc_reference_caffenet model
        self.classifier = caffe.Classifier(model_file, pretrained, \
            image_dims=(256,256), mean=mean, \
                raw_scale=255.0, channel_swap=channel_swap)



    def classify(self, imagefile):
        input_img = [self.load_image(imagefile)]
        oversample = True

        # Predict the features in the image
        scores = self.classifier.predict(input_img, oversample).flatten()

        labels_file = self.root_dir + 'synset_words.txt'

        with open(labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        labels = labels_df.sort('synset_id')['name'].values

        indices = (-scores).argsort()[:5]
        predictions = labels[indices]

        meta = []
        res = 0
        for i, p in zip(indices,predictions):
            if scores[i] > 0.1:
                meta.append((p, '%.5f' % scores[i]))
                res += 1
        if res == 0:
            return 'Nothing recognised! Please try another picture!'
        else:
            return meta


    # Load image and change the format
    def load_image(self, filename, color=True):
        img = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            if color:
                img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

