import numpy as np
import torch.nn as nn
from metric.inception import InceptionV3


## Step 3 : Your Implementation Here ##

## Implement functions for fid score measurement using InceptionV3 network ##
class FIDScoreCalculator(nn.Module):
    def __init__(self, root_dir):
        super(FIDScoreCalculator, self).__init__()

        self.root_dir = root_dir
        self.inception_v3 = InceptionV3()

    def get_image_files(self, string):
        import os

        image_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if string in file:
                    image_files.append(file)
        return image_files

    def extract_features(self, image_files):
        import cv2

        features = np.empty(0, 2048)
        for image_file in image_files:
            image = cv2.imread(image_file)
            feature = self.inception_v3(image)
            feature = feature.view(-1)
            feature = feature.data.numpy()

            np.append(features, feature)
            # TODO(bongjoon.hyun@gmail.com): fix this, it can't be executed
        return features

    def calculate_mean_and_variance(self, features):
        mean = np.mean(features, axis=0)
        variance = np.cov(features, rowvar=False)
        return mean, variance

    def calculate_fid_score(self, source, target):
        source_image_files = self.get_image_files(source)
        source_features = self.extract_features(source_image_files)
        source_mean, source_variance = self.calculate_mean_and_variance(
            source_features)

        target_image_files = self.get_image_files(target)
        target_features = self.extract_features(target_image_files)
        target_mean, target_variance = self.calculate_mean_and_variance(
            target_features)

        mean_diff = target_mean - source_mean
        covariance, _ = np.linalg.sqrtm(source_variance.dot(target_variance),
                                        disp=False)

        fid_square = mean_diff.dot(mean_diff) + np.trace(source_variance) + \
                     np.trace(target_variance) - 2 * covariance
        fid_score = np.sqrt(fid_square)
        return fid_score
