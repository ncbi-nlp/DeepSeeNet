import logging

import numpy as np

from deepseenet import deepseenet_drusen, deepseenet_pigment, deepseenet_adv_amd


def get_simplified_score(scores):
    """
    Get AREDS simplified severity score from drusen size, pigmentary abnormality, and advanced AMD.

    Args:
        scores: a dict of individual risk factors

    Returns:
        a score of 0-5
    """
    def has_adv_amd(score):
        return True if score == 1 else False

    def has_pigment(score):
        return True if score == 1 else False

    def has_large_drusen(score):
        return True if score == 2 else False

    def has_intermediate_drusen(score):
        return True if score == 1 else False

    score = 0
    if has_adv_amd(scores['advanced_amd'][0]):
        score += 5
    if has_adv_amd(scores['advanced_amd'][1]):
        score += 5
    if has_pigment(scores['pigment'][0]):
        score += 1
    if has_pigment(scores['pigment'][1]):
        score += 1
    if has_large_drusen(scores['drusen'][0]):
        score += 1
    if has_large_drusen(scores['drusen'][1]):
        score += 1
    if has_intermediate_drusen(scores['drusen'][0]) \
            and has_intermediate_drusen(scores['drusen'][1]):
        score += 1

    return 5 if score >= 5 else score


class EyesNetSimplifiedScore(object):
    def __init__(self, drusen_model='areds', pigment_model='areds', advanced_amd_model='areds'):
        """
        Args:
            drusen_model: Path or file object.
            pigment_model: Path or file object.
            advanced_amd_model: Path or file object.
        """
        self.drusen = deepseenet_drusen.DeepSeeNetDrusen(drusen_model)
        self.pigment = deepseenet_pigment.DeepSeeNetPigment(pigment_model)
        self.adv = deepseenet_adv_amd.DeepSeeNetAdvancedAMD(advanced_amd_model)
        self.models = {
            'drusen': (self.drusen, deepseenet_drusen.preprocess_image),
            'pigment': (self.pigment, deepseenet_pigment.preprocess_image),
            'advanced_amd': (self.adv, deepseenet_adv_amd.preprocess_image),
        }

    def predict(self, x_left, x_right, verbose=0):
        """
        Generates simplified severity score for one left eye and one right eye

        Args:
            x_left: input data of the left eye, as a Path or file object.
            x_right: input data of the right eye, as a Path or file object.
            verbose: Verbosity mode, 0 or 1.

        Returns:
            Numpy array of scores of 0-5
        """
        assert x_left.shape[0] == x_right.shape[0]
        scores = {}
        for model_name, (model, preprocess_image) in self.models.items():
            left_score = np.argmax(model.predict(preprocess_image(x_left)), axis=1)[0]
            right_score = np.argmax(model.predict(preprocess_image(x_right)), axis=1)[0]
            scores[model_name] = (left_score, right_score)
        if verbose == 1:
            logging.info('Risk factors: %s', scores)
        return get_simplified_score(scores)
