from ..context import deepseenet


def test_advanced_amd():
    for drusen in (0, 1, 2):
        for pigment in (0, 1):
            scores = {
                'drusen': (drusen, 0),
                'pigment': (pigment, 0),
                'advanced_amd': (0, 1)
            }
            assert deepseenet.model.get_simplified_score(scores) == 5
            scores['advanced_amd'] = (1, 0)
            assert deepseenet.model.get_simplified_score(scores) == 5
            scores['advanced_amd'] = (1, 1)
            assert deepseenet.model.get_simplified_score(scores) == 5


def test_single_eye():
    simple_scores = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 0,
        (1, 1): 1,
        (2, 0): 1,
        (2, 1): 2
    }
    scores = {'advanced_amd': (0, 0)}
    for k in simple_scores:
        scores['drusen'] = (k[0], 0)
        scores['pigment'] = (k[1], 0)
        assert deepseenet.model.get_simplified_score(scores) == simple_scores[k]
        scores['drusen'] = (0, k[0])
        scores['pigment'] = (0, k[1])
        assert deepseenet.model.get_simplified_score(scores) == simple_scores[k]
        scores['drusen'] = (k[0], 0)
        scores['pigment'] = (0, k[1])
        assert deepseenet.model.get_simplified_score(scores) == simple_scores[k]
        scores['drusen'] = (k[0], 0)
        scores['pigment'] = (0, k[1])
        assert deepseenet.model.get_simplified_score(scores) == simple_scores[k]


def test_both_eyes():
    simple_scores = {
        (0, 0): 0,
        (0, 1): 2,
        (1, 0): 1,
        (1, 1): 3,
        (2, 0): 2,
        (2, 1): 4
    }
    scores = {'advanced_amd': (0, 0)}
    for k in simple_scores:
        scores['drusen'] = (k[0], k[0])
        scores['pigment'] = (k[1], k[1])
        assert deepseenet.model.get_simplified_score(scores) == simple_scores[k]
