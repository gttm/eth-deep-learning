from __future__ import division

class TransitionType():
    ADDITION, ASSIGNMENT = range(2)


class BoundaryTransition():
    CLIP, MOD = range(2)

class SortingCriterion():
    PSD_MAX, PSD_MIN, PSD_MEAN, PSD_VAR, PSD_MODE, PSD_SKEW, PSD_KURT = range(7)
