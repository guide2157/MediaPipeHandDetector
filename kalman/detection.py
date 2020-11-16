# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlbr : array_like
        Bounding box in format `(x_min, y_min, x_max, y_max)`.
    finger_landmark : (landmark 8, landmark 12, landmark 16, landmark 20)
        X coordinates of four finger landmarks

    Attributes
    ----------
    tlbr : ndarray
        Bounding box in format `(x_min, y_min, x_max, y_max)`.

    """

    def __init__(self, tlbr, index_pos, finger_landmark=None):
        self.tlbr = np.asarray(tlbr, dtype=np.float)
        self.index_pos = index_pos
        self.finger_landmark = finger_landmark

    def to_tlwh(self):
        """Convert bounding box to format `(min x, min y, width, height)`.
                """
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.to_tlwh()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xy_index(self):
        x_min, y_min, x_max, y_max = self.tlbr
        return ((x_min + x_max) / 2, (y_max + y_min) / 2,) + self.index_pos
