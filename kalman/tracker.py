# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from . import kalman_filter
from .track import Track
from .fingers_track import FingerTrack
from scipy.stats import chi2


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    finger_kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in finger coordinates.
    track : Track
        The active track at the current time step.

    """

    def __init__(self, chi_sq=0.95, max_iou_distance=0.7, max_age=4, n_init=4):
        # self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter(chi_sq)
        self.finger_kf = kalman_filter.KalmanFilter(0.5)
        self.track = None
        self.finger_track = None
        self.num_since_last_swipe = 6

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        if self.track:
            self.track.predict(self.kf)
        if self.finger_track:
            self.finger_track.predict(self.finger_kf)
        self.num_since_last_swipe += 1

    def detect_swipe(self, detection, moved_flag):

        finger_gating_threshold = chi2.ppf(self.finger_kf.chi_sq, df=4)

        gating_distance = self.kf.gating_distance(
            self.finger_track.mean, self.finger_track.covariance, detection.finger_landmark)
        if not moved_flag and gating_distance > finger_gating_threshold:
            if sum(self.finger_track.mean[:4] - detection.finger_landmark) > 0:
                print("swipe left")
            else:
                print("swipe right")
            self.finger_track = None
            self.num_since_last_swipe = 0
        else:
            self.finger_track.update(self.finger_kf, detection)

    def update(self, detection):
        """Perform measurement update and track management.

        Parameters
        ----------
        detection : deep_sort.detection.Detection
            A detection at the current time step.
        """
        if detection is None:
            if self.track is not None:
                self.track.mark_missed()
            return
        if self.track is None:
            self._initiate_track(detection)
            return
        elif self.finger_track is None and detection.finger_landmark and self.num_since_last_swipe > 15:
            self._initiate_finger_track(detection)

        gating_threshold = chi2.ppf(self.kf.chi_sq, df=2)

        gating_distance = self.kf.gating_distance(
            self.track.mean, self.track.covariance, detection.to_xy_index(), True)
        if gating_distance > gating_threshold:
            self.track.mark_missed()

        gating_threshold = chi2.ppf(0.85, df=2)
        moved_flag = False
        if gating_distance > gating_threshold:
            moved_flag = True

        if self.track.is_deleted():
            self.track = None
            self.finger_track = None
        elif self.finger_track and self.finger_track.is_deleted():
            self.finger_track = None
        else:
            self.track.update(self.kf, detection)
            if self.finger_track:
                if detection.finger_landmark:
                    self.detect_swipe(detection, moved_flag)
                else:
                    self.finger_track.mark_missed()

    def _initiate_finger_track(self, detection):
        mean, covariance = self.finger_kf.initiate(detection.finger_landmark)
        self.finger_track = FingerTrack(mean, covariance)

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xy_index())
        self.track = Track(
            mean, covariance, self.n_init, self.max_age)
        if detection.finger_landmark:
            self._initiate_finger_track(detection)
