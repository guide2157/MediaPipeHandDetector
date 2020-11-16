# vim: expandtab:ts=4:sw=4
class FingerTrackState:
    Confirmed = 1
    Deleted = 2


class FingerTrack:
    """
    A single target track with state space `(landmark 8, landmark 12, landmark 16, landmark 20)`

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    """

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        self.time_since_update = 0
        self._max_age = 5
        self.state = FingerTrackState.Confirmed

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.finger_landmark)
        self.time_since_update = 0

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.time_since_update > self._max_age:
            self.state = FingerTrackState.Deleted

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == FingerTrackState.Deleted
