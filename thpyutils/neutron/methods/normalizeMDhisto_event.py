import numpy as np


def normalizeMDhisto_event(mdhisto):
    # Normalizes a binned workspace by number of events
    non_normalized_intensity = np.copy(mdhisto.getSignalArray())
    non_normalized_err = np.sqrt(np.copy(mdhisto.getErrorSquaredArray()))
    num_events = np.copy(mdhisto.getNumEventsArray())
    normalized_intensity = non_normalized_intensity / num_events
    normalized_error = non_normalized_err / num_events
    mdhisto.setSignalArray(normalized_intensity)
    mdhisto.setErrorSquaredArray(normalized_error ** 2)
    # Mantid will behave weirdly unless we make the number of events one.
    return mdhisto

def undo_normalizeMDhisto_event(mdhisto):
    # Normalizes a binned workspace by number of events
    non_normalized_intensity = np.copy(mdhisto.getSignalArray())
    non_normalized_err = np.sqrt(np.copy(mdhisto.getErrorSquaredArray()))
    num_events = np.copy(mdhisto.getNumEventsArray())
    normalized_intensity = non_normalized_intensity * num_events
    normalized_error = non_normalized_err * num_events
    mdhisto.setSignalArray(normalized_intensity)
    mdhisto.setErrorSquaredArray(normalized_error ** 2)
    # Mantid will behave weirdly unless we make the number of events one.
    return mdhisto
