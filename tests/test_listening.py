import tempfile
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from superdj.listening import excerpt_audit


class ListeningMeasurementsTests(unittest.TestCase):
    def test_known_bass_reduction_and_human_preference_remain_separate(self):
        sr = 16000
        t = np.arange(sr*16) / sr
        x = .2*np.sin(2*np.pi*80*t)
        x[6*sr:10*sr] *= .5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'control.wav'
            sf.write(path, np.column_stack((x,x)), sr, subtype='FLOAT')
            audit = excerpt_audit(path, 6, 10)
        self.assertAlmostEqual(audit['bass_35_180_hz']['minimum_relative_to_preceding_db'], -6.02, places=1)
        self.assertFalse(audit['taste_predicted'])
        self.assertFalse(audit['human_auditioned_by_agent'])

    def test_short_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'short.wav'
            sf.write(path, np.zeros((16000*8, 2)), 16000, subtype='FLOAT')
            with self.assertRaises(ValueError):
                excerpt_audit(path, 1, 4)
