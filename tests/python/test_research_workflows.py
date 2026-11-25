import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dkt_agent.research_tools import TRAJECTORY_ROOT, diagnose_observable_timeseries

class TestExpandedResearchWorkflows(unittest.TestCase):
    def setUp(self):
        TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix='workflow_test_', dir=TRAJECTORY_ROOT))

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_observable_diagnostics(self):
        path = self.folder / 'trajectory.csv'
        t = np.arange(120, dtype=float)
        theta = 0.8 + 0.1*np.sin(0.13*t)
        phi = 0.2*t
        x, y, z = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle); writer.writerow(['step','X','Y','Z','norm'])
            for i in range(len(t)): writer.writerow([i,x[i],y[i],z[i],1.0])
        result = json.loads(diagnose_observable_timeseries.invoke({
            'trajectory_csv': str(path), 'transient': 5, 'max_lag': 12, 'block_sizes': [5,10]
        }))
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['samples_analyzed'], 114)
        self.assertIn('X2', result['birkhoff_time_averages'])
        self.assertEqual(len(result['autocorrelation']['observables']['X']), 13)
        self.assertEqual([r['block_size'] for r in result['block_fluctuations']['X']], [5,10])

if __name__ == '__main__': unittest.main(verbosity=2)
