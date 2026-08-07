# Programmatic Usage

A minimal wrapper class for calling FastMODA from Python, plus a full feature-extraction
pipeline example. See [REST API Reference](rest-api-reference.md) for the underlying
endpoints.

```python
import requests
import numpy as np
import time

class FastMODAFeatureExtractor:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url

    def _wait_for_task(self, task_id):
        while True:
            status = requests.get(f'{self.base_url}/status/{task_id}').json()
            if status['status'] in ['complete', 'error']:
                return status
            time.sleep(0.5)

    def extract_spectral_features(self, signal, fs=100.0):
        np.save('_temp.npy', signal)
        response = requests.post(f'{self.base_url}/analyze',
            files={'file': open('_temp.npy', 'rb')},
            data={'fs': fs, 'win': 1.0, 'pen': 'auto'})
        task_id = response.json()['task_id']
        status = self._wait_for_task(task_id)

        if status['status'] == 'error':
            raise ValueError(f"Analysis failed: {status.get('error')}")

        results = status['results']
        features = {}
        for i, comp in enumerate(results['frequency_summary'][:5]):
            features[f'freq_{i}_hz'] = comp['frequency']
            features[f'freq_{i}_duration_pct'] = comp['duration_pct']
            features[f'freq_{i}_band'] = self._band_to_num(comp['band'])

        features['n_changepoints'] = status['num_changepoints']
        features['changepoint_density'] = status['num_changepoints'] / status['num_windows']
        return features

    def extract_coherence_features(self, signal1, signal2, fs=100.0):
        np.save('_temp1.npy', signal1)
        np.save('_temp2.npy', signal2)
        response = requests.post(f'{self.base_url}/analyze_coherence',
            files=[('files', open('_temp1.npy', 'rb')), ('files', open('_temp2.npy', 'rb'))],
            data={'fs': fs, 'win': 1.0, 'overlap': 0.5})
        task_id = response.json()['task_id']
        status = self._wait_for_task(task_id)
        return {'coherence_available': status['status'] == 'complete'}

    @staticmethod
    def _band_to_num(band):
        mapping = {'delta': 1, 'theta': 2, 'alpha': 3, 'beta': 4, 'gamma': 5}
        return mapping.get(band, 0)


extractor = FastMODAFeatureExtractor()
signal = np.random.randn(10000)
features = extractor.extract_spectral_features(signal, fs=100.0)

for name, value in features.items():
    print(f"  {name}: {value}")
```

This produces a flat dict of numeric features per signal — ready to feed straight into
a classifier, as shown in [Training a Classifier](training-a-classifier.md).

## curl one-liners

```bash
# Spectral features
curl -s -X POST http://localhost:5000/analyze -F "file=@signal.npy" -F "fs=100" | jq

# Poll
curl -s http://localhost:5000/status/<task_id> | jq

# GPU availability
curl -s http://localhost:5000/api/gpu-info | jq
```
