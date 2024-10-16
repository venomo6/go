from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
x= np.arange(128)/128
signals = np.sin(2**np.pi**x)
signals_noice= signals + rng.standard_normal(len(signals))
corr = signal.correlate(signals_noice, signals)
lags = signal.correlation_lags(len(signals), len(signals_noice))
corr /= np.max(corr)
fig, (ax_orig, ax_noice, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
ax_orig.plot(signals)
ax_noice.plot(signals_noice)
ax_corr.plot(lags,corr)
ax_orig.margins(0, 0.1)
ax_noice.margins(0, 0.1)
ax_corr.margins(0, 0.1)
fig.tight_layout()
plt.show()

