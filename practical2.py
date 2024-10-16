from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,1,500)
plt.plot(t,signal.sawtooth(2*np.pi*3*t, 0.5))
plt.xlabel("time")
plt.ylabel("amplitude")
plt.title("Triangualr signal")
plt.show()
