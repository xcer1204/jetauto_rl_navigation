import rpyc
import numpy as np

c = rpyc.connect("localhost", 18862, config={"allow_pickle": True})

w2c = np.eye(4, dtype=np.float32)[None, ...].tolist()  # 关键：tolist()
print(c.root.visible_ratio(w2c))
