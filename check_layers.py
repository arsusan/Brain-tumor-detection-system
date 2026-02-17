from research.src.model import BrainTumorModel
from research.src.config import Config
cfg = Config()
m = BrainTumorModel(cfg).build_cnn()
for layer in m.layers:
    print(layer.name)