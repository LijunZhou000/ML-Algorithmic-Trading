from .registry import FEATURE_REGISTRY, feature

# Importar submódulos para que los decoradores se ejecuten
from .trend import *
from .momentum import *
from .volatility import *
from .volume import *
from .derived import *
from .structural import *
from .temporal import *
from .returns import *