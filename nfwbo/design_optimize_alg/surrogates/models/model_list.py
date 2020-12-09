from design_optimize_alg.surrogates.models.FabolasModel import FabolasModel
from design_optimize_alg.surrogates.models.FidelitywarpingModel import FidelitywarpingModelWrapper
from design_optimize_alg.surrogates.models.SinglegpModel import SinglegpModel

model_list = dict()
model_list['fabolas'] = FabolasModel
model_list['fidelitywarping'] = FidelitywarpingModelWrapper
model_list['singlegp'] = SinglegpModel
