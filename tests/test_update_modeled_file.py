import sys
sys.path.append('../')
import build_dataset

dataset_loader = build_dataset.dataset()
print("Building new activity dataset")
dataset_loader.build_new_dataset()
print("Building new ef coef dataset")
dataset_loader.calculate_activity_ef_params(update=True)