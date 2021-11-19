from predict import predict
from collections import Counter

print("Prueba Alérgenos")
cp,pt = predict(method="knn", params={'leaf_size': 1, 'algorithm': 'ball_tree', 'p': 1, 'metric': 'minkowski', 'weights': 'uniform', 'n_neighbors': 3}, showAllPredictions=False, featToExtract=[True,True]
    , webApp=False, plotPosAlgn=True, plotNegAlgn=True, plotTestAlgn=True, plotAIO=True)
counter = Counter(cp)
print(str(counter))
print("Accuracy: " + str((counter['allergen']*100)/len(cp)))