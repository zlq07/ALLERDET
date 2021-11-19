from sklearn.grid_search import GridSearchCV
import time
def search_better_perform_of_rb(trainX, trainY):
    trainX = np.asarray(trainX, 'float32')
    trainX = scale(trainX) # 0-1 scaling

    #perform a grid search on the 'C' parameter of Logistic
    # Regression
    print ("SEARCHING LOGISTIC REGRESSION")
    params = {"C": [1.0, 10.0, 100.0]}
    start = time.time()
    gs = GridSearchCV(linear_model.LogisticRegression(), params, verbose = 1)
    gs.fit(trainX, trainY)

    # print diagnostic information to the user and grab the
    # best model
    print ("done in %0.3fs" % (time.time() - start))
    print ("best score: %0.3f" % (gs.best_score_))
    print ("LOGISTIC REGRESSION PARAMETERS")
    bestParams = gs.best_estimator_.get_params()

    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
            print ("\t %s: %f" % (p, bestParams[p]))

    # initialize the RBM + Logistic Regression pipeline
    rbm = BernoulliRBM()
    logistic = linear_model.LogisticRegression()
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

    # perform a grid search on the learning rate, number of
    # iterations, and number of components on the RBM and
    # C for Logistic Regression
    print ("SEARCHING RBM + LOGISTIC REGRESSION")
    params = {
            "rbm__learning_rate": [0.1, 0.01, 0.001],
            "rbm__n_iter": [20, 40, 80, 100],
            "rbm__n_components": [50, 100, 200, 300, 500, 1000],
            "logistic__C": [1.0, 10.0, 100.0, 500.0, 1000.0]}

    # perform a grid search over the parameter
    start = time.time()
    gs = GridSearchCV(classifier, params, verbose = 1)
    gs.fit(trainX, trainY)

    # print diagnostic information to the user and grab the
    # best model
    print ("\ndone in %0.3fs" % (time.time() - start))
    print ("best score: %0.3f" % (gs.best_score_))
    print ("RBM + LOGISTIC REGRESSION PARAMETERS")
    bestParams = gs.best_estimator_.get_params()

    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
            print ("\t %s: %f" % (p, bestParams[p]))

    # show a reminder message
    print ("\nIMPORTANT")
    print ("Now that your parameters have been searched, manually set")
