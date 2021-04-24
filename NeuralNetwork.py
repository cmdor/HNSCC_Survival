import pandas as pd
import math
import sys

#user settings
def main(data='ann_dataset.csv', headless=False ):

    dataset = pd.read_csv(data)
    oversample = True
    try:
        dataset.to_csv(("metrics/dataset.csv"), index=False)
    except:
        print("error: unable to write to file")
    inputs = (len(dataset.columns) - 1)
    #print(dataset.info())
    X = dataset.iloc[:, 0:-2].values
    y = dataset.iloc[:, -1].values
    
    
    #random resampling to reduce effect of minority class size

        
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    from imblearn.over_sampling import RandomOverSampler
    
    if(oversample):
        #oversampling from training set
        ros = RandomOverSampler(random_state=0)
        ros.fit(X_train,y_train)
        X_train, y_train = ros.fit_resample(X_train,y_train)
        
        #oversampling for test set
        ros.fit(X_test,y_test)
        X_test, y_test = ros.fit_resample(X_test,y_test)
        
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    # ------- Part-2: Build the ANN --------
    
    # import keras library and packages
    from keras.models import Sequential
    from keras.layers import Dense
    import livelossplot
    
    #creating the classifier and setting the layers...
    classifier = Sequential()
    classifier.add(Dense(inputs, activation='relu'))
    classifier.add(Dense(inputs, activation='relu'))
    classifier.add(Dense(math.floor(inputs), activation='sigmoid'))
    classifier.add(Dense(1, activation='sigmoid'))
    classifier.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Fitting the ANN to the training set
    num_epochs = 10
    batch = 100
    if(headless == False):
        classifier.fit(X_train, y_train,
                  batch_size=batch,
                  epochs=num_epochs,
                  callbacks = [livelossplot.PlotLossesKeras()],
                  verbose=1,
                  validation_data=(X_test, y_test))
    else:
        classifier.fit(X_train, y_train,
              batch_size=batch,
              epochs=num_epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    y_pred = classifier.predict(X_test)
    # Predicting the Test set results
    score = classifier.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    print("Classifier Summary")
    classifier.summary()
    #print("Y_test, Y_pred")
    
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred.round())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #write confusion matrix to file
    #cv2.imwrite("metrics/cm.img", cv2.imwrite(disp.plot()))
    if(headless == False):
        disp.plot()

    print("Confusion Matrix:")
    print(cm)
    
    tn = float(cm[0,0])
    fp = float(cm[0,1]) 
    tp = float(cm[1,1])   
    fn = float(cm[1,0])
    
    
    precision = tp/(tp + fp)
    recall = tp / (tp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2*(precision * recall)/(precision + recall)
    
    print("Precision: "+ str(precision * 100)+"%")
    print("Recall: "+ str(recall * 100)+"%")
    print("Sensitivity: "+ str(sensitivity * 100)+"%")
    print("Specificity: "+ str(specificity * 100)+"%")
    print("F1 Score: "+ str(f1))
    
if __name__ == "__main__":
    # execute only if run as a script
    args = ['ann_dataset.csv', False ]
    
    if(len(sys.argv) >= 1):
        if(len(sys.argv) >= 2):
            if(str(sys.argv[1]) == "--headless"):
                    args[1] = True
            else:
                args[0] = sys.argv[1]
            if(len(sys.argv) == 3):
                if(str(sys.argv[2]) == "--headless"):
                    args[1] = True
            if(len(sys.argv) > 3):
                print("incorrect number of arguments")
    for i in args:
        print(i)
    main(args[0], args[1])