def analisis_binario(Ximage,yimage, caracteres):
    """
    La función análisis binario realizará un análisis en las métricas
    y en los resultados de entrenar al perceptron simple de sklearn
    con cada combinación binaria posible de los caracteres en el diccionario
    caracteres.

    Luego se visualizará la matriz de confusión de aquellos pares de caracteres
    con los que el perceptrón tenga un peor desempeño.

    :param Ximage: dataset con las imágenes y sus pixeles.
    :param yimage: dataset con los labels de cada imágen
    :param caracteres: diccionario con todos los caracteres que aparecen en las imágenes
    """
    from sklearn.linear_model import Perceptron

    from sklearn import metrics

    from sklearn.model_selection import train_test_split

    import numpy as np

    from matplotlib import pyplot as plt

    #Para evitar la repetición al comparar caracteres que nos den los mismos resultados
    # (e.j comparación de {0,o} y de {o,0}
    #Se comprobará primero si aquel par de caracteres ya ha sido comparado

    lista_caracteres_ya_pareados = [] #Lista donde se guardaran los caracteres pareados
    ya_comparados = False

    for caracter_1 in caracteres.values():
        for caracter_2 in caracteres.values(): #Cada caracter itera sobre todos los demás

            set_caracteres = {caracter_1, caracter_2} #Set con los caracteres a comparar
            if caracter_1 == caracter_2: #Si son iguales también se salta la iteración
                continue
            for par in lista_caracteres_ya_pareados: #Se itera sobre la lista de ya_pareados
                if set_caracteres == par: #Si los sets son iguales
                    ya_comparados = True
                    break #Se rompe el primer bucle for
            if ya_comparados: #Si ya han sido comparados se sale del segundo bucle
                ya_comparados = False
                continue
            lista_caracteres_ya_pareados.append(set_caracteres) #Se añade el par a la lista

            #Se comienza con la separacion del dataset en función de los caracteres
            X_binario = Ximage[((yimage == caracter_1)| (yimage == caracter_2))] #X con solo los caracteres correspondientes
            y_binario = yimage[((yimage == caracter_1) | (yimage == caracter_2))] #Y con solo los caracteres correspondientes
            y_binario_num = (y_binario == caracter_1).astype(int)  #Se pasa Y a binario 0,1

            #Se divide X_binario e y_binario_num en los sets de entrenamiento y test
            X_train, X_test, y_train, y_test = train_test_split(X_binario, y_binario_num, stratify=y_binario_num, train_size = 0.7 )

            clf = Perceptron(random_state=None, eta0= 0.1, shuffle=True, fit_intercept=True) #Creamos el perceptrón simple
            clf.fit(X_train, y_train) #Se entrena el perceptron
            y_pred_binario = clf.predict(X_test) #Se guarda la predicción

            accuracy= round(metrics.accuracy_score(y_test,y_pred_binario),3) #Se obtiene la accuracy
            sensibilidad = round(metrics.recall_score(y_test,y_pred_binario),3) #Se obtiene la sensibilidad
            precision = round(metrics.precision_score(y_test,y_pred_binario),3) #Se obtiene la precision

            print("Exactitud:" + str(accuracy)+ "de los valores("+ str(caracter_1)+"," + str(caracter_2))
            print("Sensibilidad:" + str(sensibilidad) + "de los valores(" + str(caracter_1) + "," + str(caracter_2))
            print("Precision:" + str(precision) + "de los valores(" + str(caracter_1) + "," + str(caracter_2))

            #Visualizamos la matriz de confusión
            labels=np.unique(y_test)
            conf_mat= metrics.confusion_matrix(y_test,y_pred_binario,labels=labels)
            cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat,display_labels= [str(caracter_1),str(caracter_2)])

            fig,ax =plt.subplots(figsize=(10,10))
            cm_display.plot(ax=ax)

            plt.title("Matriz de confusión Perceptron simple de Sklearn(" + str
            (caracter_1)+"," + str(caracter_2)+ ")")
            plt.xticks(rotation='vertical')
            plt.show()