def entrena_perceptron(X, y, z, eta, t, funcion_activacion):
    """ Entrena un perceptron simple
    Parámetros
    ----------
        X -- valores de x_i para cada uno de los datos de entrenamiento
        y -- valor de salida deseada para cada uno de los datos de entrenamiento
        z -- valor del umbral para la función de activación
        eta -- coeficiente de aprendizaje
        t -- numero de epochs o iteraciones que se quieren realizar con los datos de entrenamiento
        funcion_activacion -- función de activación para el perceptrón.

    Devolución
    --------
        w -- valores de los pesos del perceptron
        J -- error cuadrático obtenido de comparar la salida deseada con la que se obtiene
            con los pesos de cada iteración
    """

    import numpy as np

    # inicialización de los pesos
    w = np.zeros(len(X[0]))
    n = 0                           # numero de iteraciones se inicializa a 0

    # Inicialización de variables adicionales
    yhat_vec = np.zeros(len(y))     # array para las predicciones de cada ejemplo
    errors = np.zeros(len(y))       # array para los errores (valor real - predicción)
    J = []                          # error total del modelo


    while n < t:
        for i in range(0,len(X)):
            # Hayamos el producto escalar entre el vector peso y el de entrada
            # De forma que hayamos el sumatorio del producto entre los elementos
            escalar_prod = np.dot(w, X[i])
            # Aplicamos la función escalon al producto escalar
            resultados_prediccion = funcion_escalon(escalar_prod,z)
            if resultados_prediccion != y[i]:
                for weights in range(0,len(w)):
                    w[weights] = w[weights] + eta * (y[i] - resultados_prediccion) * X[i][weights]  # actualiza peso w[j] de acuerdo con el error

        n += 1  # se incrementa el número de iteraciones
        # calculo del error cuadrático del modelo
        # esto no es más que la suma del cuadrado de la resta entre el valor real
        # y la predicción que se tiene en esa iteración
        for i in range(0, len(y)):
            errors[i] = (y[i] - yhat_vec[i]) ** 2
        J.append(0.5 * np.sum(errors))

    return w, J


def predice(w,X,z,funcion_escalon):
    import numpy as np
    """ Función de para la predicción
        Parámetros
        ----------
            w -- array con los pesos obtenidos en el entrenamiento del perceptrón
            x -- valores de x_i para cada uno de los datos de test
            z -- valor del umbral para la función de activación
            funcion_activacion -- función de activación para el perceptrón.

        Devolución
        --------
            y -- array con los valores predichos para los datos de test
        """
    resultado_array = np.zeros(len(X)) #Inciamos
    for x in range(0, len(X)):
        escalar_prod = np.dot(X[x], w)
        resultado_prediccion = funcion_escalon(escalar_prod, z)
        resultado_array[x] = resultado_prediccion
    return resultado_array


def evalua(y_test, y_pred):
    """ Función de activación
    Parámetros
    ----------
        y_test -- array con los valores salida conocidos para los datos de test
        y_test -- array con los valores salida estimados por el perceptrón para los datos de test

    Devolución
    --------
        acierto -- float con el valor del porcentaje de valores acertados con respecto al total de elementos
    """
    acierto = float(0)
    for i in range(0,len(y_test)):
        acierto+=1 if y_test[i] == y_pred[i] else 0
    return acierto/len(y_test)

def funcion_escalon(a,z):
    """ Función de activación
    Parámetros
    ----------
        a -- array con los valores de sumatorio de los elementos de un caso de entrenamiento
        z -- valor del umbral para la función de activación

    Devolución
    --------
        yhat_vec -- array con valores obtenidos de g(f)
    """
    # función de activación
    yhat_vec = 1 if a > z else 0
    return yhat_vec