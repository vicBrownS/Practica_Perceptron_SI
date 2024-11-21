def getdataset(images, labels, caracteres, num_pix):
    import numpy as np
    from skimage.transform import resize
    """ Obtiene los arrays de numpy con las imágenes y las etiquetas
    Parámetros
    ----------
        imagenes -- estructura de datos que contiene la información de cada una de las imágenes
        etiquetas -- estructura de datos que contiene la información de la clase a la que
        pertenece cada una de las imágenes
        caracteres -- diccionario que contiene la "traducción" a ASCII de cada una de las etiquetas
        num_pix -- valor de la resolución de la imagen (se debe obtener una imagen num_pix x num_pix)

    Devolución
    --------
        X -- array 2D (numero_imagenes x numero_pixeles) con los datos de cada una de las imágenes
        y -- array 1D (numero_imagenes) con el caracter que representa cada una de las imágenes
    """
    import warnings
    warnings.filterwarnings("ignore")
    #Iniciamos los arrays X e Y, el array X contendra 0s y
    # tendra como dimensiones el número de imágenes x el número de pixeles
    X = np.zeros((len(images), num_pix * num_pix))
    #El array Y tendrá una unica dimensión que será el numero de imágenes
    Y = np.empty(len(images), dtype= str)

    images_resize = [] #Lista donde guardaremos las imágenes con otro tamaño
    images_flat = []

    for i in range(len(images)): #Se recorren todas las imágenes y se las cambia de tamaño
        images_resize.append(resize(images[i], (num_pix, num_pix)))

    for i in range(len(images_resize)):
        images_flat.append(images_resize[i].reshape(images_resize[i].size).tolist())

    #Bloque try para posibles excepciones
    for i in range(len(images_resize)): #Se recorre images_resize
        try:
            Y[i] = caracteres[labels[i]]  # Se añade a Y el label correspondiente
        except ValueError:
            pass
        X[i] = images_flat[i]  # Se aplanan y se añaden a X

    return X, Y


