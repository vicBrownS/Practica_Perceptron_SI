def getdataset(images, labels, caracteres, num_pix):
    import numpy as np
    from skimage.transform import resize
    import warnings
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

    X = np.zeros((len(images), num_pix * num_pix))
    Y = np.empty(len(images))

    images_resize = []

    for i in range(len(images)):
        images_resize.append(resize(images[i], (num_pix, num_pix)))
    try:
        for i in range(len(images_resize)):
            X[i] = images_resize[i].reshape(images_resize[i].size).tolist()
            Y[i] = caracteres[labels[i]]
    except ValueError:
        pass

    return X, Y