def carga_data_MNIST(input_path, file_images, file_labels):
    """ Permite cargar los datos almacenados en formato IDX
    Parámetros
    ----------
        input_path: Carpeta en la que se encuentran los archivos
        file_images: nombre del archivo de imágenes
        file_labels: nombre del archivo de etiquetas

    Devolución
    --------
        images -- array 3D con los datos de cada imagen
        labels -- array 1D con el numero de clase de cada de las imágenes
    """
    import gzip

    import numpy as np
    import matplotlib.pyplot as plt

    from os.path import join

    images_path = join(input_path, file_images)
    labels_path = join(input_path, file_labels)

    ## Comenzamos con el archivo que contiene las imágenes:
    print("***********************************")
    # Instanciamos el manejador de archivo mediante la función open
    images_byte = gzip.open(images_path, 'r')

    # NÚMERO MAGICO
    head = images_byte.read(4)
    magic = int.from_bytes(head, "big")
    print("Numero mágico para el archivo de imágenes:")
    print(magic)

    # NUMERO DE IMAGENES
    head = images_byte.read(4)
    number_images = int.from_bytes(head, "big")
    print("Numero de imágenes:")
    print(number_images)

    # NUMERO DE FILAS DE LA IMAGEN
    head = images_byte.read(4)
    rows = int.from_bytes(head, "big")
    print("Numero de filas (imágenes):")
    print(rows)

    # NUMERO DE COLUMNAS DE LA IMAGEN
    head = images_byte.read(4)
    columns = int.from_bytes(head, "big")
    print("Numero de columnas (imágenes):")
    print(columns)

    # CARGA DE DATOS
    buf = images_byte.read(rows * columns * number_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    array = data.reshape(number_images, rows, columns)
    images = array.swapaxes(1, 2)

    ## A partir de aquí procesamos el archivo que contiene las etiquetas:
    print("***********************************")
    labels_byte = gzip.open(labels_path, 'r')

    # NÚMERO MAGICO
    head = labels_byte.read(4)
    magic = int.from_bytes(head, "big")
    print("Numero mágico para el archivo de etiquetas:")
    print(magic)

    # NUMERO DE ETIQUETAS
    head = labels_byte.read(4)
    number_labels = int.from_bytes(head, "big")
    print("Numero de etiquetas:")
    print(number_labels)

    # CARGA DE ETIQUETAS
    buf = labels_byte.read(number_labels)
    labels = np.frombuffer(buf, dtype=np.uint8)
    print("***********************************")

    return images, labels
