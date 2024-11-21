
"""
Se implementará la función crea_diccionario
que relaciona una clave del 0 al 46 con el caracter ASCII que representa,
esta relación será dada por el archivo claves_ASCII
que se encuentra en la carpeta DATA.
"""
def crea_diccionario(archivo_claves):
    claves = open(archivo_claves, 'r') #Abre el archivo indicado por el path
    diccionario = {} #Instancia el diccionario donde se guardarán los datos

    for line in claves: #Recorre cada linea en el archivo
        #Se utiliza el metodo strip() para quitar los espacios en blanco
        #Se utiliza el metodo split() para dividir los dos números de la linea
        clave = int(line.strip().split()[0])  #Se accede al primer número
        valor = chr(int(line.strip().split()[1])) #Se accede al segundo número y se le convierte en character
        diccionario[clave] = valor #Se añade al diccionario
    return diccionario