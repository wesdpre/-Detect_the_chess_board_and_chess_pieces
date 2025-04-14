I will create a report explaining what I did for the chess board recognition.

Initially, we first had to preprocess the image. Although the image has already a good quality

Inicialmente nos dimos cuenta que la imagen tenia una buena calidad por lo que en un principio no se tuvo que aplicar ningun filtrado para quitarle el ruido a la imagen. Así que inmediatamente decidimos inmediatamante acercarnos con la detección de bordes con el filtro de canny. Con un gradiente entre 50 y 200 se estuvo comportando bien. Sin embargo, nos dimos cuenta que con estos parametros canny estaba detectando muchos bordes falsos positivos por el reflejo de la luz. logrmos un mejor acercamiento a los bordes limitando el threshold entre 150 y 200 pero aun así habiando imagenes donde la el relejo de la luz afectaba bastanta la deteccion de algunos bordes del tablero.

Es por esto que se tuvo la idea de oscurecer la imagen disminuyendo así el contraste de la luz, luego de eso, se volvió aplicar el filtro de canny a la imagen y nos dimos cuenta que no solo la detección de los bordes mejoró si no que tambien, se evitó tambien la detección del piso de la imagen dejando así solo la detección de los bordes del tablero (imagen 1).

Una vez obtenidos los bordes, empezamos a buscar los contornos de la imagen obteniendo todos los poligonos en ella. Como el tablero tiene 4 esquinas, se configuró por defecto que solo nos diera los poligonos que tuviese 4 puntos. 
Inicialmente, funcionó bien. Pero probando con otras imagenes de diferentes angulos nos dimos cuenta que entre más inclinado el angolo, las piesas más cercanas a los bordes del tablero sobrelapan el borde haciendo que la busqueda del mismo contornos se vuelva más complicado. Es allí donde encontramos que en este caso el contorno no siempre iba a ser lineal. Al contrario podríamos encontrar estas piezas que sobrelapan el borde afectarnos la busqueda del tablero. Relaizando una busqueda https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/ explica que podemos utilizar kernels que se utilizan para conectar regiones cercanas en la imagen como lo era un borde detrás de una pieza.

Ya detectado el tablero finalmente se procede a recortar el tablero pero antes de eso realizamos una transformación de perspectiva para simular que vemos en un angulo de 180 (perspecticva perpendicular al tablero). Para ello arreglo un cuadrado perfecto de 400 x 400 pixeles. Luego realizamos la transformación de los 4 puntos detectados del tablero alargando un poco la imagen y adecuandola al cuadrado perfecto.




