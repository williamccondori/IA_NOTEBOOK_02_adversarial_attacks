import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend as k
from keras_preprocessing import image
from keras.applications.inception_v3 import InceptionV3, decode_predictions


def train(input_image_path, output_image_path, target_class, batch_size=1, perturbation=.01):
    inception_v3 = InceptionV3()

    # Imprimir un resumen del modelo convolucional InceptionV3.
    # print(inception_v3.summary())

    # Se obtiene y redimensiona la imagen de entrada.
    x = image.img_to_array(image.load_img(input_image_path, target_size=(299, 299)))

    # InceptionV3 necesita un formato de imágen de intensidad -1, 1.
    # Se reescala el formato de imagen de entrada.
    x /= 255
    x -= .5
    x *= 2

    # Se agrega en batch-size a la matriz x.

    x = x.reshape([batch_size, x.shape[0], x.shape[1], x.shape[2]])

    # Se realiza la predicción utilizando el modelo InceptionV3.
    y = inception_v3.predict(x)

    # print(decode_predictions(y))

    """
    Adversarial attacks implementation
    """

    # Obtenemos la primera y ultima capa de la red InceptionV3.
    input_layer = inception_v3.layers[0].input
    output_layer = inception_v3.layers[-1].output

    loss = output_layer[0, target_class]
    gradient = k.gradients(loss, input_layer)[0]
    optimize_gradient = k.function([input_layer, k.learning_phase()], [gradient, loss])

    adversarial = np.copy(x)

    min_perturbation = x - perturbation
    max_perturbation = x + perturbation

    cost = .0

    while cost < .95:
        gr, cost = optimize_gradient([adversarial, 0]) # k.learning_phase = 0
        adversarial += gr

        # Se restringe la actualización de píxeles para evitar perturbaciones.
        adversarial = np.clip(adversarial, min_perturbation, max_perturbation)
        adversarial = np.clip(adversarial, -1, 1)

        # Imprime el costo actual.
        print(f'Target cost: {cost}')

    # Se da el formato 0-255 al resultado.
    adversarial /= 2
    adversarial += .5
    adversarial *= 255

    # Se imprime el resultado de la imagen de salida.
    # plt.imshow(adversarial[0].astype(np.uint8))
    # plt.show()

    # Se guarda la imagen en la carpeta de salida.

    im = Image.fromarray(adversarial[0].astype(np.uint8))
    im.save(output_image_path)


def test(input_image_path, batch_size=1):
    inception_v3 = InceptionV3()

    # Se obtiene y redimensiona la imagen de entrada.
    x = image.img_to_array(image.load_img(input_image_path, target_size=(299, 299)))

    # InceptionV3 necesita un formato de imágen de intensidad -1, 1.
    # Se reescala el formato de imagen de entrada.
    x /= 255
    x -= .5
    x *= 2

    # Se agrega en batch-size a la matriz x.

    x = x.reshape([batch_size, x.shape[0], x.shape[1], x.shape[2]])

    # Se realiza la predicción utilizando el modelo InceptionV3.
    y = inception_v3.predict(x)

    # Se obtiene el resultado y se imprime en pantalla.
    result = decode_predictions(y)[0][0]
    print(f'{result[1]}: ' + '{:.2%}'.format(result[2]))


def main():
    train('cat.jpg', 'output/output.png', 951) # Guardar en png (Recomendado).

    # Clasifica la imagen utilizando el modelo InceptionV3.
    # test('output.png')


if __name__ == '__main__':
    main()
