# Hello World TinyML por Pete Warden e Daniel Situnayake 
## Resolvendo as dependências
 Primeiramente importamos as bibliotecas e *frameworks* necessários:
 ```
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import math
```
## Gerando os dados
 Iremos gerar os dados para treinamento, validação e teste usando:
 ```    
    SAMPLES = 1000
    SEED = 1337
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
    np.random.shuffle(x_values)
    y_values = np.sin(x_values)
    plt.plot(x_values, y_values, 'b.')
    plt.show()
 ```
 ![FunctSine][def]

[def]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/sine.png

Onde `SAMPLES = 1000` é o número de amostras e `SEED = 1337` seta os mesmos números randômicos toda vez que for executado novamente, podendo ser utilizado qualquer valor para `SEED`. Agora adiciona-se ruído na senóide:
```
    y_values += 0.1*np.random.randn(*y_values.shape)
    plt.plot(x_values, y_values, 'b.')
    plt.show()
```
![Noise][def2]

[def2]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/noise.png

## Separando os dados de treinamento, validação e teste
 Para esta etapa utilizamos 60% da amostra para treinamento, 20% para validação e 20% para testes:
 ```
    TRAIN_SPLIT = int(0.6*SAMPLES)
    TEST_SPLIT = int(0.2*SAMPLES+TRAIN_SPLIT)
    x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])
    assert(x_train.size + x_validate.size + x_test.size) == SAMPLES
    plt.plot(x_train, y_train, 'b.', label='Train')
    plt.plot(x_validate, y_validate, 'y.', label='Validate')
    plt.plot(x_test, y_test, 'r.', label='Test')
    plt.legend()
    plt.show()
 ```
 ![Split][def3]

 [def3]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/split.png

 ## Modelo básico
  Agora criaremos o primeiro modelo, com apenas um *layer* para recebimento dos dados e um de saída:
  ```
    from keras import layers
    model_1 = tf.keras.Sequential()
    model_1.add(layers.Dense(16, activation='relu', input_shape=(1,)))
    model_1.add(layers.Dense(1))
    model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  ```