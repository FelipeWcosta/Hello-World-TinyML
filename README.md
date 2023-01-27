# Hello World TinyML por Pete Warden e Daniel Situnayake 
## Resolvendo as ddependências
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

Onde `SAMPLES = 1000` é o número de amostras e `SEED = 1337` seta os mesmos números randômicos toda vez que for executado novamente podendo ser utilizado qualquer valor para `SEED`. Agora adiciona-se ruído na senóide:
```
    y_values += 0.1*np.random.randn(*y_values.shape)
    plt.plot(x_values, y_values, 'b.')
    plt.show()
```
![Noise][def]

[def]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/noise.png