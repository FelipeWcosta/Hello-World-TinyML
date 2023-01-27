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
  Para treinar o `model_1`:
  ```
    history_1 = model_1.fit(x_train, y_train, epochs=1000, batch_size=16,
                            validation_data=(x_validate, y_validate))
  ```
  Pode-se plotar as perdas para os dados de treinamento e validação:
  ```
    loss = history_1.history['loss']
    val_loss = history_1.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'g.', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
  ``` 

  ![Training][def4]

  [def4]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/training.png

  Dando um *zoom* na figura anterior percebe-se que não há uma evolução considerável do nosso modelo a   
  partir da 600ª época além de perdas considerávelmente elevadas:

  ```
    SKIP = 100
    plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
    plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
  ```
  ![Zoom][def5]

  [def5]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/zoom.png

  Pode-se análisar também o erro médio absoluto `MAE`, como segue:
  ```
    mae = history_1.history['mae']
    val_mae = history_1.history['val_mae']
    plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
    plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
  ```

  ![MAE][def6]

  [def6]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/MAE.png

  Onde percebe-se um maior erro para os dados de validação *(azul)* do que para os dados de treinamento 
  *(verde)*, o que pode ser caracterizado como *overfitting*. Para solucionar o problema de 
  *overfitting* da nossa rede neural podemos aumentar a base de dados ou modificar a estrutura da 
  rede, podemos ver graficamente este problema da rede neural plotando os valores atuais e os valores 
  predizidos pela rede.

  ```
  predictions = model_1.predict(x_train)
  plt.clf()
  plt.title('Training data predicted vs actual values')
  plt.plot(x_test, y_test, 'b.', label='Actual')
  plt.plot(x_train, predictions, 'r.', label='Predicted')
  plt.legend()
  plt.show()
  ```  
  ![predicted][def7]

  [def7]: https://github.com/FelipeWcosta/Hello-World-TinyML/blob/main/Figs/predicted.png