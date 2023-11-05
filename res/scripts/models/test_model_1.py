# Model Function
def apple_elixir_model(input_shape, output_shape, seed=SEED):
  tf.random.set_seed(seed)

  input_layer = tfkl.Input(shape=input_shape, name='Input')

  x = tfkl.Conv2D(filters=32, kernel_size=3,padding='same',activation='relu',name='CN0000')(input_layer)
  x = tfkl.Conv2D(filters=32, kernel_size=3,padding='same',activation='relu',name='CN0001')(x)
  x = tfkl.MaxPooling2D(name='Mp0')(x)
  x = tfkl.Conv2D(filters=64, kernel_size=3,padding='same',activation='relu',name='CN0010')(x)
  x = tfkl.Conv2D(filters=64, kernel_size=3,padding='same',activation='relu',name='CN0011')(x)
  # x = tfkl.MaxPooling2D(name='Mp1')(x)
  # x = tfkl.Conv2D(filters=128, kernel_size=3,padding='same',activation='relu',name='CN0100')(x)
  # x = tfkl.Conv2D(filters=128, kernel_size=3,padding='same',activation='relu',name='CN0101')(x)
  x = tfkl.MaxPooling2D(name='Mp2')(x)
  x = tfkl.Conv2D(filters=256, kernel_size=3,padding='same',activation='relu',name='CN0110')(x)
  x = tfkl.Conv2D(filters=256, kernel_size=3,padding='same',activation='relu',name='CN0111')(x)

  x = tfkl.GlobalAveragePooling2D(name='gap')(x)
  x = tfkl.Dense(units = 10, activation='relu')(x)
  x = tfkl.Dense(units = 10, activation='relu')(x)
  
  output_layer = tfkl.Dense(units=output_shape ,activation='softmax', name='Output')(x)

  model = tfk.Model(inputs=input_layer, outputs=output_layer, name='Convnet')
  model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(weight_decay=5e-4), metrics=['accuracy'])

  return model