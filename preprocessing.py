def scaling(data):
  scaled_data = data.map(lambda x,y: (x/255, y))
  return scaled_data

def spliting(data):
  train_size = int(len(data)*0.7)
  val_size = int(len(data)*0.2)
  test_size = int(len(data)*0.1)

  train = data.take(train_size)
  val = data.skip(train_size).take(val_size)
  test = data.skip(train_size+val_size).take(test_size)

  return train, val, test