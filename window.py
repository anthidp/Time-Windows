class WindowGenerator:

  """ Create time windows for single/multi output & single/multi step predictions.
  The width of the window is the number of time steps. 
  
  window windth = label + input windth 

  Parameters
  ----------

  window_windth : window windth
  label_windth : label windth
  batch_size : number of timeseries samples in each batch
  shift : number of input elements to shift between the start of each window

  """

  def __init__(self, window_windth, label_windth, batch_size, shift=1 ):
    
    self.window_windth = window_windth
    self.label_windth = label_windth
    self.batch_size = batch_size
    self.shift = shift

    assert self.window_windth > self.label_windth,  "Label windth cannot exceed window windth."
 

  def make_dataset(self, series, label_columns = None):

    """ Split dataset to windows. Return a tensorflow BatchDataset. 
    
    Parameters
    ----------

    series : input dataset
    label_columns : features for output prediction
    
    """

    self.series = series
    self.label_columns = label_columns
    
    dataset = tf.data.Dataset.from_tensor_slices(self.series)
    dataset = dataset.window(self.window_windth, self.shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(self.window_windth))

    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in 
                                    enumerate(label_columns)}
      self.column_indices = {name: i for i, name in
                             enumerate(self.series.columns)}
      label = [self.column_indices[name] for name in self.label_columns]
      dataset = dataset.map(lambda window: (window[:-self.label_windth, :], tf.gather(window[-label_windth:], label, axis=1) ))

    else:
      dataset = dataset.map(lambda window: (window[:-self.label_windth, :], window[-self.label_windth:, :] ))

    dataset = dataset.batch(self.batch_size)

    return dataset
