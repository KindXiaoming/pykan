Example 14: Knot supervised
===========================

.. code:: ipython3

    import pandas as pd
    import numpy as np
    import torch
    from kan import *
    import copy
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
    df = pd.read_csv("./knot_data.csv")
    df.keys()
    
    X = df[df.keys()[1:-1]].to_numpy()
    Y = df[['signature']].to_numpy()
    
    # normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean[np.newaxis,:])/X_std[np.newaxis,:]
    input_normalier = [X_mean, X_std]
    
    # normalize Y
    max_signature = np.max(Y)
    min_signature = np.min(Y)
    Y = ((Y-min_signature)/2).astype(int)
    n_class = int((max_signature-min_signature)/2+1)
    output_normalier = [min_signature, 2]
    
    dataset = {}
    num = X.shape[0]
    n_feature = X.shape[1]
    train_ratio = 0.8
    train_id_ = np.random.choice(num, int(num*train_ratio), replace=False)
    test_id_ = np.array(list(set(range(num))-set(train_id_)))
    
    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(X[train_id_]).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(Y[train_id_][:,0]).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X[test_id_]).type(dtype).to(device)
    dataset['test_label'] = torch.from_numpy(Y[test_id_][:,0]).type(torch.long).to(device)



::


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    /var/folders/6j/b6y80djd4nb5hl73rv3sv8y80000gn/T/ipykernel_75986/3212158569.py in <module>
         11 
         12 # Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
    ---> 13 df = pd.read_csv("./knot_data.csv")
         14 df.keys()
         15 


    ~/opt/anaconda3/lib/python3.9/site-packages/pandas/util/_decorators.py in wrapper(*args, **kwargs)
        309                     stacklevel=stacklevel,
        310                 )
    --> 311             return func(*args, **kwargs)
        312 
        313         return wrapper


    ~/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        676     kwds.update(kwds_defaults)
        677 
    --> 678     return _read(filepath_or_buffer, kwds)
        679 
        680 


    ~/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
        573 
        574     # Create the parser.
    --> 575     parser = TextFileReader(filepath_or_buffer, **kwds)
        576 
        577     if chunksize or iterator:


    ~/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
        930 
        931         self.handles: IOHandles | None = None
    --> 932         self._engine = self._make_engine(f, self.engine)
        933 
        934     def close(self):


    ~/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
       1214             # "Union[str, PathLike[str], ReadCsvBuffer[bytes], ReadCsvBuffer[str]]"
       1215             # , "str", "bool", "Any", "Any", "Any", "Any", "Any"
    -> 1216             self.handles = get_handle(  # type: ignore[call-overload]
       1217                 f,
       1218                 mode,


    ~/opt/anaconda3/lib/python3.9/site-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        784         if ioargs.encoding and "b" not in ioargs.mode:
        785             # Encoding
    --> 786             handle = open(
        787                 handle,
        788                 ioargs.mode,


    FileNotFoundError: [Errno 2] No such file or directory: './knot_data.csv'


.. code:: ipython3

    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())
    
    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())
    
    model = KAN(width=[n_feature,1,n_class], grid=5, k=3, seed=seed, device=device)
    model.fit(dataset, lamb=0.005, batch=1024, loss_fn = nn.CrossEntropyLoss(), metrics=[train_acc, test_acc], display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc']);

.. code:: ipython3

    model.plot(scale=1.0, beta=0.2)
    
    n = 17
    for i in range(n):
        plt.gcf().get_axes()[0].text(1/(2*n)+i/n-0.005,-0.02,df.keys()[1:-1][i], rotation=270, rotation_mode="anchor")

.. code:: ipython3

    scores = model.feature_score
    features = list(df.keys()[1:-1])
    
    y_pos = range(len(features))
    plt.bar(y_pos, scores)
    plt.xticks(y_pos, features, rotation=90);
    plt.ylabel('feature importance')
