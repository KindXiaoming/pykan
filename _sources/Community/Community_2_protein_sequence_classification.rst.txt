Community 2: Protein Sequence Classification
============================================

**Disclaimer: This is uploaded from a github user, not the KAN authors.
KAN authors did not writer this or proofread this carefully, hence are
not responsible for mistakes in this notebook. If you have questions,
please consult the github user who uploaded it. Thank you!**

In this example, we will see how to use KAN in protein sequence
classification. We will be using one hot encoding to encode the amino
acids.

This is just an example how it can be used for protein sequences. Need to use real data to actually observe the performance.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from kan import *
    import torch
    import random
    import numpy as np

.. code:: ipython3

    # Hyperparameters
    PROTEIN_WINDOW_SIZE = 5 
    
    # define the universe of possible input amino acids, ie. vocab list
    aa_list = 'ARNDCQEGHILKMFPSTWYVX'

.. code:: ipython3

    def one_hot_encode(protein_sequence):
        """
        One-hot encodes a protein sequence.
    
        Args:
            protein_sequence (str): The input protein sequence.
    
        Returns:
            numpy.array: The one-hot encoded representation of the protein sequence.
        """
        # Create a dictionary mapping amino acids to indices
        aa_to_index = {aa: i for i, aa in enumerate(aa_list)}
        
        # Initialize an array of zeros with shape (sequence_length, alphabet_length)
        encoding = np.zeros((len(protein_sequence), len(aa_list)))
        
        # Iterate over the protein sequence and set the corresponding index to 1
        for i, aa in enumerate(protein_sequence):
            if aa in aa_to_index:
                encoding[i, aa_to_index[aa]] = 1
            else:
                # If the amino acid is not in the alphabet, set the last index to 1 (unknown)
                encoding[i, -1] = 1
        
        return encoding

.. code:: ipython3

    def generate_sample_protein_dataset(num_samples=20, protein_window_size=5):
        """
        Generate a dataset of protein sequences of length 11, keeping Lysine(K) in the center for label 1 and Serine(S) for label 0. 
    
        Args:
            num_samples (int): Number of samples to generate.
            protein_window_size (int): Length of the protein sequence.
    
        Returns:
            dict: A dictionary containing train_input, test_input, train_label, and test_label.
        """
        
        dataset = {'train_input': [], 'test_input': [], 'train_label': [], 'test_label': []}
        alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    
        # Generate half of the samples with label 1 and half with label 0
        label_sequence = [1] * (num_samples // 2) + [0] * (num_samples // 2)
        random.shuffle(label_sequence)
    
        for label in label_sequence:
            # Generate a protein sequence with 'K' in the middle for label 1 and 'S' for label 0
            if label == 1:
                center_aa = 'K'
            else:
                center_aa = 'S'
            sequence = ''.join(random.choices(alphabet.replace(center_aa, ''), k=protein_window_size//2)) + center_aa + ''.join(random.choices(alphabet.replace(center_aa, ''), k=protein_window_size//2))
            print(sequence, label)
            encoded_sequence = one_hot_encode(sequence).flatten()
    
            # Split the dataset into train and test (50% each)
            if len(dataset['train_input']) < num_samples // 2:
                dataset['train_input'].append(encoded_sequence)
                dataset['train_label'].append(label)
            else:
                dataset['test_input'].append(encoded_sequence)
                dataset['test_label'].append(label)
    
        # Convert lists to tensors
        dataset['train_input'] = torch.tensor(dataset['train_input'])
        dataset['test_input'] = torch.tensor(dataset['test_input'])
        dataset['train_label'] = torch.tensor(dataset['train_label']).view(-1, 1)
        dataset['test_label'] = torch.tensor(dataset['test_label']).view(-1, 1)
    
        return dataset
    
    # Generate dataset with 10 samples
    dataset = generate_sample_protein_dataset(40)


.. parsed-literal::

    GTKYX 1
    TTKPP 1
    AESVY 0
    MYSFD 0
    SQKNT 1
    IDKAC 1
    AXKTA 1
    TESDW 0
    YXSTF 0
    VTSYF 0
    HYKYE 1
    RDSPA 0
    MDSNK 0
    SCKFH 1
    AHKED 1
    EFKYA 1
    EPKLR 1
    GWSRE 0
    GMSYE 0
    IPSKD 0
    NSKQA 1
    TWKNL 1
    TCKFF 1
    HNKSG 1
    QNSKR 0
    RVKYC 1
    TESCP 0
    SMKXE 1
    IYSEV 0
    XQSKD 0
    VKSYN 0
    EESGV 0
    IISMQ 0
    FLKGE 1
    VMKGH 1
    PTKMH 1
    TLSIQ 0
    TTSMA 0
    ATKEE 1
    MGSFT 0


.. code:: ipython3

    print(dataset)


.. parsed-literal::

    {'train_input': tensor([[0., 0., 0.,  ..., 0., 0., 1.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [1., 0., 0.,  ..., 1., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.float64), 'test_input': tensor([[0., 0., 1.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 0., 0., 0.],
            [1., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.float64), 'train_label': tensor([[1],
            [1],
            [0],
            [0],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [1],
            [0],
            [0],
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0]]), 'test_label': tensor([[1],
            [1],
            [1],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [0],
            [0],
            [1],
            [0]])}


.. code:: ipython3

    # define model
    # create a KAN: 105 inputs, 2D output, and 3 hidden neurons. k=2, 3 grid intervals (grid=3).
    # considering window size: 5, 5 times 21(vocab size), input-> 21 * 5
    
    model = KAN(width=[105,3,2], grid=3, k=2)

.. code:: ipython3

    def train_acc():
        return torch.mean((torch.round(model(dataset['train_input'])[:,0]) == dataset['train_label'][:,0]).float())
    
    def test_acc():
        return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label'][:,0]).float())
    
    results = model.train(dataset, opt="LBFGS", steps=5, metrics=(train_acc, test_acc));
    results['train_acc'][-1], results['test_acc'][-1]


.. parsed-literal::

    train loss: 1.04e-03 | test loss: 2.33e-01 | reg: 6.38e+01 : 100%|████| 5/5 [00:15<00:00,  3.00s/it]




.. parsed-literal::

    (1.0, 0.949999988079071)



.. code:: ipython3

    lib = ['x','x^2']
    
    model.auto_symbolic(lib=lib)


.. parsed-literal::

    fixing (0,0,0) with x^2, r2=0.9999999665312771
    fixing (0,0,1) with x^2, r2=0.9999979934036755
    fixing (0,0,2) with x^2, r2=0.9999999622133074
    fixing (0,1,0) with x^2, r2=0.9999999799949156
    fixing (0,1,1) with x^2, r2=0.9991883825579457
    fixing (0,1,2) with x^2, r2=0.9999994895376765
    fixing (0,2,0) with x^2, r2=0.9999990593107048
    fixing (0,2,1) with x^2, r2=0.9999996655563207
    fixing (0,2,2) with x^2, r2=0.999999966951783
    fixing (0,3,0) with x^2, r2=0.0
    fixing (0,3,1) with x^2, r2=0.0
    fixing (0,3,2) with x^2, r2=0.0
    fixing (0,4,0) with x^2, r2=0.0
    fixing (0,4,1) with x^2, r2=0.0
    fixing (0,4,2) with x^2, r2=0.0
    fixing (0,5,0) with x^2, r2=0.9999998808271742
    fixing (0,5,1) with x^2, r2=0.9999998953621121
    fixing (0,5,2) with x^2, r2=0.999999968375537
    fixing (0,6,0) with x^2, r2=0.9981315108075913
    fixing (0,6,1) with x^2, r2=0.999999843899342
    fixing (0,6,2) with x^2, r2=0.9999999589830514
    fixing (0,7,0) with x^2, r2=0.0
    fixing (0,7,1) with x^2, r2=0.0
    fixing (0,7,2) with x^2, r2=0.0
    fixing (0,8,0) with x^2, r2=0.9999998200480685
    fixing (0,8,1) with x^2, r2=0.9999999862277233
    fixing (0,8,2) with x^2, r2=0.9999813684975204
    fixing (0,9,0) with x^2, r2=0.9999999870502827
    fixing (0,9,1) with x^2, r2=0.9997068764841773
    fixing (0,9,2) with x^2, r2=0.9999999768060073
    fixing (0,10,0) with x^2, r2=0.0
    fixing (0,10,1) with x^2, r2=0.0
    fixing (0,10,2) with x^2, r2=0.0
    fixing (0,11,0) with x^2, r2=0.0
    fixing (0,11,1) with x^2, r2=0.0
    fixing (0,11,2) with x^2, r2=0.0
    fixing (0,12,0) with x^2, r2=0.9999996829291468
    fixing (0,12,1) with x^2, r2=0.9999747579126426
    fixing (0,12,2) with x^2, r2=0.999999983307972
    fixing (0,13,0) with x^2, r2=0.9999999625943928
    fixing (0,13,1) with x^2, r2=0.9999999376278957
    fixing (0,13,2) with x^2, r2=0.9999982391574459
    fixing (0,14,0) with x^2, r2=0.9999999540837675
    fixing (0,14,1) with x^2, r2=0.999993702906714
    fixing (0,14,2) with x^2, r2=0.9999996570009488
    fixing (0,15,0) with x^2, r2=0.999994330617256
    fixing (0,15,1) with x^2, r2=0.9999996275829637
    fixing (0,15,2) with x^2, r2=0.9999999847151517
    fixing (0,16,0) with x^2, r2=0.9999999965050976
    fixing (0,16,1) with x^2, r2=0.9999999736671104
    fixing (0,16,2) with x^2, r2=0.9999999930306683
    fixing (0,17,0) with x^2, r2=0.0
    fixing (0,17,1) with x^2, r2=0.0
    fixing (0,17,2) with x^2, r2=0.0
    fixing (0,18,0) with x^2, r2=0.0
    fixing (0,18,1) with x^2, r2=0.0
    fixing (0,18,2) with x^2, r2=0.0
    fixing (0,19,0) with x^2, r2=0.9999999090971862
    fixing (0,19,1) with x^2, r2=0.999999811862135
    fixing (0,19,2) with x^2, r2=0.9999989774097001
    fixing (0,20,0) with x^2, r2=0.9999998410838922
    fixing (0,20,1) with x^2, r2=0.999999954524944
    fixing (0,20,2) with x^2, r2=0.9999995236701958
    fixing (0,21,0) with x^2, r2=0.0
    fixing (0,21,1) with x^2, r2=0.0
    fixing (0,21,2) with x^2, r2=0.0
    fixing (0,22,0) with x^2, r2=0.0
    fixing (0,22,1) with x^2, r2=0.0
    fixing (0,22,2) with x^2, r2=0.0
    fixing (0,23,0) with x^2, r2=0.9999999953439344
    fixing (0,23,1) with x^2, r2=0.9999999811625986
    fixing (0,23,2) with x^2, r2=0.9999999555240675
    fixing (0,24,0) with x^2, r2=0.0
    fixing (0,24,1) with x^2, r2=0.0
    fixing (0,24,2) with x^2, r2=0.0
    fixing (0,25,0) with x^2, r2=0.9999998811160122
    fixing (0,25,1) with x^2, r2=0.9999999304599131
    fixing (0,25,2) with x^2, r2=0.9999998146150727
    fixing (0,26,0) with x^2, r2=0.9999984806067732
    fixing (0,26,1) with x^2, r2=0.9999999378197437
    fixing (0,26,2) with x^2, r2=0.9999994597119173
    fixing (0,27,0) with x^2, r2=0.9999991631417857
    fixing (0,27,1) with x^2, r2=0.9999995673636365
    fixing (0,27,2) with x^2, r2=0.9999999532647686
    fixing (0,28,0) with x^2, r2=0.9999999703007609
    fixing (0,28,1) with x^2, r2=0.999999684803164
    fixing (0,28,2) with x^2, r2=0.9999999512126377
    fixing (0,29,0) with x^2, r2=0.0
    fixing (0,29,1) with x^2, r2=0.0
    fixing (0,29,2) with x^2, r2=0.0
    fixing (0,30,0) with x^2, r2=0.9999999361143834
    fixing (0,30,1) with x^2, r2=0.9999999526237395
    fixing (0,30,2) with x^2, r2=0.9999999758476676
    fixing (0,31,0) with x^2, r2=0.9999999772937739
    fixing (0,31,1) with x^2, r2=0.999998823370015
    fixing (0,31,2) with x^2, r2=0.9999999951682172
    fixing (0,32,0) with x^2, r2=0.9999998454496639
    fixing (0,32,1) with x^2, r2=0.9999902771971996
    fixing (0,32,2) with x^2, r2=0.9993939197671529
    fixing (0,33,0) with x^2, r2=0.9979543880597602
    fixing (0,33,1) with x^2, r2=0.9999999733685552
    fixing (0,33,2) with x^2, r2=0.9999999872961335
    fixing (0,34,0) with x^2, r2=0.0
    fixing (0,34,1) with x^2, r2=0.0
    fixing (0,34,2) with x^2, r2=0.0
    fixing (0,35,0) with x^2, r2=0.0
    fixing (0,35,1) with x^2, r2=0.0
    fixing (0,35,2) with x^2, r2=0.0
    fixing (0,36,0) with x^2, r2=0.9999997063428989
    fixing (0,36,1) with x^2, r2=0.9999999499783073
    fixing (0,36,2) with x^2, r2=0.9999997789665279
    fixing (0,37,0) with x^2, r2=0.9999999009788131
    fixing (0,37,1) with x^2, r2=0.9999999715302882
    fixing (0,37,2) with x^2, r2=0.9999994175010077
    fixing (0,38,0) with x^2, r2=0.9999998691174623
    fixing (0,38,1) with x^2, r2=0.9999932563050576
    fixing (0,38,2) with x^2, r2=0.9999999113693885
    fixing (0,39,0) with x^2, r2=0.9999998298601666
    fixing (0,39,1) with x^2, r2=0.9999889526353061
    fixing (0,39,2) with x^2, r2=0.9999999603098101
    fixing (0,40,0) with x^2, r2=0.9999941430142316
    fixing (0,40,1) with x^2, r2=0.9999907490633038
    fixing (0,40,2) with x^2, r2=0.9999999184598747
    fixing (0,41,0) with x^2, r2=0.0
    fixing (0,41,1) with x^2, r2=0.0
    fixing (0,41,2) with x^2, r2=0.0
    fixing (0,42,0) with x^2, r2=0.0
    fixing (0,42,1) with x^2, r2=0.0
    fixing (0,42,2) with x^2, r2=0.0
    fixing (0,43,0) with x^2, r2=0.0
    fixing (0,43,1) with x^2, r2=0.0
    fixing (0,43,2) with x^2, r2=0.0
    fixing (0,44,0) with x^2, r2=0.0
    fixing (0,44,1) with x^2, r2=0.0
    fixing (0,44,2) with x^2, r2=0.0
    fixing (0,45,0) with x^2, r2=0.0
    fixing (0,45,1) with x^2, r2=0.0
    fixing (0,45,2) with x^2, r2=0.0
    fixing (0,46,0) with x^2, r2=0.0
    fixing (0,46,1) with x^2, r2=0.0
    fixing (0,46,2) with x^2, r2=0.0
    fixing (0,47,0) with x^2, r2=0.0
    fixing (0,47,1) with x^2, r2=0.0
    fixing (0,47,2) with x^2, r2=0.0
    fixing (0,48,0) with x^2, r2=0.0
    fixing (0,48,1) with x^2, r2=0.0
    fixing (0,48,2) with x^2, r2=0.0
    fixing (0,49,0) with x^2, r2=0.0
    fixing (0,49,1) with x^2, r2=0.0
    fixing (0,49,2) with x^2, r2=0.0
    fixing (0,50,0) with x^2, r2=0.0
    fixing (0,50,1) with x^2, r2=0.0
    fixing (0,50,2) with x^2, r2=0.0
    fixing (0,51,0) with x^2, r2=0.0
    fixing (0,51,1) with x^2, r2=0.0
    fixing (0,51,2) with x^2, r2=0.0
    fixing (0,52,0) with x^2, r2=0.0
    fixing (0,52,1) with x^2, r2=0.0
    fixing (0,52,2) with x^2, r2=0.0
    fixing (0,53,0) with x^2, r2=0.9999999987614517
    fixing (0,53,1) with x^2, r2=0.9999999995688087
    fixing (0,53,2) with x^2, r2=0.999999999716506
    fixing (0,54,0) with x^2, r2=0.0
    fixing (0,54,1) with x^2, r2=0.0
    fixing (0,54,2) with x^2, r2=0.0
    fixing (0,55,0) with x^2, r2=0.0
    fixing (0,55,1) with x^2, r2=0.0
    fixing (0,55,2) with x^2, r2=0.0
    fixing (0,56,0) with x^2, r2=0.0
    fixing (0,56,1) with x^2, r2=0.0
    fixing (0,56,2) with x^2, r2=0.0
    fixing (0,57,0) with x^2, r2=0.9999999977865017
    fixing (0,57,1) with x^2, r2=0.999999999143338
    fixing (0,57,2) with x^2, r2=0.9999999998290019
    fixing (0,58,0) with x^2, r2=0.0
    fixing (0,58,1) with x^2, r2=0.0
    fixing (0,58,2) with x^2, r2=0.0
    fixing (0,59,0) with x^2, r2=0.0
    fixing (0,59,1) with x^2, r2=0.0
    fixing (0,59,2) with x^2, r2=0.0
    fixing (0,60,0) with x^2, r2=0.0
    fixing (0,60,1) with x^2, r2=0.0
    fixing (0,60,2) with x^2, r2=0.0
    fixing (0,61,0) with x^2, r2=0.0
    fixing (0,61,1) with x^2, r2=0.0
    fixing (0,61,2) with x^2, r2=0.0
    fixing (0,62,0) with x^2, r2=0.0
    fixing (0,62,1) with x^2, r2=0.0
    fixing (0,62,2) with x^2, r2=0.0
    fixing (0,63,0) with x^2, r2=0.0
    fixing (0,63,1) with x^2, r2=0.0
    fixing (0,63,2) with x^2, r2=0.0
    fixing (0,64,0) with x^2, r2=0.0
    fixing (0,64,1) with x^2, r2=0.0
    fixing (0,64,2) with x^2, r2=0.0
    fixing (0,65,0) with x^2, r2=0.9999999302979558
    fixing (0,65,1) with x^2, r2=0.9999902406071391
    fixing (0,65,2) with x^2, r2=0.9999998684472524
    fixing (0,66,0) with x^2, r2=0.0
    fixing (0,66,1) with x^2, r2=0.0
    fixing (0,66,2) with x^2, r2=0.0
    fixing (0,67,0) with x^2, r2=0.9999999655544946
    fixing (0,67,1) with x^2, r2=0.9999995390688572
    fixing (0,67,2) with x^2, r2=0.9999997366108699
    fixing (0,68,0) with x^2, r2=0.9999999735303753
    fixing (0,68,1) with x^2, r2=0.9999999539372727
    fixing (0,68,2) with x^2, r2=0.9999998409922631
    fixing (0,69,0) with x^2, r2=0.9999999975190795
    fixing (0,69,1) with x^2, r2=0.9999998840699803
    fixing (0,69,2) with x^2, r2=0.9999999748333692
    fixing (0,70,0) with x^2, r2=0.9999999638112955
    fixing (0,70,1) with x^2, r2=0.999999996122007
    fixing (0,70,2) with x^2, r2=0.9999990113519382
    fixing (0,71,0) with x^2, r2=0.0
    fixing (0,71,1) with x^2, r2=0.0
    fixing (0,71,2) with x^2, r2=0.0
    fixing (0,72,0) with x^2, r2=0.9999999782223539
    fixing (0,72,1) with x^2, r2=0.9999996360566132
    fixing (0,72,2) with x^2, r2=0.9999994783563169
    fixing (0,73,0) with x^2, r2=0.0
    fixing (0,73,1) with x^2, r2=0.0
    fixing (0,73,2) with x^2, r2=0.0
    fixing (0,74,0) with x^2, r2=0.9999999430582801
    fixing (0,74,1) with x^2, r2=0.9999999373180665
    fixing (0,74,2) with x^2, r2=0.9999999928808172
    fixing (0,75,0) with x^2, r2=0.9999999675795376
    fixing (0,75,1) with x^2, r2=0.9999999926331626
    fixing (0,75,2) with x^2, r2=0.9999999455360133
    fixing (0,76,0) with x^2, r2=0.9999999894203153
    fixing (0,76,1) with x^2, r2=0.999999852706142
    fixing (0,76,2) with x^2, r2=0.9999994569257162
    fixing (0,77,0) with x^2, r2=0.0
    fixing (0,77,1) with x^2, r2=0.0
    fixing (0,77,2) with x^2, r2=0.0
    fixing (0,78,0) with x^2, r2=0.9999969548814738
    fixing (0,78,1) with x^2, r2=0.999999895396509
    fixing (0,78,2) with x^2, r2=0.9999997624575255
    fixing (0,79,0) with x^2, r2=0.0
    fixing (0,79,1) with x^2, r2=0.0
    fixing (0,79,2) with x^2, r2=0.0
    fixing (0,80,0) with x^2, r2=0.0
    fixing (0,80,1) with x^2, r2=0.0
    fixing (0,80,2) with x^2, r2=0.0
    fixing (0,81,0) with x^2, r2=0.9999999633167932
    fixing (0,81,1) with x^2, r2=0.9999999924423665
    fixing (0,81,2) with x^2, r2=0.9999999407891473
    fixing (0,82,0) with x^2, r2=0.0
    fixing (0,82,1) with x^2, r2=0.0
    fixing (0,82,2) with x^2, r2=0.0
    fixing (0,83,0) with x^2, r2=0.9964873061598577
    fixing (0,83,1) with x^2, r2=0.9999998536697641
    fixing (0,83,2) with x^2, r2=0.9999999474125241
    fixing (0,84,0) with x^2, r2=0.9999999434524759
    fixing (0,84,1) with x^2, r2=0.9999999848500863
    fixing (0,84,2) with x^2, r2=0.9999997362933968
    fixing (0,85,0) with x^2, r2=0.9999784391692933
    fixing (0,85,1) with x^2, r2=0.9999999123872062
    fixing (0,85,2) with x^2, r2=0.9999981066188347
    fixing (0,86,0) with x^2, r2=0.9999999470214042
    fixing (0,86,1) with x^2, r2=0.9999999622653485
    fixing (0,86,2) with x^2, r2=0.9999999256587131
    fixing (0,87,0) with x^2, r2=0.9999838246792585
    fixing (0,87,1) with x^2, r2=0.9999998906573028
    fixing (0,87,2) with x^2, r2=0.9997398325048757
    fixing (0,88,0) with x^2, r2=0.9999903305520499
    fixing (0,88,1) with x^2, r2=0.9999999129937596
    fixing (0,88,2) with x^2, r2=0.9999994338574667
    fixing (0,89,0) with x^2, r2=0.9999999969824458
    fixing (0,89,1) with x^2, r2=0.9999998811902262
    fixing (0,89,2) with x^2, r2=0.9999999955608072
    fixing (0,90,0) with x^2, r2=0.9999999968821633
    fixing (0,90,1) with x^2, r2=0.9999999231999729
    fixing (0,90,2) with x^2, r2=0.999999921201756
    fixing (0,91,0) with x^2, r2=0.9999734544061402
    fixing (0,91,1) with x^2, r2=0.9999966985161072
    fixing (0,91,2) with x^2, r2=0.9999999489971586
    fixing (0,92,0) with x^2, r2=0.9999999864791468
    fixing (0,92,1) with x^2, r2=0.9999999698743414
    fixing (0,92,2) with x^2, r2=0.9998985820640515
    fixing (0,93,0) with x^2, r2=0.0
    fixing (0,93,1) with x^2, r2=0.0
    fixing (0,93,2) with x^2, r2=0.0
    fixing (0,94,0) with x^2, r2=0.9999572021042229
    fixing (0,94,1) with x^2, r2=0.9999999403042822
    fixing (0,94,2) with x^2, r2=0.9999984955483119
    fixing (0,95,0) with x^2, r2=0.0
    fixing (0,95,1) with x^2, r2=0.0
    fixing (0,95,2) with x^2, r2=0.0
    fixing (0,96,0) with x^2, r2=0.0
    fixing (0,96,1) with x^2, r2=0.0
    fixing (0,96,2) with x^2, r2=0.0
    fixing (0,97,0) with x^2, r2=0.9999999855742208
    fixing (0,97,1) with x^2, r2=0.9999990622913814
    fixing (0,97,2) with x^2, r2=0.9999999661558678
    fixing (0,98,0) with x^2, r2=0.9999998924577429
    fixing (0,98,1) with x^2, r2=0.9999999075025128
    fixing (0,98,2) with x^2, r2=0.9999925555905432
    fixing (0,99,0) with x^2, r2=0.0
    fixing (0,99,1) with x^2, r2=0.0
    fixing (0,99,2) with x^2, r2=0.0
    fixing (0,100,0) with x^2, r2=0.9999999888884751
    fixing (0,100,1) with x^2, r2=0.9999999053398424
    fixing (0,100,2) with x^2, r2=0.9999999274642732
    fixing (0,101,0) with x^2, r2=0.0
    fixing (0,101,1) with x^2, r2=0.0
    fixing (0,101,2) with x^2, r2=0.0
    fixing (0,102,0) with x^2, r2=0.0
    fixing (0,102,1) with x^2, r2=0.0
    fixing (0,102,2) with x^2, r2=0.0
    fixing (0,103,0) with x^2, r2=0.9999997998513549
    fixing (0,103,1) with x^2, r2=0.9999999874737161
    fixing (0,103,2) with x^2, r2=0.9999999891891058
    fixing (0,104,0) with x^2, r2=0.0
    fixing (0,104,1) with x^2, r2=0.0
    fixing (0,104,2) with x^2, r2=0.0
    fixing (1,0,0) with x^2, r2=0.9827286380576173
    fixing (1,0,1) with x^2, r2=0.9753307156038028
    fixing (1,1,0) with x^2, r2=0.99206369703365
    fixing (1,1,1) with x^2, r2=0.9950033104451041
    fixing (1,2,0) with x^2, r2=0.9980758555730187
    fixing (1,2,1) with x^2, r2=0.9973139539011773


.. code:: ipython3

    formula1, formula2 = model.symbolic_formula()[0]
    formula1




.. math::

    \displaystyle 0.44 \left(0.02 \left(- x_{1} - 1\right)^{2} + 0.02 \left(x_{10} + 1\right)^{2} + 0.04 \left(- x_{101} - 1\right)^{2} + 0.01 \left(- x_{13} - 1\right)^{2} - 0.02 \left(- x_{14} - 1\right)^{2} - 0.02 \left(- x_{15} - 1\right)^{2} + 0.02 \left(- x_{17} - 1\right)^{2} + 0.03 \left(x_{2} + 1\right)^{2} - 0.01 \left(x_{20} + 1\right)^{2} - 0.01 \left(x_{21} + 1\right)^{2} - 0.03 \left(- x_{24} - 1\right)^{2} + 0.01 \left(- x_{26} - 1\right)^{2} - 0.02 \left(- x_{29} - 1\right)^{2} - 0.02 \left(- x_{31} - 1\right)^{2} + 0.01 \left(x_{32} + 1\right)^{2} + 0.01 \left(- x_{33} - 1\right)^{2} - 0.01 \left(x_{37} + 1\right)^{2} - 0.01 \left(- x_{39} - 1\right)^{2} - 0.01 \left(- x_{40} - 1\right)^{2} - 0.02 \left(- x_{54} - 1\right)^{2} + 0.02 \left(- x_{58} - 1\right)^{2} - 0.01 \left(- x_{6} - 1\right)^{2} - 0.01 \left(- x_{66} - 1\right)^{2} - 0.02 \left(- x_{68} - 1\right)^{2} + 0.02 \left(- x_{69} - 1\right)^{2} - 0.04 \left(x_{70} + 1\right)^{2} + 0.01 \left(- x_{71} - 1\right)^{2} + 0.03 \left(- x_{73} - 1\right)^{2} + 0.01 \left(- x_{75} - 1\right)^{2} + 0.01 \left(- x_{76} - 1\right)^{2} + 0.02 \left(- x_{77} - 1\right)^{2} - 0.01 \left(- x_{82} - 1\right)^{2} - 0.01 \left(- x_{85} - 1\right)^{2} - 0.02 \left(x_{87} + 1\right)^{2} - 0.01 \left(x_{9} + 1\right)^{2} - 0.04 \left(x_{90} + 1\right)^{2} + 0.03 \left(- x_{91} - 1\right)^{2} + 0.02 \left(x_{93} + 1\right)^{2} + 0.03 \left(x_{98} + 1\right)^{2} - 0.01 \left(- x_{99} - 1\right)^{2} - 1\right)^{2} + 0.7 \left(- 0.03 \left(- x_{1} - 1\right)^{2} - 0.02 \left(x_{10} + 1\right)^{2} + 0.02 \left(x_{101} + 1\right)^{2} - 0.03 \left(x_{104} + 1\right)^{2} + 0.05 \left(- x_{13} - 1\right)^{2} + 0.01 \left(- x_{15} - 1\right)^{2} - 0.05 \left(x_{16} + 1\right)^{2} - 0.02 \left(- x_{17} - 1\right)^{2} - 0.01 \left(- x_{2} - 1\right)^{2} + 0.01 \left(- x_{21} - 1\right)^{2} + 0.02 \left(x_{24} + 1\right)^{2} - 0.01 \left(- x_{26} - 1\right)^{2} + 0.01 \left(- x_{27} - 1\right)^{2} - 0.02 \left(- x_{28} - 1\right)^{2} - 0.03 \left(- x_{29} - 1\right)^{2} + 0.03 \left(- x_{3} - 1\right)^{2} + 0.04 \left(- x_{31} - 1\right)^{2} + 0.05 \left(- x_{32} - 1\right)^{2} + 0.03 \left(- x_{34} - 1\right)^{2} - 0.01 \left(- x_{37} - 1\right)^{2} + 0.02 \left(- x_{39} - 1\right)^{2} - 0.03 \left(x_{40} + 1\right)^{2} - 0.02 \left(x_{41} + 1\right)^{2} - 0.07 \left(- x_{54} - 1\right)^{2} + 0.09 \left(- x_{58} - 1\right)^{2} + 0.03 \left(x_{6} + 1\right)^{2} - 0.02 \left(- x_{66} - 1\right)^{2} - 0.01 \left(x_{68} + 1\right)^{2} + 0.02 \left(- x_{69} - 1\right)^{2} - 0.03 \left(x_{7} + 1\right)^{2} + 0.02 \left(x_{70} + 1\right)^{2} - 0.01 \left(x_{73} + 1\right)^{2} + 0.04 \left(x_{75} + 1\right)^{2} + 0.01 \left(x_{76} + 1\right)^{2} - 0.01 \left(x_{79} + 1\right)^{2} + 0.01 \left(- x_{82} - 1\right)^{2} + 0.03 \left(- x_{84} - 1\right)^{2} + 0.01 \left(x_{85} + 1\right)^{2} + 0.02 \left(- x_{87} - 1\right)^{2} + 0.01 \left(x_{89} + 1\right)^{2} + 0.05 \left(- x_{90} - 1\right)^{2} - 0.01 \left(- x_{91} - 1\right)^{2} - 0.03 \left(x_{92} + 1\right)^{2} + 0.01 \left(- x_{95} - 1\right)^{2} + 0.03 \left(- x_{98} - 1\right)^{2} - 1\right)^{2} + 0.17 \left(- 0.01 \left(- x_{1} - 1\right)^{2} + 0.05 \left(- x_{101} - 1\right)^{2} - 0.07 \left(x_{104} + 1\right)^{2} + 0.06 \left(- x_{14} - 1\right)^{2} + 0.01 \left(- x_{15} - 1\right)^{2} + 0.02 \left(- x_{16} - 1\right)^{2} + 0.02 \left(- x_{17} - 1\right)^{2} + 0.02 \left(- x_{20} - 1\right)^{2} - 0.07 \left(- x_{21} - 1\right)^{2} + 0.05 \left(x_{24} + 1\right)^{2} + 0.05 \left(- x_{26} - 1\right)^{2} - 0.06 \left(- x_{27} - 1\right)^{2} - 0.01 \left(- x_{28} - 1\right)^{2} - 0.02 \left(- x_{29} - 1\right)^{2} - 0.02 \left(x_{3} + 1\right)^{2} + 0.06 \left(- x_{31} - 1\right)^{2} + 0.01 \left(- x_{32} - 1\right)^{2} + 0.05 \left(- x_{34} - 1\right)^{2} + 0.06 \left(- x_{37} - 1\right)^{2} + 0.03 \left(- x_{38} - 1\right)^{2} + 0.01 \left(- x_{39} - 1\right)^{2} - 0.13 \left(- x_{54} - 1\right)^{2} + 0.09 \left(- x_{58} - 1\right)^{2} - 0.04 \left(x_{6} + 1\right)^{2} + 0.02 \left(x_{68} + 1\right)^{2} + 0.07 \left(x_{69} + 1\right)^{2} + 0.04 \left(- x_{7} - 1\right)^{2} - 0.02 \left(- x_{70} - 1\right)^{2} + 0.08 \left(- x_{71} - 1\right)^{2} + 0.02 \left(- x_{73} - 1\right)^{2} + 0.03 \left(- x_{75} - 1\right)^{2} - 0.06 \left(- x_{76} - 1\right)^{2} + 0.02 \left(- x_{77} - 1\right)^{2} - 0.04 \left(x_{79} + 1\right)^{2} - 0.08 \left(x_{82} + 1\right)^{2} - 0.04 \left(x_{84} + 1\right)^{2} + 0.06 \left(x_{85} + 1\right)^{2} + 0.05 \left(- x_{86} - 1\right)^{2} + 0.07 \left(- x_{87} - 1\right)^{2} + 0.04 \left(x_{88} + 1\right)^{2} - 0.05 \left(- x_{89} - 1\right)^{2} + 0.12 \left(x_{9} + 1\right)^{2} - 0.02 \left(x_{90} + 1\right)^{2} - 0.02 \left(- x_{91} - 1\right)^{2} - 0.01 \left(- x_{92} - 1\right)^{2} - 0.04 \left(- x_{93} - 1\right)^{2} - 0.06 \left(- x_{95} - 1\right)^{2} + 0.01 \left(x_{98} + 1\right)^{2} - 0.05 \left(- x_{99} - 1\right)^{2} - 1\right)^{2} - 0.57



.. code:: ipython3

    model.plot()



.. image:: Community_2_protein_sequence_classification_files/Community_2_protein_sequence_classification_13_0.png


