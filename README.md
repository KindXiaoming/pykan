<img width="600" alt="kan_plot" src="https://github.com/KindXiaoming/pykan/assets/23551623/a2d2d225-b4d2-4c1e-823e-bc45c7ea96f9">

# Kolmogorov-Arnold Networks (KANs)

This is the github repo for the paper ["KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2404.19756). Find the documentation [here](https://kindxiaoming.github.io/pykan/). Here's [author's note](https://github.com/KindXiaoming/pykan/blob/master/README.md#authors-note) responding to current hype of KANs.

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of both model **accuracy** and **interpretability**. A quick intro of KANs [here](https://kindxiaoming.github.io/pykan/intro.html).

<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

## Accuracy
**KANs have faster scaling than MLPs. KANs have better accuracy than MLPs with fewer parameters.**

**Example 1: fitting symbolic formulas**
<img width="1824" alt="Screenshot 2024-04-30 at 10 55 30" src="https://github.com/KindXiaoming/pykan/assets/23551623/e1fc3dcc-c1f6-49d5-b58e-79ff7b98a49b">

**Example 2: fitting special functions**
<img width="1544" alt="Screenshot 2024-04-30 at 11 07 20" src="https://github.com/KindXiaoming/pykan/assets/23551623/b2124337-cabf-4e00-9690-938e84058a91">

**Example 3: PDE solving**
<img width="1665" alt="Screenshot 2024-04-30 at 10 57 25" src="https://github.com/KindXiaoming/pykan/assets/23551623/5da94412-c409-45d1-9a60-9086e11d6ccc">

**Example 4: avoid catastrophic forgetting**
<img width="1652" alt="Screenshot 2024-04-30 at 11 04 36" src="https://github.com/KindXiaoming/pykan/assets/23551623/57d81de6-7cff-4e55-b8f9-c4768ace2c77">

## Interpretability
**KANs can be intuitively visualized. KANs offer interpretability and interactivity that MLPs cannot provide. We can use KANs to potentially discover new scientific laws.**

**Example 1: Symbolic formulas**
<img width="1510" alt="Screenshot 2024-04-30 at 11 04 56" src="https://github.com/KindXiaoming/pykan/assets/23551623/3cfd1ca2-cd3e-4396-845e-ef8f3a7c55ef">

**Example 2: Discovering mathematical laws of knots**
<img width="1443" alt="Screenshot 2024-04-30 at 11 05 25" src="https://github.com/KindXiaoming/pykan/assets/23551623/80451ac2-c5fd-45b9-89a7-1637ba8145af">

**Example 3: Discovering physical laws of Anderson localization**
<img width="1295" alt="Screenshot 2024-04-30 at 11 05 53" src="https://github.com/KindXiaoming/pykan/assets/23551623/8ee507a0-d194-44a9-8837-15d7f5984301">

**Example 4: Training of a three-layer KAN**

![kan_training_low_res](https://github.com/KindXiaoming/pykan/assets/23551623/e9f215c7-a393-46b9-8528-c906878f015e)



## Installation
There are two ways to install pykan, through pypi or github.

**Installation via github**

```python
git clone https://github.com/KindXiaoming/pykan.git
cd pykan
pip install -e .
```

**Installation via pypi**

```python
pip install pykan
```

Requirements

```python
# python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
```

To install requirements:
```python
pip install -r requirements.txt
```

## Computation requirements

Examples in [tutorials](tutorials) are runnable on a single CPU typically less than 10 minutes. All examples in the paper are runnable on a single CPU in less than one day. Training KANs for PDE is the most expensive and may take hours to days on a single CPU. We use CPUs to train our models because we carried out parameter sweeps (both for MLPs and KANs) to obtain Pareto Frontiers. There are thousands of small models which is why we use CPUs rather than GPUs. Admittedly, our problem scales are smaller than typical machine learning tasks, but are typical for science-related tasks. In case the scale of your task is large, it is advisable to use GPUs.

## Documentation
The documentation can be found [here](https://kindxiaoming.github.io/pykan/).

## Tutorials

**Quickstart**

Get started with [hellokan.ipynb](./hellokan.ipynb) notebook.

**More demos**

More Notebook tutorials can be found in [tutorials](tutorials).

## Citation
```python
@misc{liu2024kan,
      title={KAN: Kolmogorov-Arnold Networks}, 
      author={Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Soljačić and Thomas Y. Hou and Max Tegmark},
      year={2024},
      eprint={2404.19756},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
If you have any questions, please contact zmliu@mit.edu

## Author's note
I would like to thank everyone who's interested in KANs. When I designed KANs and wrote codes, I have math & physics examples (which are quite small scale!) in mind, so did not consider much optimization in efficiency or reusability. It's so honored to receive this unwarranted attention, which is way beyond my expectation. So I accept any criticism from people complaning about the efficiency and resuability of the codes, my apology. My only hope is that you find `model.plot()` fun to play with :).

For users who are interested in scientific discoveries and scientific computing (the orginal users intended for), I'm happy to hear your applications and collaborate. This repo will continue remaining mostly for this purpose, probably without signifiant updates for efficiency. In fact, there are already implmentations like [efficientkan](https://github.com/Blealtan/efficient-kan) or [fouierkan](https://github.com/GistNoesis/FourierKAN/) that look promising for improving efficiency.

For users who are machine learning focus, I have to be honest that KANs are likely not a simple plug-in that can be used out-of-the box (yet). Hyperparameters need tuning, and more tricks special to your applications should be introduced. For example, [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) suggests that KANs should better be used in latent space (need embedding and unembedding linear layers after inputs and before outputs). [KANRL](https://github.com/riiswa/kanrl) suggests that some trainable parameters should better be fixed in reinforcement learning to increase training stability.

The most common question I've been asked lately is whether KANs will be next-gen LLMs. I don't have good intuition about this. KANs are designed for applications where one cares about high accuracy and/or interpretability. We do care about LLM interpretability for sure, but interpretability can mean wildly different things for LLM and for science. Do we care about high accuracy for LLMs? I don't know, scaling laws seem to imply so, but probably not too high precision. Also, accuracy can also mean different things for LLM and for science. This subtlety makes it hard to directly transfer conclusions in our paper to LLMs, or machine learning tasks in general. However, I would be very happy if you have enjoyed the high-level idea (learnable activation functions on edges, or interacting with AI for scientific discoveries), which is not necessariy *the future*, but can hopefully inspire and impact *many possible futures*. As a physicist, the message I want to convey is less of "KANs are great", but more of "try thinking of current architectures critically and seeking fundamentally different alternatives that can do fun and/or useful stuff".

I would like to welcome people to be critical of KANs, but also to be critical of critiques as well. Practice is the only criterion for testing understanding (实践是检验真理的唯一标准). We don't know many things beforehand until they are really tried and shown to be succeeding or failing. As much as I'm willing to see success mode of KANs, I'm equally curious about failure modes of KANs, to better understand the boundaries. KANs and MLPs cannot replace each other (as far as I can tell); they each have advantages in some settings and limitations in others. I would be intrigued by a theoretical framework that encompasses both and could even suggest new alternatives (physicists love unified theories, sorry :).

