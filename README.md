<h1 align="center">Augmentation Strategy Optimization for Language Understanding 🐗</h1>

<p align="center">Generating adversarial examples by stacking mutiple augmentation methods automatically.</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#setup">Setup</a> •
  <a href="#main-usage">Main Usage</a> •
  <a href="#other-usage">Other Usage</a> •
  <a href="#design">Design</a> 
</p>
  
## About

Augmentation Strategy Optimization for Language Understanding is a Python framework for adversarial attacks, data augmentation, and model training in NLP.
Stacked data augmentation (SDA) is a Python framework for stacking different augmentation methods automatically with reinforcement learning.

## Setup

### Installation for Code

You should be running Python 3.6.13 to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed.

SDA can install irectly from GitHub. For a basic install, run:

```bash
git clone https://github.com/BigPigKing/Adversarial_Data_Boost.git
cd Adversarial_Data_Boost
pip3 install -r requirements.txt
```

### Installation for Prelimilary Directory

```bash
rsync -avh -e 'ssh -p 12030' god@140.113.170.36:/home/god/lab/Adversarial_Data_Boost/data .  # Installation for dataset
rsync -avh -e 'ssh -p 12030' god@140.113.170.36:/home/god/lab/Intergration/model_record .  # Installation for model recording dataset
```

## Main Usage

The procedure for training with SDA from scratch can be divided into four steps.

### 1. Selection of Target Dataset

In SDA, there are totally six dataset can be selected for testing, which are **SST-2, SST-5, MPQA, TREC-6, CR, and SUBJ**.

To start with the specific dataset, just load the model config json file provided in **model_configs** directory.

``` bash
rm model_config.json
cp model_configs/sst2_model_config.json model_config.json
```

### 2. Training the Baseline Text-model from Scratch

To commit an adversarial training, we need to first train an text model from scratch with clean data.

``` bash
python3 sst_complete.py
```

Waiting until the training is finished, if the training process going to the training of REINFORCE. Press **Ctrl^C** for cancling it.

Otherwise the training of generator and discriminator will be intereact for **one time**.

### 3. Record the Baseline Model for Different Hyperparameter Setting

It is essential to retain the parameter of original clean model, thus the comparasion can be coducted easily

```bash
cd model_record
cp -r text_model_weights test_bed  # Retain the original clean model
```

### 4. Going to the *model_config.json* and Change the selected model to ***1***

``` bash
vim model_config.json
```
And you will get the *modelconfig.json* like the below.

![](https://i.imgur.com/5dEyp8D.png)

### 5. Running the *run.sh* to Attack and Defense for Multiple Times

``` bash
./run.sh
```

The times can be choose by change the number of seq. (30 in the figure)
![](https://i.imgur.com/LkVyUg6.png)

### 6. Clean Log file and Get Original Model

Once you finish the experiments, one can use *./clean.sh* to back to the original text model which *training only using clean dataset without adversarial training.*

```bash
./clean.sh
```

## Other Usage
There are also many different function is supported in SDA including ***Visualization, TextAttack, TextAugment and ModelLoading***.

### Visualization
One can check the training process of adversarial training using ***Tensorboard***

``` bash
tensorboard --logdir=runs --samples_per_plugin=text=100
```
![](https://i.imgur.com/NRC2klc.png)
![](https://i.imgur.com/bKlwMRW.png)
![](https://i.imgur.com/XOYbrDj.png)

Or check the **log.txt** for more detailed information

``` bash
cat log.txt
```
![](https://i.imgur.com/puLN74W.png)

### TextAttack
SDA also provides the training process of the other adversarial training methods. It can be done by leverageing the power of **TextAttack Module**

``` bash
pip3 install textattack  # it should already installed in previous steps
./attack.sh
```

Running the attack.sh, and it will automatically running three different AT methods, ***DWB, PWWS, and TextBugger***.

If you want to change differentt target model, just change the model card in ***--target-model***

![](https://i.imgur.com/W8lJw14.png)

And the detailed of different AT method is provided in below:

<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th><strong>Attack Recipe Name</strong></th>
<th><strong>Goal Function</strong></th>
<th><strong>ConstraintsEnforced</strong></th>
<th><strong>Transformation</strong></th>
<th><strong>Search Method</strong></th>
<th><strong>Main Idea</strong></th>
</tr>
</thead>
<tbody>
  <tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on classification tasks, like sentiment classification and entailment:<br></strong></td></tr>

<tr>
<td><code>bae</code> <span class="citation" data-cites="garg2020bae"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>BERT Masked Token Prediction</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>BERT masked language model transformation attack from (["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019)](https://arxiv.org/abs/2004.01970)). </td>
</tr>

<tr>
<td><code>deepwordbug</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>Levenshtein edit distance</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)</sub></td>
</tr>
<tr>
<td> <code>fast-alzantot</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Modified, faster version of the Alzantot et al. genetic algorithm, from (["Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)](https://arxiv.org/abs/1909.00986))</sub></td>
</tr>
<tr>
<td><code>hotflip</code> (word swap) <span class="citation" data-cites="Ebrahimi2017HotFlipWA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Word Embedding Cosine Similarity, Part-of-speech match, Number of words perturbed</sub></td>
<td><sub>Gradient-Based Word Swap</sub></td>
<td><sub>Beam search</sub></td>
<td ><sub> (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751))</sub></td>
</tr>
<tr>
<td><code>iga</code> <span class="citation" data-cites="iga-wang2019natural"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Improved genetic algorithm -based word substitution from (["Natural Language Adversarial Attacks and Defenses in Word Level (Wang et al., 2019)"](https://arxiv.org/abs/1909.06723)</sub></td>
</tr>
<tr>
<td><code>input-reduction</code> <span class="citation" data-cites="feng2018pathologies"></span></td>
<td><sub>Input Reduction</sub></td>
<td></td>
<td><sub>Word deletion</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking , Reducing the input while maintaining the prediction through word importance ranking (["Pathologies of Neural Models Make Interpretation Difficult" (Feng et al., 2018)](https://arxiv.org/pdf/1804.07781.pdf))</sub></td>
</tr>
<tr>
<td><code>kuleshov</code> <span class="citation" data-cites="Kuleshov2018AdversarialEF"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Thought vector encoding cosine similarity, Language model similarity probability</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy word swap</sub></td>
<td ><sub>(["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)) </sub></td>
</tr>
<tr>
<td><code>pso</code> <span class="citation" data-cites="pso-zang-etal-2020-word"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>HowNet Word Swap</sub></td>
<td><sub>Particle Swarm Optimization</sub></td>
<td ><sub>(["Word-level Textual Adversarial Attacking as Combinatorial Optimization" (Zang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.540/)) </sub></td>
</tr>
<tr>
<td><code>pwws</code> <span class="citation" data-cites="pwws-ren-etal-2019-generating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>WordNet-based synonym swap</sub></td>
<td><sub>Greedy-WIR (saliency)</sub></td>
<td ><sub>Greedy attack with word importance ranking based on word saliency and synonym swap scores (["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/))</sub> </td>
</tr>
<tr>
<td><code>textbugger</code> : (black-box) <span class="citation" data-cites="Li2019TextBuggerGA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).</sub></td>
</tr>
<tr>
<td><code>textfooler</code> <span class="citation" data-cites="Jin2019TextFooler"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Word Embedding Distance, Part-of-speech match, USE sentence encoding cosine similarity</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking  (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))</sub> </td>
</tr>

</tbody>
</font>
</table>

### TextAugment

SDA also provides the function to augment the target dataset automatically using ten different augmentation methods.

Including SEDA, EDA, Word Embedding, Clare, Checklist, Charswap, BackTranslation (De, Zh, Ru), and Spelling.

```bash
./make_noisy_to_all.sh
```
![](https://i.imgur.com/5CScJ2G.png)

If you want to change the specific hyperparameter for different augmentation methods.

```bash
vim make_noisy.sh
```
![](https://i.imgur.com/kGe4YKH.png)

And change the hyperparameter to the favor one.

### ModelLoading

SDA will store all the model weight in ***model_record***. 
![](https://i.imgur.com/eXBWICR.png)

For the weights of differen AT methods, it will stored in ***outputs***
![](https://i.imgur.com/CZGvav0.png)
![](https://i.imgur.com/AZVOvJU.png)

And all the detailed can be accessed.

## Citing SDA

If you use Augmentation Strategy Optimization for Language Understanding for your research, please cite

```bibtex
```

