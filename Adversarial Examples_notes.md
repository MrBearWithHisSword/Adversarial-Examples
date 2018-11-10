[TOC]

# Adversarial Examples for Evaluating Reading Comprehensio System

Robin Jia.  Percy Liang. 

## **Problem:**

从现有标准度量的角度来看，Reading Comprehension Systems正在取得快速的进步，但是它们是否真的理解了自然语言仍然是一个有待商榷的问题。



## **Goal:**

针对上述问题，这篇文章在SQuAD数据集上提出了一种产生对抗样本来对模型进行对抗性评估的方法，以推进对aversarial attack具有一定鲁棒性的DNN traing的研究。



## **Method:**

SQuAD数据集中的样本是以(P,Q,A)的形式给出的，其中P(paragraph)是一段文本,Q(question)是针对P所提出的问题,而A(answer)是Q的正确答案。（保证A是P的一部分）

这篇文章是通过在P的末尾拼接一段新的句子作为扰动(Perturbation)来产生对抗样本以迷惑模型。

（这里保证添加的Perturbatoin不影响paragraph中包含正确答案的语意信息）

具体流程如下图所示：

![image-20181110134741368](/Users/haochenshi/Library/Application Support/typora-user-images/image-20181110134741368.png)



## **Result：**

作者在16个开源模型上做了测试，结果表明在当时没有任何模型能够有效抵御这种Adversarial Attack: 这些模型的平均F1 score从75%降到了36%.

在CV领域中防御Adversarial Attack的一种较为有效的方式(《Generating Natural Language Adversarial Examples》文中指出这也是当时唯一有效的方式）是进行Adversarial Training: 
​	即在训练的每一轮产生新的Adversarial Examples, 并将之喂给神经网络进行学习。

该方法在CV领域能够work的一大原因是可以通过梯度来快速产生新的adversarial examples. 但是在NLP领域由于adversarial example难以通过基于梯度的方式产生，该方法并不适用。
(虽然可将离散的语言Embedding到feature space中后看作是连续的，但在feature space中对某一点进行扰动所得到的结果往往是难以转换为有意义的语言表示的。)

作者在尝试了将产生的adversarial examples作为data augmentation的一种方式对模型进行训练后，发现模型仅仅学到了忽略paragraph中最后一个句子对answer的影响，并没有对adversarial attack产生足够的防御能力：
一旦更换adversarial examples的生成模式，模型的表现结果立马下降到与原来相仿的水平。

结果如下：

![image-20181110145201513](/Users/haochenshi/Library/Application Support/typora-user-images/image-20181110145201513.png)





# Generating Natural Language Adversarial Examples

Alzantot et al.

这篇文章与上一篇文章的出发点是相通的，都希望可以通过提出adversarial examples的生成方法来推进robust DNN的研究。



这篇文章针对Sentiment Analysis和Textual Entailment两个NLP任务提出了一种基于种群的的优化方法来产生adversarial examples的方法，并证明了其adversarial attack的有效性。

文章的实验部分也做了adversarial training, 实验结果表明在NLP领域adversarial traning难以有效的提升模型对adversarial attack的鲁棒性。



# Analysis

Adversarial Examples: 通过对训练样本进行不影响其语意的扰动，得到的能够有效mislead模型的样本。

Adversarial Training: 将adversarial exampls作为data augmentation来训练模型，以提升模型对adversarial attack的防御能力。



### Point 1:

第一篇文章中添加扰动的主要方式是：通过替换部分命名实体的名称、反义词替换等方式得到与原样本中语意信息<u>***不同***</u> 的一个句子，并将之作为扰动添加到段落末尾。

与CV领域中DNN对部分非语意信息过于敏感不同的是，这篇文章试图说明当前的reading comprehension system对语意信息的改变***<u>不够敏感</u>***，无法有效分辨不同的语意信息。

我认为这种对语意信息的不敏感可能不仅仅是这些模型的问题，也跟当前word embedding的方式有一定关系，目前的word embedding主要是通过Language Modeling任务来得到：
Language Modeling是通过在未标注的大量语料上根据上下文来预测当前位置的单词。（而且考虑到效率的原因，模型通常不会使用过多的上下文来预测当前词汇）
在这种方式下产生的词性相似而具有相反语意的词向量，往往在特征空间中具有较大的相似度。(例如，若上下文为: It tastes \_\_\_\_\_\_\_\_.  那么往往空缺处预测为'good'或'bad'的概率是相仿的。)

为验证这一想法，我使用在百科上使用Skip-Gram with Negative Sampling训练得到的300纬词向量进行了如下测试：

![image-20181110155319687](/Users/haochenshi/Library/Application Support/typora-user-images/image-20181110155319687.png)

可以看到，仅从词向量的余弦相似度和曼哈顿距离上很难区分'好'、'坏'和'很好',甚至'good'和'bad'的相似度达到了0.82.
由于这些具有相反语意的词汇在特征空间中的表示很接近，因此若仅将文本中的good替换为bad,那么文本在特征空间中的表示也会很接近，而若要训练模型能够正确区分文本的语意是positive的还是negative的，就需要模型能够发现并利用这些反义词在特征空间中某些维度上较小的差异，从而模型就会对这些纬度较为敏感，那么模型就会缺乏对这些纬度上的扰动的鲁棒性。



### Point 2：x2

值得注意的一点是，第2篇文章中构造adversarial examples的方式是使用在特征空间中距离较小且语意相近的词汇来代替原词汇作为扰动。即，与第一篇文章恰好相反，第2篇文章在试图说明现有的NLP领域中的模型对与原文语意信息***<u>相同</u>*** 的扰动<u>***十分敏感***</u>。

adversarial training能够在CV领域work, 而在NLP难以work的原因有可能是Natural Language的离散性导致的产生的adversarial example在样本空间采样不足所导致的。

由于自然语言的离散型，我们难以在语意空间中某一点的周围充分采样，从而导致模型的Loss在训练样本处往往都是一些比较sharp的点：

![image-20181110160920737](/Users/haochenshi/Library/Application Support/typora-user-images/image-20181110160920737.png)

如上图所示，突出的尖点是在训练集中出现的样本，模型能够正确处理训练集中出现过的点，却无法有效处理在语意空间中表示相近，却未在训练集中出现过的样本（文章2中的adversarial examples)。

在CV领域能够通过梯度的方式较为方便且较为充足得对语意空间中接近的点进行采样，再将这些点作为data augmentation训练模型来缓解aversarial attack, 然而在NLP领域很难对语意空间中接近的点进行较为充分的采样，使得其在很多点上依旧十分sharp,因此文章1、2中的结果显示adversarial training未能有效提升模型的鲁棒性:

![image-20181110162731436](/Users/haochenshi/Library/Application Support/typora-user-images/image-20181110162731436.png)



我们或许可以通过增加模型的泛化能力来绕过在语意空间中充分采样的难点，但是如point 1中所述的原因，在特征空间中接近的文本或词语，在语意空间中中并不一定十分接近，甚至可能拥有相反的语意。因此若要通过这种方式来尝试解决这个问题的话，首先应在构造word embedding是将语意信息更加充分的利用起来。（或许可以通过在language modeling时使用更多的上下文，或其他方式将更多的语音信息编码到word embedding中)

