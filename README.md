*Group 3 split up, but this portion of the project fairly belongs to all 3 members since we all worked on it evenly.

# Company name: Clarity Inc.

“Clarity Incorporated shines light on the truth behind your data.”

# Team members:

• Ziad Abou-Ghali

• Matthew Barrera

## Kaggle Competition:

You can view the Kaggle competition page using this [link](https://www.kaggle.com/competitions/contradictory-my-dear-watson).

## Learning method:

To be determined after more research and experimentation.

## Proposal:

The problem that we at Clarity Inc. are trying to solve as a group is about Natural Languages. The goal is to detect when two sentences are related or not across 15 different languages utilizing 3 labels: entailment, neutral, or contradiction. This problem is interesting because this sort of language processing can help immensely when it comes to grammar checking, context identification, and fact checking/fake news detection. It’s also fascinating how a computer can take a context-based activity like language and identify the hidden semantics.

We will be analyzing existing papers that analyze sentence relations as a central focus for their algorithm. We will look for algorithms applied in the past on similar problems to what we are trying to achieve and reason at what is necessary for us to include. The Kaggle competition provides training and evaluation data sets in 15 languages, so we will utilize the given resources. Since the algorithm will be one involving NLP (Natural Language Processing), we will research one relevant to the question at hand. Mainly, we want to adapt an algorithm to be an effective sentence relation classifier across the 15 different languages.

We are going to be evaluating our results quantitatively, by seeing how many sentences we are able to classify correctly and providing accuracy scores for our predictions with the use of the test data set. We will also explore plotting the results across the different languages using a bar chart.

The languages included in the competition's data sets are:
```
Arabic, Bulgarian, Chinese, English, French, German, Greek, Hindi, Russian, Spanish, Swahili, Thai, Turkish, Urdu and Vietnamese.
```

**Utilizing JAX to Analyze the Human Language
Proposed By
Matthew Barrera, Ziad Abou-Ghali_**
 
   
 
 
 
 
 
 
 
  ## Abstract
Natural Language Processing (NLP) is a branch in machine learning that is utilized to solve complex real-world problems. Due to its ability to analyze the human language, NLP’s can recognize concepts like synonyms, irony, and sentence relationships. For our project, we focused on sentence relationships. The problem we aimed to solve is as follows: given two sentences, which of the three potential relationships can be concluded: entailment, contradiction, or no relation? Our solution to this problem is to use Transformers and the Bidirectional Encoder Representations from Transformers (BERT) algorithm to train a learning model to recognize sentence structure patterns. A transformer consists primarily of an encoder and a decoder. The encoder takes in input data and ciphers it through encoder layers in order to give a predicted output. The decoder then produces output probabilities based on the encoder's relevant data. Rather than utilizing left-to-right or right-to-left sequence parsing, the Transformer uses the BERT algorithm to achieve bidirectionality; constructing a model that recognizes context of the input in any direction. Given a set of known relations between a hypothesis and premise, the learning model will then be able to decide patterns in other sentence pairs. In preparation for algorithm usage, we used a pretrained model that recognized different languages and a tokenizer that turned each hypothesis-premise sentence pair into an array of numbers.  Once all of the calculations were finished, a prediction value of 0, 1, or 2 would be computed representing entailment, neutrality, and contradiction respectively. Our code was able to achieve valid predictions given a training set of data. Although integrating Jax into our program proved quite troublesome, we were able to use it when finalizing our predictions.
 
 
 
## Introduction
Categorizing sentence relationships is a sensitive yet critical aspect of some professional fields. Such an important task can’t be left to suffer from the potential for human error. Algorithms have evolved drastically, especially in artificial intelligence techniques such as Natural Language Processing (NLP). This branch focuses on human language and utilizes automation and manipulation to train a computer to think like a human. Given an input, a trained computer can be capable of doing many things like problem solving, risk management, and even telling funny jokes. After analyzing previously used NLP algorithms and manipulations in existing pre-trained language models, the most effective one found is the BERT algorithm proposed by Google AI. The model is based on transformers which include encoders for the input and decoders to give an output. BERT is highly efficient against the other algorithms that we will discuss later because it is able to recognize the context of all the words in the sentence. BERT operates using bidirectionality. This means that no matter which direction the model is trained to parse on the sequence given, it is able to achieve correlation with much faster processing time and less hardware constriction than the other algorithms. In the coming sections of this report, the following points will be discussed:
     1. The use of transformers and other techniques that preceded BERT that makes BERT the superior technique as a solution to the “Contradictory, My Dear Watson” Kaggle problem;
    2. How BERT works and what are the steps that allow it to be efficient in its nature of pattern recognition and overall execution;
    3. What type of data is being used and how is the data partitioned in order for the model to operate smoothly;
    4. The JAX approach to the problem rather than the direct use of numpy and pandas and what different implementations were utilized that others didn’t; and,
    5. The key results and main take away from the BERT algorithm and changes could be done and it can benefit sectors in the professional space that ultimately can further automate.
 
## Related Work
The origins of the BERT algorithm is rooted in previous works such as semi-supervised learning, generative pre-training, and ELMo (Embeddings from Language Models). Semi-supervised learning is the medium between unsupervised and supervised learning. Knowing that we have a lot of data available on the world wide web to serve our purpose for these algorithms, it is very time consuming and memory costly in order to label the data to make it accessible for the learning model. Thus, working on unlabeled data by clustering and also classifying the dataset with a small input of label data gives us semi-supervised learning. This was a core starting point for BERT because it gave a concise way to generalize the algorithm for large amounts of data. Generative Pre-Training or GPT is where BERT runs a close relationship with as GPT’s take in sequence as input and output text that correlates to problems such as question answering, language translation, and other complex problems in automation. Just like BERT, GPT runs as a transformer model with an encoder and decoder architecture, but the only drawbacks are in its long term memory across multiple trials and some biases according to papers. ELMo also gives a relation to BERT as it takes the word from input and converts them to numbers in order to make operations. ELMo is context based, the model senses the words around it and given the task at hand, can return back the same word in different meanings in correlation to the sentence. Overall, all of these methods are the precursor to the BERT algorithm as some involve transformers which is the core to how the algorithm works.
 
 
 
 
## Data

The main type of data we are working with in this project is string data. To elaborate, the training data and testing data shares five common attributes split up into five columns: the id, the premise, the hypothesis, the language abbreviation, and the language. For all intents and purposes, the language abbreviation and language represent the same value, but it should be noted that the data contains both. Also, the training data contains a sixth attribute that is not present in the testing data: a label. This represents a known relationship between the premise and hypothesis that will be later used to train a learning model. The data is compiled from the internet spanning 15 languages and was given to us via the Kaggle competition. There are two major inputs to this project: the premise and the hypothesis. Using these, we can draw a parallel to discrete mathematics as implications or conditional statements. In total there are around 5,200 test rows of data and 12,000 rows of training data, each row containing all of its partitioned sections fully filled. No filtering was required in the data as the training file was given with a label on the data and the testing file removed the labeling so that the algorithm can be deemed as a semi-supervised model.
 
## Methods

After research of methods that related to transformers and algorithms of sequence contextualization and manipulation, the BERT transformer model is the most optimal solution to the Contradictory, My Dear Watson problem. First, research was conducted on what a transformer was and how it worked and where it came from. Recurrent Neural Networks (RNN) gives us the foundation, where a neural network learns to give an output based on a previous layer of input. Imagine we were given an incomplete sentence and wanted to predict the next word, we would train the model to recognize certain tokens and the context of the words in order for it to get to the next word and produce a prediction. This was the idea for our problem, we wanted to find a correlation between sentences and this was giving a sense of hope to understand deeply how contextualization given a sequence can occur. This then led to our research discovery of  Transformers and what an attention mechanism was. Known as a sequence to sequence model, the encoder of the architecture takes in the input sequence and creates a context and inserts it in a vector. Then the decoder is initialized in tandem to the encoded context vector and it then creates an output, for example the next word in a sentence. But, it came to our realization that there is a drawback to the RNN being applied and the transformer associated with it; it can’t remember long sequence input and this would cause an exploding gradient. Thus, according to (Cho et al 2014), the encoder-decoder architecture “degrades rapidly as the length of the input sentence increases”. This gave us doubt that the attention mechanism without tweaking would give a problem with the input as some of the premises / hypotheses had long sentences.  We discovered that the BERT algorithm used bidirectionality, which although is costly to memory, is very effective because the context on both sides of the sentence would be intact and not lost in translation as unidirectional models (left-to-right or right--to-left). Aslo, the model reads the entire sentence at once rather than word by word, which cuts down on possible errors in the model. After realizing the BERT algorithm was the efficient model for the problem, we began the code construction and as well as the data inputting. First, we downloaded the data, specifically the training and testing csv files. Then, we took a look at the pretrained model and a tokenizer must be downloaded with it in order to give relevance to the words by assigning an array number for each word. The BERT model used three kinds of input data which were input ID’s, the input masks, and the input type ID. Along with this generic tokenization, it also takes 15% of the words in each sequence and replaces it with [MASK] token. This gives a chance for the model to predict the value of the masked words in conjunction with the context of the non masked words. Also, the loss function associated with BERT is considerate of the prediction of masked values and ignores the non-masked word values. More specific tokens such as the [CLS] and [SEP] token were inputted to determine the beginning of the input and as well as the separation between the premise and the hypothesis. Finally, the determination of whether or not the sentences entail, neural, or contradict is computed by probability values.

![**The following figure shows the convergence of the Masked method vs. the Left-to-Right method. This goes to show how accurate bidirectionality is even though it is costly on the memory, it is able to keep track of the sequence and keep nothing out of context.
**
](https://mino-park7.github.io/images/2019/02/%EA%B7%B8%EB%A6%BC8-ablation-result3.png)

**The following figure shows the convergence of the Masked method vs. the Left-to-Right method. This goes to show how accurate bidirectionality is even though it is costly on the memory, it is able to keep track of the sequence and keep nothing out of context.**
 
## Experiments

Since we could not actually retrieve other algorithms than BERT in order to test solutions to our problem, we however do have results of similar NLP problems and why the other techniques are problematic. We initially noticed that the hyperparameters were going to change the same no matter what fine tuning was applied to BERT to apply it to other NLP problems. Firstly, as revolutionary RNN’s are, they experience the “Long term dependency Problem''. This means that when a given sequence is long enough, the model will forget the context overall. Thus, we knew that if we gave an RNN input of the whole sentence, it was going to take it out of context. It is also good to note that RNN on their own can’t take sentences but only word by word or letter by letter or character by character. The bidirectionality and contextualization is superior in the BERT models and that is why they were created in the first place. Another disadvantage we saw with Sequence to Sequence models in the encoder-decoder architecture is that it was only allowing a fixed length context vector, thus it was not able to thrive with longer sentence inputs. This is faulty because models should be able to withstand inputs at scale, and not be constricted to fixed character sizes, thus giving another superiority with the BERT model. Aside from the mentioned advantages, BERT with enough fine tuning can be applied to a plethora of NLP problems, is scalable to over 100 languages, and since it has been exhausted on a larger corpus in training, it has no problem with smaller sized input. Some drawbacks with the BERT model are reasons outside of the scope of this problems such as the model being too large because of the amount of training needed, it’s slight inability to directly apply to standalone programs thus requires fine tuning, and must be used with pre training models rather than new ones because of the time consumption. The support for this claim is in the following figure: 
![Fine Tuning vs. From Scratch](https://github.com/[username]/[reponame]/blob/[branch]/fineTuningVsFineTuningFromScratch.png?raw=true)

## Conclusion
The BERT algorithm has shown a tremendous amount of success across many NLP complex problems and showcased it with the Contradictory, My Dear Watson Problem. BERT’s transformer bidirectionality feature was able to create such a groundbreaking method of recognition of all words given an input sequence that was superior to all other networks trying to accomplish similar tasks. Although BERT does require fine tuning and doesn’t respond well to newly trained models, it does the job on the critical NLP questions and future problems to come. We have learned that there is a way to systematically organize data, and be able to work from first principles in order to train a computer to do what humans have been doing in the professional space for years. In retrospect, the BERT algorithm is just the start to complex problem solving and is a gateway to the NLP branch of artificial intelligence for further innovation. 
 
## Citations

Bert NLP model explained for complete beginners. ProjectPro. (n.d.). Retrieved April 24, 2022, from https://www.projectpro.io/article/bert-nlp-model-explained/558#:~:text=Disadvantages%20of%20BERT,-Most%20of%20the&text=They%20include%3A,It%20is%20expensive. 
Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014, September 3). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv.org. Retrieved April 24, 2022, from https://arxiv.org/abs/1406.1078 
Why Bert fails in commercial environments. KDnuggets. (n.d.). Retrieved April 24, 2022, from https://www.kdnuggets.com/2020/03/bert-fails-commercial-environments.html 
 
