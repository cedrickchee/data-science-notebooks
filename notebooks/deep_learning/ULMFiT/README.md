# ULMFiT Language Model for Malay Language

State-of-the-Art Language Modeling and text classification in Malay language with perplexity of **29.30245** on Malay Wikipedia and **77.5% accuracy** on [Malaya sentiment analysis](https://github.com/DevconX/Malaya/wiki/Models-Comparison#sentiment_analysis).

## The Project

This directory contains relevant files for implementating ULMFiT language model by Jeremy Howard (fast.ai) and Sebastian Ruder applied to NLP tasks for the Malay language using the [Malay Wikipedia](https://ms.wikipedia.org) corpus.

### Background

_This work is part of my project while studying [fast.ai's 2018 edition of Cutting Edge Deep Learning for Coders, Part 2](http://course.fast.ai/part2.html) course._

I took this opportunity to implement [Universal Language Model Fine-tuning for Text Classification (ULMFiT) paper](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html) in different languages together with the fast.ai community. fast.ai will soon launch a [model zoo with pre-trained language models for many languages](http://forums.fast.ai/t/language-model-zoo-gorilla/14623). You can learn more about ULMFiT in lesson 4 and lesson 10. What I learned from lesson 10 ([my notes](https://cedrickchee.gitbook.io/knowledge/courses/fast.ai/deep-learning-part-2-cutting-edge-deep-learning-for-coders/2018-edition/lesson-10-transfer-learning-nlp)) is:
- how pre-training a full language model from scratch can greatly surpass previous approaches based on simple word vectors
- transfer learning for NLP by using this language model to show a new state of the art result in text classification, in some sense like [NLP's ImageNet moment has arrived](http://ruder.io/nlp-imagenet/)

---

## Malay Language Modeling

The goal of this project is to train Malay word embeddings using the fast.ai version of [AWD-LSTM Language Model](https://arxiv.org/abs/1708.02182)—basically LSTM with dropouts—with data from [Wikipedia](https://dumps.wikimedia.org/mswiki/20180901/mswiki-20180901-pages-articles.xml.bz2) (last updated Sept 2, 2018). The AWD-LSTM language model achieved the state of the art performance on the English language.

A state-of-the-art language modeling with perplexity of 29.30245 on Malay Wikipedia has been achieved compared to state-of-the-art as of June 12, 2018 at 40.68 for English WikiText-2 by [Yang et al (2017)](https://arxiv.org/abs/1711.03953) and 29.2 for English WikiText-103 by [Rae et al (2018)](https://arxiv.org/abs/1803.10049). Lower perplexity means better performance. Obviously, the perplexity of the language model on Malay Wikipedia can't be compared with both mentioned papers due to completely different dataset, but as reference, I hope it can be still useful. To the best of my knowledge, there is no comparable research in Malay language at the point of writing (Sept 21, 2018).

My workflow is as follows:
- Perform 90/10 train-validation split
- Vocabulary size of 60,002 and embeddings at 400 dimensions
- Minimal text cleaning and tokenization using our own tokenizer
- Train language model
- Evaluate model based on perplexity and eyeballing
- Get embeddings of dataset from train set

See [`malay_language_model.ipynb`](https://nbviewer.jupyter.org/github/cedrickchee/data-science-notebooks/blob/master/notebooks/deep_learning/ULMFiT/malay_language_model.ipynb) notebook for more details.

For the community to reuse the model directly, I am contributing the Jupyter notebook (and code) together with the pre-trained weights to the [fast.ai model zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623).

~~Due to some challenges to find curated and publicly available dataset for Malay text, I can't provide a benchmark for text classification yet, but as soon as I can find one (please contact me if you have one), I will update my research.~~

The language model can also be used to extract text features for other downstream tasks such as text classification and speech recognition.

## Text Classification

Since there is no other comparable Malay language model, we need to create a downstream task and compare its accuracy. A text classification was chosen for this purpose, but it is a big challenge to find curated or publicly available dataset for Malay text. Nevertheless, a small curated Malay dataset was found eventually. It is [Malaya, the NLP for bahasa Malaysia](https://github.com/DevconX/Malaya) dataset created by [DevCon Community](https://www.devcon.my/). It contains 277,225 words from [various online sources](https://github.com/DevconX/Malaya/tree/master/crawl) and Malaysia websites. The corpus has 2 categories:

- Positive polarity
- Negative polarity

### Benchmark

We will compare the performance and result of various models for sentiment analysis task.

Source: [models comparison for Malaya sentiment analysis](https://github.com/DevconX/Malaya/wiki/Models-Comparison#sentiment_analysis)

Model              | Metric        | Value
------------------ | ------------- | ---------
[Multinomial][1]   | Accuracy      | 0.73
[XGBoost][2]       | Accuracy      | 0.71
[Bahdanau][3]      | Accuracy      | 0.66
[Bidirectional][4] | Accuracy      | 0.69
[Luong][5]         | Accuracy      | 0.64
[Hierarchical][6]  | Accuracy      | 0.70
[fastText][7]      | Accuracy      | 0.71
[**ULMFiT**][8]    | Accuracy      | **0.77**

[1]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/multinomial-split.ipynb
[2]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/xgb-split.ipynb
[3]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/bahdanau-split.ipynb
[4]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/bidirectional-split.ipynb
[5]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/luong-split.ipynb
[6]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/hierarchical-split.ipynb
[7]: https://nbviewer.jupyter.org/github/DevconX/Malaya/blob/master/session/sentiment/fast-text-split.ipynb
[8]: https://nbviewer.jupyter.org/github/cedrickchee/data-science-notebooks/blob/master/notebooks/deep_learning/ULMFiT/malay_text_classification.ipynb

It shows that text classification using ULMFiT outperforms other algorithms using classical machine learning or other neural network models.

## Dependencies

- Python 3+ (tested with 3.6.5)
- PyTorch 0.4+ (tested with 0.4.0)
- fast.ai 0.7.0 ([conda installation from source](https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652))

## Version History

### v0.1

- Pretrained language model based on Malay Wikipedia with the perplexity of 29.30245.

### v0.2

- Text classification implementation using [Malaya](https://github.com/DevconX/Malaya) [dataset](https://github.com/DevconX/Malaya/wiki/Dataset).
- Text classification (sentiment analysis) benchmark of 77.5% accuracy [compared to 73% by Malaya for 2-label classification (positive or negative)](https://github.com/DevconX/Malaya/wiki/Models-Comparison#sentiment_analysis).

## Pre-trained model

You can download the files from [Google Drive](https://drive.google.com/drive/folders/1p5fsrD97iRD-Vz6C_ae5fo4c5wY0KrJd?usp=sharing):
- Weights for the pre-trained model (lm_malay_final.h5.tar.gz)
    - Uncompress and put the weights (.h5 file) into `{project_root}/data/models/`.
- Index-to-word mapping [itos (index -> string) mapping] (itos.pkl.tar.gz)
    - Uncompress and put the pickled objects (.pkl files) into `{project_root}/data/model/malay/tmp/`.
- Pre-processed training dataset of Malay Wikipedia:
    - tokenized training text data (tok_trn.npy.tar.gz)
    - tokenized validation text data (tok_val.npy.tar.gz)
    - indexed representation of train set (trn_ids.npy.tar.gz)
    - indexed representation of validation set (val_ids.npy.tar.gz)
    - Uncompress and put the numpy array binary (.npy files) into `{project_root}/data/model/malay/tmp/`.

The weights (model state dict) and the optimizer state for the model were saved at the end of the training.

_Note: the model was last trained on 2018-09-22 and the weights last updated on 2018-09-22._

## Inference

### Test 1: Generate text using the language model

Generate sentences using some random strings. Examples:
- "Menara Petronas"
  - menara petronas. pada tahun 2005, sebuah syarikat yang dikenali sebagai Petronas, yang dimiliki oleh Petronas, telah membeli sebuah syarikat yang dimiliki oleh Petronas, Petronas. pada tahun 2008, Petronas membeli saham mereka daripada Petronas, dan Petronas, dan Petronas, dan Petronas, dan Petronas, dan Petronas, dan Petronas, yang telah membeli saham mereka pada tahun 2008.
- "Dr Mahathir merupakan Perdana Menteri"
  - dr mahathir merupakan perdana menteri malaysia yang pertama. beliau juga merupakan seorang ahli parlimen bagi kawasan DUN N53. beliau juga merupakan ahli parlimen bagi kawasan DUN N53. beliau juga merupakan ahli parlimen bagi kawasan DUN N53. beliau juga merupakan ahli parlimen bagi kawasan DUN N53. beliau juga merupakan ahli parlimen bagi kawasan DUN N53. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, negeri sembilan. beliau juga
- "Tunku ialah Bapa Kemerdekaan"
  - tunku ialah bapa kemerdekaan malaysia. beliau juga merupakan seorang ahli politik yang pernah berkhidmat sebagai perdana menteri malaysia ke-6. beliau juga merupakan seorang ahli parlimen bagi kawasan bukit besi, malaysia. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, pulau pinang. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, pulau pinang. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, pulau pinang. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, pulau
- "Syarikat penerbangan"
  - syarikat penerbangan ini. pada tahun 2005, sebuah syarikat penerbangan antarabangsa, BAE systems, telah ditubuhkan. pada tahun 2005, syarikat penerbangan ini telah melancarkan operasi untuk melancarkan pesawat pejuang generasi baru, "ATR 72". pada tahun 2009, syarikat penerbangan ini telah melancarkan operasi untuk melancarkan pesawat pejuang generasi baru, "ATR 72". pada tahun 2014, syarikat penerbangan ini telah melancarkan operasi untuk melancarkan pesawat pejuang generasi
- "Durian ialah buah"
  - durian ialah buah - buahan yang ditanam di dalam hutan.
- "P Ramlee ialah seorang"
  - p ramlee ialah seorang ahli perniagaan yang berjaya. beliau juga merupakan seorang ahli perniagaan yang berjaya. beliau juga merupakan seorang ahli perniagaan yang berjaya. beliau juga merupakan seorang ahli perniagaan dan ahli perniagaan. beliau juga merupakan seorang ahli perniagaan dan ahli perniagaan. beliau juga merupakan seorang ahli perniagaan dan ahli perniagaan. beliau juga merupakan seorang ahli perniagaan dan ahli perniagaan. beliau juga merupakan seorang ahli perniagaan dan ahli perniagaan. beliau juga merupakan seorang ahli perniagaan dan
- "Pemenang badminton Piala Thomas"
  - pemenang badminton piala thomas, PBB. pada tahun 2005, persatuan bola sepak malaysia (FAM) mengumumkan bahawa FAM akan menubuhkan persatuan bola sepak malaysia (FAM). pada tahun 2009, FAM mengumumkan bahawa FAM akan menyertai AFF pada tahun 2013. FAM juga telah mencadangkan bahawa FAM akan menyertai AFF pada tahun 2013. FAM juga telah bersetuju untuk menyertai AFF pada tahun 2014.
- "Lee Chong Wei dan badminton"
  - lee chong wei dan badminton. pada tahun 2005, sebuah lagi acara sukan diadakan di stadium nasional bukit jalil, kuala lumpur. pada tahun 2009, persatuan bola sepak malaysia (FAM) telah mengadakan perlawanan persahabatan menentang (FAM), (FAM), (FAM), (FAM), (FAM), (FAM), (FAM), AFC dan FIFA. pada tahun 2009, FAM mengumumkan bahawa mereka akan bertanding dalam piala malaysia pada tahun
- "Jurulatih Rashid Sidek"
  - jurulatih rashid sidek, seorang ahli politik yang telah berkhidmat sebagai perdana menteri malaysia. beliau juga merupakan seorang ahli parlimen bagi kawasan DUN N53. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, negeri sembilan. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, negeri sembilan. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, negeri sembilan. beliau juga merupakan ahli parlimen bagi kawasan bukit gelugor, negeri sembilan. beliau juga merupakan
- "Pokok getah"
  - pokok getah. pada tahun 2011, sebuah lagi projek baru yang dikenali sebagai "the new york times" telah dilancarkan. pada tahun 2011, sebuah lagi projek baru yang dikenali sebagai "the new york times" telah dilancarkan. pada tahun 2013, sebuah siri "the last airbender" telah diterbitkan semula sebagai siri "the last airbender". pada tahun 2013, "the new york times" melaporkan bahawa "the new york
- "Industri kelapa sawit di Malaysia"
  - industri kelapa sawit di malaysia. pada tahun 2005, sebuah syarikat yang dikenali sebagai Petronas, telah ditubuhkan untuk membangunkan dan memajukan Petronas sebagai sebuah syarikat yang bertanggungjawab untuk pembangunan dan pembangunan teknologi maklumat. Petronas telah ditubuhkan pada tahun 1990, dan pada tahun 1992, Petronas telah menjadi syarikat yang pertama untuk membangunkan dan menghasilkan produk - produk yang berkualiti. Petronas telah membeli dan membeli sebuah syarikat yang dikenali sebagai Petronas Petronas,
- "Penyelidikan minyak sawit"
  - penyelidikan minyak sawit, dan juga beberapa jenis produk yang berkaitan dengan industri.
- "Negara terbesar di Asia Tenggara ialah"
  - negara terbesar di asia tenggara ialah UNESCO. sejarah. pada tahun 2011, sebuah lagi muzium sejarah di indonesia, iaitu muzium negara indonesia, telah dibuka di jakarta, indonesia. muzium ini telah dirasmikan oleh YAB perdana menteri malaysia, tun dr. mahathir bin mohamad pada 1 jun 2007. muzium ini mempamerkan koleksi seni bina yang unik dan menarik. muzium ini mempamerkan koleksi seni bina yang unik dan menarik. muzium ini mempamerkan koleksi seni bina
- "Proton Saga adalah"
  - proton saga adalah sebuah kereta kebal utama yang digunakan oleh tentera udara diraja malaysia.
- "Penyanyi terkenal"
  - penyanyi terkenal, penyanyi, penyanyi, penulis lagu, komposer, komposer, komposer, komposer, dan artis. lagu. lagu ini digubah oleh komposer terkenal, komposer terkenal, komposer terkenal, ahmad nawab. lagu ini digubah oleh ahmad nawab, dan liriknya ditulis oleh ahmad nawab. lagu ini digubah oleh ahmad nawab, dan liriknya ditulis oleh ahmad nawab. lagu ini digubah oleh ahmad nawab, dan dinyanyikan oleh p. ramlee.

## TODO

- [x] Download and extract Malay Wikipedia corpus
- [x] Process text (clean and tokenize text)
- [x] Create train and validation set
- [x] Create data loader for training
- [x] Numericalize the text
- [x] AWD-LSTM model setup
- [x] Train model
- [x] Tune hyper-paramters
- [x] Evaluate language model
- [x] Bug fixes
  - [x] Figure out why the model state that's being reset before every inference is remembering the previous generated sentences
- [x] Fine-tune language model for text classification task
- [x] Build model for text classification
- [x] Find curated or publicly available labelled dataset for Malay corpus
- [ ] ~~Create my own dataset by curating and labelling Malay text scrapped from news sites~~
- [x] Benchmark model for text classification
- [ ] ~~Use continuous cache pointer (from here: https://github.com/salesforce/awd-lstm-lm)~~
- [ ] ~~Try QRNN (from here: https://github.com/salesforce/pytorch-qrnn/)~~
- [x] Identify new datasets for sentiment analysis
- [ ] Share text classification pre-trained model and weights

## License

This repository contains a variety of content; some developed by Cedric Chee, and some from third-parties. The third-party content is distributed under the license provided by those parties.

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer.*

The content developed by Cedric Chee is distributed under the following license:

### Code

The code in this repository, including all code samples in the notebooks listed above, is released under the [MIT license](../../../LICENSE). Read more at the [Open Source Initiative](https://opensource.org/licenses/MIT).

### Text

The text content of the book is released under the CC-BY-NC-ND license. Read more at [Creative Commons](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode).