# xStream

**Python** implementation of the xStream algorithm proposed by the paper:

https://cmuxstream.github.io/

* xStream detects outliers in feature-evolving data streams, where the full feature-space is unknown a-priori and evolves over time.

* xStream is accurate in all three settings: (i) static data, (ii) row-streams, and (iii) feature-evolving streams, as demonstrated over multiple datasets in each setting.

## Demonstration

**Evolving:** We've set up a [Google Colab notebook](https://colab.research.google.com/drive/1riknnKQm9S5lkroI4SrsqkvkCLd-Bilr?usp=sharing), which fits and scores the evolving datapoints as described in the paper.

**Row-streaming:** We've also set up an additional [Google Colab notebook](https://colab.research.google.com/drive/1S14lEizH_gsN_E0knZvPwEOk3KsILcm_?usp=sharing), which fits and scores the datapoints in a stream manner (not evolving data points). Hence, the number of feature is known in advance.

## Authors

* Ariel Ramos
* Maya Awada
* Nehmat Touma
