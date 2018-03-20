# When Sheep Shop: Measuring Herding Effects in Product Ratings with Natural Experiments

This repository contains all the code used to produce the Figures and Tables for the article [**When Sheep Shop: Measuring Herding Effects in Product Ratings with Natural Experiments**](https://arxiv.org/abs/1802.06578v2). You are free to use any of the code or use the data after requesting it (have a look at [this file](https://github.com/epfl-dlab/when_sheep_shop/blob/master/data/README.md) for the request) at the condition that our article is explicitly mentionned, see section [Authorization and Reference](#authorization-and-reference). For the moment, the article is only available on arXiv for viewing, however it has been published at WWW2018.

You can also find a blog article preceeding this research on the website of the laboratory DLAB at EPFL, [here](https://dlab.epfl.ch/2017-08-30-of-sheep-and-beer/).

If you have any questions about the code, the data or the results in our article, feel free to contact the authors: Gael Lederrey [gael.lederrey@epfl.ch](mailto:gael.lederrey@epfl.ch) and Robert West [robert.west@epfl.ch](mailto:robert.west@epfl.ch).

## Content of the repository

This repository contains all the figures used in this article in the folder [`figures`](https://github.com/epfl-dlab/when_sheep_shop/tree/master/figures). You can find the iPython notebooks that have been used to generate these Figures as well as the Tables in the folder [`code/notebooks`](https://github.com/epfl-dlab/when_sheep_shop/tree/master/code/notebooks). In the folder [`code/python`](https://github.com/epfl-dlab/when_sheep_shop/tree/master/code/python), you can find some functions that have been used. 

We present now all the iPython notebooks in the folder [`code/notebooks`](https://github.com/epfl-dlab/when_sheep_shop/tree/master/code/notebooks):
- [`1-matching.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/1-matching.ipynb): This notebook is used to match the data from the parsed data (see [this file](https://github.com/epfl-dlab/when_sheep_shop/blob/master/data/README.md) for more information about the data). This process takes a lot of time, therefore the matched data are also available upon request.
- [`2-ratings.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/2-ratings.ipynb): This notebook is used to plot Figure [3(a)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/ratings_all_beers.pdf) in the article.
- [`3-avg_and_std_per_year.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/3-avg_and_std_per_year.ipynb): This notebook is used to plot Figures [3(b)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/avg_rating_per_year.pdf) and [3(c)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/std_rating_per_year.pdf) in the article.
- [`4-zscores.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/4-zscores.ipynb): This notebook is used to plot Figures [4(a)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/zscore_matched_beers.pdf) and [4(b)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/hexhist_zscores_example.pdf) in the article.
- [`5-lost_rhino.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/5-lost_rhino.ipynb): This notebook is used to plot Figures [1(a)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/timeseries_zscore_example.pdf) and [1(b)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/timeseries_avg_zscore_example.pdf) in the article.
- [`6-herding.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/timeseries_avg_zscore_example.pdf): This notebook is used to plot Figures [6(a)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/herding_extreme_global.pdf), [6(b)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/herding_medium_global.pdf), and [6(c)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/lta_herding_global.pdf) in the article.
- [`7-sanity_checks.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/7-sanity_checks.ipynb): This notebook is used to plot Figures [5(a)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/boxplots_ratings.pdf), [5(b)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/boxplots_nbr_ratings.pdf), and [5(c)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/boxplots_nbr_beers_breweries.pdf) in the article.
- [`8-tables.ipynb`](https://github.com/epfl-dlab/when_sheep_shop/blob/master/code/notebooks/8-tables.ipynb): This notebook gives all the Tables (1 to 6) that you can find in the article.

In addition, we also give Figures [2(a)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/bayes_net_exp.pdf), [2(b)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/bayes_net_naive.pdf), [2(c)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/bayes_net_good_nat_exp.pdf), and [2(d)](https://github.com/epfl-dlab/when_sheep_shop/blob/master/figures/bayes_net_bad_nat_exp.pdf) that were done on illustrator. 

## Authorization and Reference
> *We authorize the use of these datasets and any part of the code at the condition that our article is explicitly mentionned using the following reference:* **G. Lederrey and R. West (2018). When Sheep Shop: Measuring Herding Effects in Product Ratings with Natural Experiments. arXiv preprint arXiv:1802.06578v2**. *The article is available for viewing on [arXiv](https://arxiv.org/abs/1802.06578v2).*

