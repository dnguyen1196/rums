Repository for the paper `Efficient and Accurate Learning of Mixtures of Plackett-Luce Models' (AAAI '23). This repo includes the implementation of the spectral EM algorithm and various other algorithms for learning mixtures of Plackett-Luce models from ranking data. 

## The algorithms

**Algorithms for learning mixtures of PL.** The main algorithm can be found in `rums/mixtures/spectral_em_pl.py`. It uses the iterative Luce spectral ranking of Maystre and Grossglauser (2016) which is implemented in `rums/pl/iterative_luce.py` and might be of independent interest. Other mixture learning algorithms include EM-GMM, EM-CML, EMM and a Bayesian algorithm.

**Algorithms for learning a single PL model**. These can be found in `rums/pl/` and include iterative Luce spectral ranking, the MM algorithm, accelerated spectral ranking of Agarwal et al (2018).

## To get the datasets

Create a folder called `datasets` within `rums/data` and download the appropriate datasets.


## Reference

If you find the code here useful, please consider citing the following papers for the corresponding algorithm.

Spectral EM algorithm of Nguyen and Zhang (2023).

> @article{nguyen2023efficient,
>  title={Efficient and Accurate Learning of Mixtures of Plackett-Luce Models},
>  author={Nguyen, Duc and Zhang, Anderson Y},
>  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
>  year={2023}
> }

> @article{maystre2015fast,
>  title={Fast and accurate inference of Plackett--Luce models},
>  author={Maystre, Lucas and Grossglauser, Matthias},
>  journal={Advances in neural information processing systems},
>  volume={28},
>  year={2015}
> }

EM-CML algorithm of Liu et al. (2022).

> @article{zhao2022learning,
>  title={Learning mixtures of random utility models with features from incomplete preferences},
>  author={Zhao, Zhibing and Liu, Ao and Xia, Lirong},
>  journal={arXiv preprint arXiv:2006.03869},
>  year={2022}
> }

EM-GMM algorithm of Liu et al. (2016).

> @inproceedings{zhao2016learning,
>  title={Learning mixtures of Plackett-Luce models},
>  author={Zhao, Zhibing and Piech, Peter and Xia, Lirong},
>  booktitle={International Conference on Machine Learning},
>  pages={2906--2914},
>  year={2016},
>  organization={PMLR}
> }

Bayesian algorithm for learning mixtures of Plackett-Luce.

> @article{mollica2017bayesian,
>  title={Bayesian Plackett--Luce mixture models for partially ranked data},
>  author={Mollica, Cristina and Tardella, Luca},
>  journal={Psychometrika},
>  volume={82},
>  number={2},
>  pages={442--458},
>  year={2017},
>  publisher={Springer}
> }