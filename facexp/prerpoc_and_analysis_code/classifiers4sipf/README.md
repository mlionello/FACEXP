# An interpretable, efficient, and accurate framework for facial expression recognition in the wild.

M. Lionello, E. Bucci, E. Sampaolo, L. Cecchetti

IMT Lucca - IMT Lucca - Lucca
Social and Affective Neuroscience (SANe) group, MoMiLab - IMT School for Advanced

### Introduction:
Facial expressions are immediate means for communicating emotions. Facial
expression recognition in the wild (FER) is crucial for social, commercial, and scientific tasks. Deep
Learning models (DNN) excel in FER but generally lack self-explanatory features. Here, we propose
a framework relying on pairwise distances between face landmarks, which we validate using three
datasets.
### Methods:
Three video datasets are used: RAVDESS[1], including different modalities for
each emotion, PEDFE[2], which comprises human ratings for genuine and posed expressions
(hit-rates), and FACEXP, a dataset with multiple repetitions of the same emotion for each actor. To
model facial displacement, we detect face landmarks with Medusa[3] and extract 20 principal
components from pairwise distances. ONE-NN models and k-fold cross-validation are used to
assess overall performance. In RAVDESS and FACEXP, we also perform leave-1-subject-out, while
in PEDFE, we test varying training sizes and hit-rate levels. Analyses are restricted to Ekman's 6
basic emotions. Code available at https://github.com/mlionello/FACEXP.
### Results:
Overall, in
RAVDESS, the model yields 87.2% accuracy. In PEDFE, 70.0% for posed, 62.4% for genuine, and
67.0% for mixed expressions. In FACEXP, 93.1% with two repetitions and 95.6% with all repetitions.
In both RAVDESS and FACEXP, accuracy drops to ~35% when facial expressions of one
participant are predicted from those of all the other actors. In RAVDESS, the combination of
modalities does not affect performance (p>0.05). In FACEXP, reintroducing into the training set,
one repetition from the test-set actor increases the classifier performance up to 90% (min. 60%). In
PEDFE, including low hit-rates has marginal impact on classification of genuine expressions
(decreasing from 77.1% to 62.7%). The classification of mixed and genuine expressions stabilizes at
1139 and 524 training-samples (72.6% and 67.6% respectively), except for Sadness and Fear when
modelling the mixed dataset.
### Discussion: 
In summary, the performance of the proposed model
matches top DNN solutions. However, leave-1-subject-out reveals high variability between-subjects
and proves that successful classification is driven by the presence of samples from the validated
subject into the training-set. Moreover, it suggests that facial expressions are highly consistent
within-subject. Posed expressions show more canonical learning patterns than genuine ones. Lastly,
the analysis of hit-rates indicates that, in some challenging cases, our model extracts meaningful
features from facial displacement more accurately than humans.
### Conclusion:
The model we propose
is interpretable, efficient, and performs similarly to DNN in FER. However, generalizability of the
prediction to independent participants still represents an open challenge.
### References: 
Livingstone SR, Russo FA. The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of
facial and vocal expressions in North American English. PLOS ONE (2018). 13(5)

Miolla, A., Cardaioli, M., Scarpazza, C. Padova Emotional Dataset of FacialExpressions (PEDFE): A unique dataset of genuine and posed emotional
facial expressions. Behav Res. (2022)

Snoek, L., Jack, R., Schyns, P. Dynamic face imaging: a novel analysis
framework for 4D social face perception and expression. IEEE 17th
International Conference on Automatic Face and Gesture Recognition (FG)
(2023). 1-4.
