# Model Card

## Model Details
Creator: Saoussen Chaabnia

Model Date: April 19, 2022

Model Version: 1.0.0

Model Type: Random Forest Classifier

Citation Detail: Udacity Machine Learning DevOps Project #3

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Intended Use
Intended use: The primary intended use is to determine a salary range based on limited information about them.

Intended user: Toy dataset, not intended to be used in any official capacity.

Out-of-scope uses: Never rely on this to provide an accurate salary for an indiviual.
## Training Data
A 32561 x 15 dataframe with the following columns.

* age
* workclass
* fnlgt
* education
* maritial status
* occupation
* relationship
* race
* sex
* capital-gain
* capital-loss
* hours-per-week
* native-counrty
* salary (target)
## Evaluation Data
The dataset was given by Udacity, it is publicly available Census Bureau data. Categorical data was One Hot Encoded and the target was passed through a Label Binarizer
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision was the primary metric for the model, recall and F-beta scores are also reported.

The scores are as follows:
* Precision: 0.7850877192982456
* Recall: 0.6076035302104549
* F-Beta Score: 0.6850363566781478

## Ethical Considerations
While the data is anonymous it does contain sensitive data about class, occupation, race, country of origin, sex. The predicted output of this model should not be taken with a high degree of confidence without further testing for bias.
## Caveats and Recommendations
More accurate an up-to-date data is needed. 
The dataset is somewhat imbalanced with approximately 25% of labels >50K and 75%% <=650K.