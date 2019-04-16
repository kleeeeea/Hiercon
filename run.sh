!/bin/bash
# declare an array called array and define 3 vales
# array=( one two three )
# for i in "${array[@]}"
# do
#     echo $i
# done

export DATASET='pubmed'
# physmath

# concept mining
## candidate generation preprocessing
python preprocess/preprocess.py $DATASET
python candidate_generation/to_json/autophrase.py $DATASET &
python candidate_generation/to_json/spacy_extract.py $DATASET &
python candidate_generation/to_json/nltk_extract.py $DATASET &
wait
echo 'done candidate_generation'
python candidate_generation/merge_span.py $DATASET

## embedding
python econ/embedding.py $DATASET

# generate document representation
# then wait for some time...
python econ/scoring_feature_generation.py $DATASET &
python document_representation/feasibility_feature.py $DATASET
python document_representation/importance_feature.py $DATASET

# manually generate dataless labels, and manually enable evaluation file

# evaluation

python Hierarchical-attention-networks-pytorch/train.py $DATASET
