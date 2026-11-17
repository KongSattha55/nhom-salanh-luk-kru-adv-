# Summary of Evaluation Methods for LLMs

Name : Kong Sattha \
ID: e20210963

------------------------------------------------------------------------

## Topic 1: Exact Match (EM) Evaluation

### 1. Introduction

Exact Match (EM) is an evaluation metric used to measure how often a Large Language Model (LLM) produces an answer that exactly matches the correct reference answer.

It is commonly used in question-answering tasks, especially when the expected answer is short and specific.

### 2. Theory

Exact Match compares the model's predicted answer with the reference answer.

**Formula:** $$EM = \frac{\text{Number of exact matches}}{\text{Total number of questions}} \times 100\%$$

For example, if a model answers 10 questions and 8 answers match exactly, its EM score is $80\%$. A matching answer receives $1$, while a non-matching answer receives $0$.

### 3. Practical Examples

-   **Example 1: Exact match**
    -   **Reference:** Phnom Penh
    -   **Prediction:** Phnom Penh
    -   **Result:** Match ($1$)
-   **Example 2: Different wording**
    -   **Reference:** Phnom Penh
    -   **Prediction:** The capital is Phnom Penh.
    -   **Result:** No exact match ($0$)
    -   *Note:* The second answer is correct in meaning, but its wording is different from the reference.

### 4. Python Implementation

We can use Python to compare predicted answers with correct answers and calculate the percentage of exact matches.

The Python implementation is included in `code/Implementation_code.ipynb` (`topic1: EM`).

### 5. Experiment

We can compare two LLMs using the same questions and reference answers. For a simple demonstration, two sets of example predictions were evaluated:

| Metric        | Demo Model A | Demo Model B |
|:--------------|:------------:|:------------:|
| Raw EM        |    $80\%$    |    $80\%$    |
| Normalized EM |    $80\%$    |   $100\%$    |

*These are illustrative results, not actual LLM experiments. Normalization here means converting text to lowercase and removing extra spaces.*

### 6. Advantages and Limitations

**Advantages:** \* **Highly objective:** Binary scoring leaves no room for interpretation. \* **Reproducible:** Easy to implement and yields consistent results. \* **Efficient:** Fast to compute, suitable for large datasets. \* **Focused:** Excellent for tasks requiring precise, factual answers.

**Limitations:** \* **Overly strict:** Penalizes semantically correct but different answers. \* **Sensitive:** Small variations in wording (e.g., punctuation, synonyms) result in a $0$. \* **Limited scope:** Not suitable for open-ended or creative text generation. \* **No partial credit:** Fails to capture degrees of correctness.

### 7. Conclusion

Exact Match is useful when an LLM must produce a precise answer. However, it is not suitable as the only evaluation metric for long or creative responses because different words can express the same correct meaning.

------------------------------------------------------------------------

## Topic 2: Accuracy Evaluation

### 1. Introduction

Accuracy Evaluation is a metric used to measure how often a Large Language Model (LLM) produces correct predictions compared to the total number of predictions.

It is commonly used in classification tasks, such as sentiment analysis, spam detection, and text classification.

### 2. Theory

Accuracy measures the proportion of correct predictions made by a model.

**Formula:** $$Accuracy = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} \times 100\%$$

For binary classification, accuracy can also be calculated using: $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \times 100\%$$

Where: \* $TP$ (True Positive): Correctly predicted positive. \* $TN$ (True Negative): Correctly predicted negative. \* $FP$ (False Positive): Incorrectly predicted positive. \* $FN$ (False Negative): Incorrectly predicted negative.

For example, if a model classifies 10 customer reviews and correctly predicts 8 of them, its accuracy is $80\%$.

### 3. Practical Examples

-   **Example 1: Sentiment Classification** An LLM predicts whether customer reviews are positive or negative.
    -   **Total reviews:** 10
    -   **Correct predictions:** 8
    -   **Incorrect predictions:** 2
    -   **Accuracy:** $(8/10) \times 100\% = 80\%$
    -   **Result:** The model correctly classified $80\%$ of the reviews.
-   **Example 2: Spam Detection** An LLM predicts whether emails are spam or not spam.
    -   **Total emails:** 20
    -   **Correct predictions:** 18
    -   **Incorrect predictions:** 2
    -   **Accuracy:** $(18/20) \times 100\% = 90\%$
    -   **Result:** The model correctly classified $90\%$ of the emails.

### 4. Python Implementation

We can use Python to compare the model's predicted labels with the correct labels and calculate the percentage of correct predictions.

Python also provides the `accuracy_score()` function from the `sklearn.metrics` library to calculate accuracy easily.

The Python implementation is included in `code/Implementation_code.ipynb` (`Topic2: Accuracy`).

### 5. Experiment

We can compare two LLMs using the same questions and correct reference labels. For a simple demonstration, two sets of example predictions were evaluated:

| Metric                | Demo Model A | Demo Model B |
|:----------------------|:------------:|:------------:|
| Total predictions     |      10      |      10      |
| Correct predictions   |      8       |      9       |
| Incorrect predictions |      2       |      1       |
| Accuracy              |    $80\%$    |    $90\%$    |

*These are illustrative results, not actual LLM experiments.*

**Result:** Demo Model B performs better than Demo Model A because it correctly predicts 9 out of 10 examples, while Demo Model A correctly predicts 8. However, higher accuracy does not always mean better overall performance, especially when the dataset contains imbalanced classes.

### 6. Advantages and Limitations

**Advantages:** \* Simple and easy to understand. \* Fast to calculate. \* Useful for classification tasks. \* Easy to compare different models.

**Limitations:** \* Can be misleading when classes are imbalanced. \* Does not show which types of errors the model makes. \* Does not measure semantic similarity between generated texts. \* Not suitable as the only metric for open-ended text generation.

### 7. Conclusion

Accuracy Evaluation is useful for measuring how often an LLM makes correct predictions in classification tasks. It is simple, fast, and easy to interpret. However, accuracy alone may not provide a complete understanding of model performance, especially with imbalanced datasets. Therefore, accuracy is often used together with other metrics such as Precision, Recall, and F1-score to provide a more complete evaluation.

------------------------------------------------------------------------

## Topic 3: Precision, Recall, and F1-score

### 1. Introduction

Precision, Recall, and F1-score are evaluation metrics used to measure how well a Large Language Model (LLM) classifies information into categories.

They are commonly used in classification tasks such as spam detection, sentiment analysis, and identifying harmful content. These metrics are especially helpful when some types of mistakes are more important than others.

### 2. Theory

These three metrics use the following terms: \* $TP$ (True Positive): The model correctly predicts the positive class. \* $TN$ (True Negative): The model correctly predicts the negative class. \* $FP$ (False Positive): The model predicts positive, but the correct label is negative. \* $FN$ (False Negative): The model predicts negative, but the correct label is positive.

**Precision** measures how many predicted positive results are actually correct. $$Precision = \frac{TP}{TP + FP} \times 100\%$$

**Recall** measures how many actual positive results the model correctly finds. $$Recall = \frac{TP}{TP + FN} \times 100\%$$

**F1-score** combines Precision and Recall into one score using their harmonic mean. $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

For example, if $Precision = 80\%$ and $Recall = 80\%$, the F1-score is also $80\%$.

### 3. Practical Examples

-   **Example 1: Spam Detection** An LLM identifies spam emails.
    -   **True Positives (**$TP$): 8
    -   **False Positives (**$FP$): 2
    -   **False Negatives (**$FN$): 2
    -   **Precision:** $8/(8+2) \times 100\% = 80\%$
    -   **Recall:** $8/(8+2) \times 100\% = 80\%$
    -   **F1-score:** $80\%$
    -   **Result:** The model is correct for $80\%$ of emails it marks as spam, and it finds $80\%$ of all actual spam emails.
-   **Example 2: Negative Review Detection** An LLM identifies negative customer reviews.
    -   **True Positives (**$TP$): 6
    -   **False Positives (**$FP$): 2
    -   **False Negatives (**$FN$): 4
    -   **Precision:** $6/(6+2) \times 100\% = 75\%$
    -   **Recall:** $6/(6+4) \times 100\% = 60\%$
    -   **F1-score:** $2 \times (75 \times 60)/(75+60) = 66.67\%$
    -   **Result:** The model is fairly precise when it predicts a negative review, but it misses $40\%$ of actual negative reviews.

### 4. Python Implementation

We can use Python to compare the correct labels with predicted labels and calculate Precision, Recall, and F1-score.

The Python implementation is included in `code/Implementation_code.ipynb` (`topic3: precision_recall_f1`). The script uses only the Python standard library, so no additional packages are required.

### 5. Experiment

We can compare two LLMs using the same labeled data and the same classification task. For a simple demonstration, two sets of example predictions were evaluated. There are 10 examples: 5 Positive and 5 Negative.

| Metric                 | Demo Model A | Demo Model B |
|:-----------------------|:------------:|:------------:|
| True Positives ($TP$)  |      4       |      5       |
| True Negatives ($TN$)  |      4       |      3       |
| False Positives ($FP$) |      1       |      2       |
| False Negatives ($FN$) |      1       |      0       |
| Precision              |    $80\%$    |  $71.43\%$   |
| Recall                 |    $80\%$    |   $100\%$    |
| F1-score               |    $80\%$    |  $83.33\%$   |
| Accuracy               |    $80\%$    |    $80\%$    |

*These are illustrative results, not actual LLM experiments.*

**Result:** Both models have the same accuracy, but Demo Model A has higher Precision, while Demo Model B has higher Recall and F1-score. Demo Model B finds every actual positive example but makes more false positive predictions.

### 6. Advantages and Limitations

**Advantages:** \* Provide more detail than accuracy alone. \* Useful when the classes are imbalanced. \* Precision helps measure the reliability of positive predictions. \* Recall helps measure how many actual positives are found. \* F1-score combines Precision and Recall into one value.

**Limitations:** \* More difficult to understand than accuracy. \* Precision and Recall may trade off against each other. \* F1-score does not directly consider True Negatives. \* These metrics depend on correct reference labels. \* They do not directly measure the quality of long, open-ended LLM responses.

### 7. Conclusion

Precision, Recall, and F1-score are useful for evaluating LLMs in classification tasks. Precision focuses on the correctness of positive predictions, Recall focuses on finding actual positives, and F1-score balances both measurements. These metrics are especially important when accuracy alone does not fully explain model performance. They should be selected based on the task and the types of errors that matter most.

------------------------------------------------------------------------

## Topic 4: BLEU Score

### 1. Introduction

BLEU (Bilingual Evaluation Understudy) is an evaluation metric used to measure how similar a Large Language Model's (LLM) generated text is to a correct reference answer.

It is commonly used in machine translation and other text-generation tasks. BLEU compares words and groups of words between the generated text and the reference answer.

### 2. Theory

BLEU evaluates generated text by measuring matching words and word sequences called **n-grams**. There are different types of n-grams: \* **Unigram (1-gram):** One word, such as `"the"`. \* **Bigram (2-gram):** Two words, such as `"the student"`. \* **Trigram (3-gram):** Three words, such as `"the student reads"`. \* **4-gram:** Four words, such as `"the student reads a"`.

BLEU uses two important concepts: \* **Modified Precision:** Measures how many predicted n-grams match the reference, without counting repeated words more times than they appear in the reference. \* **Brevity Penalty (**$BP$): Reduces the score when the generated text is shorter than the reference.

**Brevity Penalty formula:** $$BP = 1, \quad \text{when } c \ge r$$ $$BP = e^{(1 - r/c)}, \quad \text{when } c < r$$

Where: \* $c =$ Length of the generated text. \* $r =$ Length of the reference text.

**General BLEU formula:** $$BLEU = BP \cdot \exp\left(\frac{1}{N}\sum_{n=1}^{N}\ln(p_n)\right)$$

Where: \* $BP =$ Brevity Penalty. \* $N =$ Maximum n-gram size. \* $p_n =$ Modified n-gram precision.

**Simplified formula for BLEU-1:** $$BLEU\text{-}1 = BP \times \text{Modified Unigram Precision} \times 100\%$$

BLEU-1 compares individual words only. BLEU-4 combines modified precision for 1-grams, 2-grams, 3-grams, and 4-grams using a geometric mean. A higher BLEU score means more words and word sequences match the reference text.

### 3. Practical Examples

-   **Example 1: Exact Match** An LLM generates the same sentence as the reference.
    -   **Reference:** `"the cat is on the mat"`
    -   **Prediction:** `"the cat is on the mat"`
    -   **BLEU-4:** $100\%$
    -   **Result:** All words and word sequences match the reference, so the model receives a perfect BLEU score.
-   **Example 2: Shorter Sentence** An LLM generates a shorter sentence.
    -   **Reference:** `"the cat is on the mat"`
    -   **Prediction:** `"the cat is on mat"`
    -   **Modified unigram precision:** $5/5 = 100\%$
    -   **Brevity Penalty:** $0.8187$
    -   **BLEU-1:** $0.8187 \times 100\% = 81.87\%$
    -   **Result:** The predicted words match the reference, but the score is reduced because the generated sentence is shorter.

### 4. Python Implementation

We can use Python to compare generated text with reference text and calculate BLEU scores. The process includes splitting sentences into words, counting matching n-grams, calculating modified precision, and applying a brevity penalty.

The Python implementation is included in `code/Implementation_code.ipynb` (`Blue_Score`). The script calculates BLEU-1 and BLEU-4 using only the Python standard library.

### 5. Experiment

We can compare two LLMs using the same sentences and reference translations. For a simple demonstration, two sets of example predictions were evaluated using five reference sentences.

| Metric          | Demo Model A | Demo Model B |
|:----------------|:------------:|:------------:|
| Total sentences |      5       |      5       |
| BLEU-1          |  $71.50\%$   |  $97.50\%$   |
| BLEU-4          |  $37.36\%$   |  $92.37\%$   |
| Brevity Penalty |   $0.8382$   |   $1.0000$   |

*These are illustrative results, not actual LLM experiments.*

**Result:** Demo Model B performs better than Demo Model A because its generated sentences have more matching words and word sequences compared with the references. However, a higher BLEU score does not always mean that the generated text has better meaning or quality.

### 6. Advantages and Limitations

**Advantages:** \* Simple and fast to calculate automatically. \* Useful for evaluating machine translation. \* Measures matching words and word sequences. \* Helps compare different language models.

**Limitations:** \* Does not understand the meaning of sentences. \* May give low scores to correct answers that use different words. \* Depends on the quality of the reference text. \* Sensitive to tokenization and sentence length. \* Does not fully measure grammar, fluency, or factual correctness.

### 7. Conclusion

BLEU Score is useful for evaluating LLM-generated text, especially in machine translation tasks. It measures how closely generated text matches reference answers by comparing words and word sequences. However, BLEU does not fully understand semantic meaning. Therefore, it is often combined with other evaluation methods or human evaluation to provide a better understanding of text-generation quality.

------------------------------------------------------------------------

## Topic 5: ROUGE Score

### 1. Introduction

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is an evaluation metric used to measure how similar a Large Language Model's (LLM) generated summary is to a correct reference summary.

It is commonly used in text summarization and other text-generation tasks. ROUGE compares matching words, word sequences, and word order between the generated summary and the reference summary.

### 2. Theory

ROUGE measures how much information in a reference summary is captured by the generated summary. There are three common types of ROUGE: \* **ROUGE-1:** Compares individual words (unigrams). \* **ROUGE-2:** Compares pairs of consecutive words (bigrams). \* **ROUGE-L:** Compares the Longest Common Subsequence (LCS), which means matching words that appear in the same order.

ROUGE can be calculated using Precision, Recall, and F1-score:

**Precision** measures how many words or word sequences in the generated summary match the reference. $$Precision = \frac{\text{Matching units}}{\text{Total predicted units}} \times 100\%$$

**Recall** measures how many words or word sequences in the reference appear in the generated summary. $$Recall = \frac{\text{Matching units}}{\text{Total reference units}} \times 100\%$$

**F1-score** combines Precision and Recall into one score. $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

For ROUGE-1 and ROUGE-2, the matching units are individual words and word pairs. For ROUGE-L, the calculation uses the length of the longest common subsequence. A higher ROUGE score means more overlap with the reference summary, but it does not always mean that the generated summary is factually correct.

### 3. Practical Examples

-   **Example 1: Shorter Summary** An LLM generates a shorter summary.
    -   **Reference:** `"the cat sat on the mat"`
    -   **Prediction:** `"the cat sat on mat"`
    -   **Matching words:** 5
    -   **ROUGE-1 Precision:** $5/5 \times 100\% = 100\%$
    -   **ROUGE-1 Recall:** $5/6 \times 100\% = 83.33\%$
    -   **ROUGE-1 F1-score:** $90.91\%$
    -   **Result:** All predicted words match the reference, but the generated summary is missing one word. Therefore, Recall is lower than Precision.
-   **Example 2: Different Word Order** An LLM generates a sentence with the same words but in a different order.
    -   **Reference:** `"the cat eats fresh fish"`
    -   **Prediction:** `"fresh fish the cat eats"`
    -   **ROUGE-1 F1-score:** $100\%$
    -   **ROUGE-2 F1-score:** $75\%$
    -   **ROUGE-L F1-score:** $60\%$
    -   **Result:** ROUGE-1 gives a perfect score because all words match. However, ROUGE-2 and ROUGE-L give lower scores because the word order is different.

### 4. Python Implementation

We can use Python to compare generated summaries with correct reference summaries and calculate ROUGE scores. The process includes splitting sentences into words, counting matching unigrams and bigrams, finding the longest common subsequence, and calculating Precision, Recall, and F1-score.

The Python implementation is included in `code/Implementation_code.ipynb` (`Topic5: ROUGE_Score`). The script calculates ROUGE-1, ROUGE-2, and ROUGE-L using only the Python standard library.

### 5. Experiment

We can compare two LLMs using the same documents and reference summaries. For a simple demonstration, two sets of example summaries were evaluated using five reference summaries.

| Metric (Average F1-score) | Demo Model A | Demo Model B |
|:--------------------------|:------------:|:------------:|
| Total summaries           |      5       |      5       |
| ROUGE-1                   |  $49.74\%$   |  $95.14\%$   |
| ROUGE-2                   |  $22.18\%$   |  $83.38\%$   |
| ROUGE-L                   |  $49.74\%$   |  $95.14\%$   |

*These are illustrative results, not actual LLM experiments. The scores are averages of individual summary F1-scores.*

**Result:** Demo Model B performs better than Demo Model A because its generated summaries contain more matching words and word sequences compared with the reference summaries. However, a higher ROUGE score does not always mean that a generated summary has better meaning or factual correctness.

### 6. Advantages and Limitations

**Advantages:** \* Simple and fast to calculate automatically. \* Useful for evaluating text summarization. \* Measures matching words and word sequences. \* ROUGE-L also considers the order of matching words. \* Helps compare different language models.

**Limitations:** \* Does not fully understand the meaning of sentences. \* May give low scores to correct summaries that use different words or synonyms. \* Depends on the quality of the reference summaries. \* Sensitive to tokenization and sentence length. \* Does not directly measure factual correctness or hallucinations.

### 7. Conclusion

ROUGE Score is useful for evaluating LLM-generated summaries by comparing them with correct reference summaries. ROUGE-1 measures matching words, ROUGE-2 measures matching word pairs, and ROUGE-L measures matching words in a similar order. However, ROUGE does not fully understand semantic meaning or factual correctness. Therefore, it is often combined with other evaluation methods or human evaluation to provide a more complete understanding of summary quality.

------------------------------------------------------------------------

## Topic 6: METEOR Score

### 1. Introduction

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is an evaluation metric used to compare a Large Language Model's (LLM) generated sentence with a correct reference sentence.

It is commonly used in machine translation and other text-generation tasks. Unlike exact word matching, METEOR can recognize related word forms and synonyms, and it also considers word order.

### 2. Theory

METEOR evaluates generated text using three types of word matching: \* **Exact Match:** Matches identical words, such as `"cat"` and `"cat"`. \* **Stemming:** Matches different forms of the same word, such as `"play"` and `"playing"`. \* **Synonym Match:** Matches words with similar meanings, such as `"car"` and `"automobile"`.

METEOR also considers word order using a **fragmentation penalty**, which reduces the score when matching words appear in a different order.

**Precision** measures how many generated words match the reference: $$Precision = \frac{\text{Matched words}}{\text{Total generated words}}$$

**Recall** measures how many reference words are found in the generated sentence: $$Recall = \frac{\text{Matched words}}{\text{Total reference words}}$$

**Recall-weighted F-mean formula:** $$F\text{-mean} = \frac{10 \times Precision \times Recall}{Recall + 9 \times Precision}$$

**Fragmentation Penalty formula:** $$Penalty = 0.5 \times \left(\frac{\text{Number of chunks}}{\text{Number of matched words}}\right)^3$$

**Final METEOR formula:** $$METEOR = F\text{-mean} \times (1 - Penalty) \times 100\%$$

These formulas use the commonly taught METEOR parameter settings. A higher METEOR score generally means the generated sentence has more matching words, related word forms, synonyms, and similar word order.

### 3. Practical Examples

-   **Example 1: Exact Word Match** An LLM generates the same sentence as the reference.
    -   **Reference:** `"the student reads a book"`
    -   **Prediction:** `"the student reads a book"`
    -   **Matched words:** 5 out of 5
    -   **Precision:** $100\%$
    -   **Recall:** $100\%$
    -   **Chunks:** 1
    -   **Penalty:** $0.004$
    -   **METEOR:** $(1 - 0.004) \times 100\% = 99.60\%$
    -   **Result:** All words match in the same order. The score is very high, with a small remaining fragmentation penalty.
-   **Example 2: Synonym Match** An LLM generates a sentence using different words with similar meanings.
    -   **Reference:** `"the car is fast"`
    -   **Prediction:** `"the automobile is quick"`
    -   `"car"` and `"automobile"` are synonyms; `"fast"` and `"quick"` are synonyms.
    -   **Matched words:** 4 out of 4
    -   **Precision:** $100\%$
    -   **Recall:** $100\%$
    -   **METEOR:** $99.22\%$
    -   **Result:** METEOR gives a high score because it recognizes synonym matches in the same order. If the prediction changes to `"quick is the automobile"`, the score falls to $78.91\%$ because the matching words appear in a different order.
    -   *Note:* These examples use NLTK's METEOR calculation with a small offline synonym dictionary.

### 4. Python Implementation

We can use Python and the NLTK library to calculate METEOR scores by comparing generated sentences with reference sentences. The process includes splitting sentences into words, finding exact matches, stemmed matches, and synonym matches, calculating Precision and Recall, and applying a word-order penalty.

The Python implementation is included in `code/Implementation_code.ipynb` (`METEOR_Score`). The script uses NLTK and includes a small offline synonym dictionary for demonstration. It also supports the full WordNet corpus for broader synonym matching.

### 5. Experiment

We can compare two LLMs using the same sentences and reference answers. For a simple demonstration, two sets of example predictions were evaluated using five reference sentences.

| Metric          | Demo Model A | Demo Model B |
|:----------------|:------------:|:------------:|
| Total sentences |      5       |      5       |
| Average METEOR  |  $47.28\%$   |  $99.48\%$   |

*These are illustrative results, not actual LLM experiments. The scores are averages of sentence-level METEOR scores calculated using NLTK with a limited offline synonym dictionary.*

**Result:** Demo Model B performs better than Demo Model A because its generated sentences contain more matching words, related word forms, and synonyms in a similar order. However, a higher METEOR score does not always mean the generated text is factually correct or has better overall meaning.

### 6. Advantages and Limitations

**Advantages:** \* Recognizes exact word matches, stemming, and synonyms. \* Considers both Precision and Recall. \* Gives more importance to Recall. \* Considers word order using a fragmentation penalty. \* Useful for evaluating machine translation and generated text.

**Limitations:** \* More complicated to calculate than Exact Match or Accuracy. \* Depends on language resources for synonym matching. \* Does not fully understand the meaning of sentences. \* May give high scores to sentences containing factual mistakes. \* Results can vary depending on the implementation and settings.

### 7. Conclusion

METEOR Score is useful for evaluating LLM-generated text by comparing it with reference sentences. It considers exact words, related word forms, synonyms, and word order while combining Precision and Recall. However, METEOR does not fully measure semantic meaning or factual correctness. Therefore, it is often combined with other evaluation metrics or human evaluation to provide a more complete understanding of generated-text quality.

------------------------------------------------------------------------

## Topic 7: BERTScore

### 1. Introduction

BERTScore is an evaluation metric used to measure how similar a Large Language Model's (LLM) generated text is to a correct reference text.

It is commonly used in machine translation, question answering, and text summarization. Unlike BLEU and ROUGE, BERTScore uses contextual word embeddings to compare the meaning of words instead of only checking exact word matches.

### 2. Theory

BERTScore uses a pretrained language model, such as BERT or RoBERTa, to convert words or tokens into numerical vectors called **embeddings**. These embeddings represent information about a word and its surrounding context.

BERTScore compares the embeddings using **Cosine Similarity** to measure how similar two tokens are. It calculates three main metrics: \* **BERTScore Precision:** Measures how well the generated tokens match the reference tokens. \* **BERTScore Recall:** Measures how well the reference tokens are covered by the generated tokens. \* **BERTScore F1:** Combines Precision and Recall into one score.

**Cosine Similarity Formula:** $$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Where $A$ and $B$ are embedding vectors.

**BERTScore Precision Formula:** $$Precision = \text{Average of the highest similarity scores for generated tokens}$$

**BERTScore Recall Formula:** $$Recall = \text{Average of the highest similarity scores for reference tokens}$$

**BERTScore F1 Formula:** $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

These formulas describe the basic calculation without optional IDF weighting or baseline rescaling. A higher BERTScore usually indicates greater similarity between the generated and reference texts. However, it does not guarantee that the generated text is factually correct.

### 3. Practical Examples

-   **Example 1: Exact Sentence Match** An LLM generates the same sentence as the reference.
    -   **Reference:** `"the cat is sleeping"`
    -   **Prediction:** `"the cat is sleeping"`
    -   **BERTScore F1:** Approximately $100\%$ with ordinary unrescaled scoring.
    -   **Result:** The sentences contain the same words in the same context, so their token representations match.
-   **Example 2: Similar Meaning with Different Words** An LLM generates a sentence using different words with similar meanings.
    -   **Reference:** `"fast car"`
    -   **Prediction:** `"quick automobile"`
    -   `"fast"` and `"quick"` have similar meanings; `"car"` and `"automobile"` have similar meanings.
    -   For a simple educational example, suppose the highest token similarity scores are $0.90$ and $0.92$ in both matching directions:
    -   **Precision:** $(0.90 + 0.92)/2 = 91\%$
    -   **Recall:** $(0.90 + 0.92)/2 = 91\%$
    -   **F1-score:** $91\%$
    -   **Result:** The score is high because the words have related meanings, even though the sentences do not contain the same words.
    -   *Note:* The $91\%$ score is an illustrative calculation using assumed similarities, not an actual BERTScore measurement.

### 4. Python Implementation

We can use Python and the `bert-score` library to calculate BERTScore by comparing generated sentences with reference sentences. The process includes converting tokens into contextual embeddings, calculating cosine similarity, finding the best token matches, and calculating Precision, Recall, and F1-score.

The Python implementation is included in `code/Implementation_code.ipynb` (`BERTScore`). The script supports an offline mathematical demonstration and a real BERTScore calculation using a pretrained language model.

### 5. Experiment

We can compare two LLMs using the same questions and correct reference answers. For a simple demonstration, two sets of example predictions were compared using five reference phrases.

| Metric                | Demo Model A | Demo Model B |
|:----------------------|:------------:|:------------:|
| Total examples        |      5       |      5       |
| Average toy Precision |  $56.70\%$   |  $90.70\%$   |
| Average toy Recall    |  $56.90\%$   |  $90.70\%$   |
| Average toy F1-score  |  $56.80\%$   |  $90.70\%$   |

*These results are calculated from manually assigned token similarities to demonstrate the BERTScore formulas. They are not actual BERTScore measurements or real LLM experiment results.*

**Result:** Demo Model B has a higher illustrative similarity score because its predictions were assigned higher similarity values to the reference phrases. For a real experiment, we must collect outputs from two actual LLMs and calculate their BERTScore using the same evaluation model and settings.

### 6. Advantages and Limitations

**Advantages:** \* Can recognize similarities between different words. \* Uses contextual embeddings instead of exact matching alone. \* Measures Precision, Recall, and F1-score. \* Useful for machine translation and text summarization. \* Can evaluate paraphrases more effectively than simple word-overlap methods.

**Limitations:** \* Requires more computational resources than simple evaluation metrics. \* Depends on a pretrained language model. \* Does not always detect factual mistakes. \* Results depend on the evaluation model and its settings. \* May not correctly evaluate all differences in meaning, numbers, or negation.

### 7. Conclusion

BERTScore is useful for evaluating LLM-generated text by comparing contextual token embeddings with reference answers. It measures token similarity and combines Precision and Recall into an F1-score, allowing it to recognize some similarities even when different words are used. However, BERTScore does not guarantee factual correctness or complete understanding of meaning. Therefore, it is often combined with other evaluation metrics or human evaluation to provide a more complete assessment of generated-text quality.

------------------------------------------------------------------------

## Topic 8: Word Error Rate (WER)

### 1. Introduction

Word Error Rate (WER) is an evaluation metric used to measure how many word-level mistakes appear in a generated sentence compared with the correct reference sentence.

It is commonly used in speech recognition and transcription tasks, including evaluating LLM-based transcription systems. WER checks whether words are substituted, deleted, or inserted.

### 2. Theory

WER compares the predicted sentence with the reference sentence and counts three types of errors: \* **Substitution (**$S$): A reference word is replaced by a different word. *Example:* `"cat"` becomes `"dog"`. \* **Deletion (**$D$): A reference word is missing from the prediction. *Example:* `"I like blue cars"` becomes `"I like cars"`. \* **Insertion (**$I$): An extra word appears in the prediction. *Example:* `"I like cars"` becomes `"I really like cars"`.

**WER Formula:** $$WER = \frac{S + D + I}{N} \times 100\%$$

Where: \* $S =$ Number of substitutions. \* $D =$ Number of deletions. \* $I =$ Number of insertions. \* $N =$ Total number of words in the reference sentence.

WER uses the minimum number of word edits needed to change the reference sentence into the prediction. This is called **Levenshtein edit distance**.

A lower WER score means better performance: \* $WER = 0\%$: All words match correctly. \* $WER = 25\%$: One word error out of four reference words. \* $WER = 100\%$: The number of word errors equals the number of reference words. \* WER can also exceed $100\%$ when a prediction contains many extra words.

### 3. Practical Examples

-   **Example 1: Substitution Error** An LLM-based transcription system produces one incorrect word.
    -   **Reference:** `"the cat is sleeping"`
    -   **Prediction:** `"the dog is sleeping"`
    -   **Substitutions (**$S$): 1
    -   **Deletions (**$D$): 0
    -   **Insertions (**$I$): 0
    -   **Total reference words (**$N$): 4
    -   **WER:** $(1+0+0)/4 \times 100\% = 25\%$
    -   **Result:** The model makes one word substitution error, so its WER is $25\%$.
-   **Example 2: Insertion and Deletion Errors** A model adds an extra word and misses one reference word.
    -   **Reference:** `"the teacher explained a problem clearly"`
    -   **Prediction:** `"yesterday the teacher explained a problem"`
    -   **Substitutions (**$S$): 0
    -   **Deletions (**$D$): 1 (`"clearly"` is missing)
    -   **Insertions (**$I$): 1 (`"yesterday"` is added)
    -   **Total reference words (**$N$): 6
    -   **WER:** $(0+1+1)/6 \times 100\% = 33.33\%$
    -   **Result:** The model makes two word errors, so its WER is $33.33\%$.

### 4. Python Implementation

We can use Python to compare predicted sentences with correct reference sentences and calculate Word Error Rate. The process includes splitting sentences into words, finding the minimum number of substitutions, deletions, and insertions, and calculating the percentage of word errors.

The Python implementation is included in `code/Implementation_code.ipynb` (`Topic8: WER`). The script uses dynamic programming to calculate Levenshtein edit distance and requires only the Python standard library.

The implementation uses simple lowercase, punctuation-insensitive word tokenization for English examples. Other languages, such as Khmer, may require a different word tokenizer.

### 5. Experiment

We can compare two LLMs or transcription systems using the same reference sentences. For a simple demonstration, two sets of example predictions were evaluated using five reference sentences containing 22 words in total.

| Metric                | Demo Model A | Demo Model B |
|:----------------------|:------------:|:------------:|
| Total sentences       |      5       |      5       |
| Total reference words |      22      |      22      |
| Substitutions ($S$)   |      4       |      0       |
| Deletions ($D$)       |      4       |      2       |
| Insertions ($I$)      |      0       |      0       |
| Total errors          |      8       |      2       |
| Word Error Rate       |  $36.36\%$   |   $9.09\%$   |

*These are illustrative results, not actual LLM experiments. The overall WER is calculated by dividing the total number of errors by the total number of reference words.*

**Result:** Demo Model B performs better than Demo Model A because it produces fewer word errors. Demo Model A has a WER of $36.36\%$, while Demo Model B has a WER of only $9.09\%$. Unlike Accuracy, BLEU, or ROUGE, a lower WER score indicates better performance.

### 6. Advantages and Limitations

**Advantages:** \* Simple and easy to understand. \* Measures substitutions, deletions, and insertions. \* Useful for speech recognition and transcription evaluation. \* Easy to calculate and compare between models. \* Does not require a pretrained evaluation model.

**Limitations:** \* Does not understand the meaning of sentences. \* Treats all word errors equally, even when some mistakes are more important. \* May give high error rates to correct sentences that use different words. \* Depends on consistent word tokenization and text normalization. \* Not suitable as the only metric for open-ended text generation.

### 7. Conclusion

Word Error Rate is useful for evaluating how accurately a model reproduces the words in a reference sentence, especially in speech recognition and transcription tasks. It measures substitution, deletion, and insertion errors. A lower WER indicates fewer word errors, while a WER of $0\%$ means a perfect word-level match under the chosen preprocessing rules. However, WER does not understand semantic meaning or factual correctness. Therefore, it should be combined with other evaluation methods when evaluating general LLM-generated text.

------------------------------------------------------------------------

## Topic 9: Character Error Rate (CER)

### 1. Introduction

Character Error Rate (CER) is an evaluation metric used to measure how many character-level mistakes appear in generated text compared with the correct reference text.

It is commonly used in Optical Character Recognition (OCR), speech transcription, and text-recognition tasks, including evaluating LLM-assisted transcription systems. CER checks whether characters are substituted, deleted, or inserted.

### 2. Theory

CER compares the predicted text with the reference text and counts three types of errors: \* **Substitution (**$S$): A character is replaced by a different character. *Example:* `"cat"` becomes `"cut"`. \* **Deletion (**$D$): A character is missing from the prediction. *Example:* `"hello"` becomes `"helo"`. \* **Insertion (**$I$): An extra character appears in the prediction. *Example:* `"book"` becomes `"books"`.

**CER Formula:** $$CER = \frac{S + D + I}{N} \times 100\%$$

Where: \* $S =$ Number of character substitutions. \* $D =$ Number of character deletions. \* $I =$ Number of character insertions. \* $N =$ Total number of characters in the reference text.

CER uses the minimum number of character edits needed to change the reference into the prediction. This is called **Levenshtein edit distance**.

A lower CER score means better performance: \* $CER = 0\%$: All characters match correctly. \* $CER = 25\%$: One character error out of four reference characters. \* $CER = 100\%$: The number of character errors equals the number of reference characters. \* CER can also exceed $100\%$ when the prediction contains many extra characters.

### 3. Practical Examples

-   **Example 1: Substitution Error** A text-recognition model predicts one incorrect character.
    -   **Reference:** `"cat"`
    -   **Prediction:** `"cut"`
    -   **Substitutions (**$S$): 1
    -   **Deletions (**$D$): 0
    -   **Insertions (**$I$): 0
    -   **Total reference characters (**$N$): 3
    -   **CER:** $(1+0+0)/3 \times 100\% = 33.33\%$
    -   **Result:** The model makes one character substitution error, so its CER is $33.33\%$.
-   **Example 2: Deletion and Insertion Errors** A model either misses a character or adds an extra character.
    -   **Deletion:**
        -   **Reference:** `"hello"`
        -   **Prediction:** `"helo"`
        -   **Deletions (**$D$): 1
        -   **Total reference characters:** 5
        -   **CER:** $1/5 \times 100\% = 20\%$
    -   **Insertion:**
        -   **Reference:** `"book"`
        -   **Prediction:** `"books"`
        -   **Insertions (**$I$): 1
        -   **Total reference characters:** 4
        -   **CER:** $1/4 \times 100\% = 25\%$
    -   **Result:** The first prediction is missing one character, while the second contains an extra character. Both are counted as errors.

### 4. Python Implementation

We can use Python to compare predicted text with correct reference text and calculate Character Error Rate. The process includes comparing characters, finding the minimum number of substitutions, deletions, and insertions, and calculating the percentage of character errors.

The Python implementation is included in `code/Implementation_code.ipynb` (`Topic9: CER`). The script uses dynamic programming to calculate Levenshtein edit distance and requires only the Python standard library.

For consistency, the implementation normalizes Unicode text, ignores uppercase/lowercase differences, and removes surrounding whitespace. Internal spaces and punctuation are counted as characters.

### 5. Experiment

We can compare two LLMs or text-recognition systems using the same reference texts. For a simple demonstration, two sets of example predictions were evaluated using five reference words containing 21 characters in total.

| Metric                     | Demo Model A | Demo Model B |
|:---------------------------|:------------:|:------------:|
| Total examples             |      5       |      5       |
| Total reference characters |      21      |      21      |
| Substitutions ($S$)        |      2       |      1       |
| Deletions ($D$)            |      2       |      1       |
| Insertions ($I$)           |      1       |      0       |
| Total errors               |      5       |      2       |
| Character Error Rate       |  $23.81\%$   |   $9.52\%$   |

*These are illustrative results, not actual LLM experiments. The overall CER is calculated by dividing total character errors by total reference characters.*

**Result:** Demo Model B performs better than Demo Model A because it makes fewer character errors. Demo Model A has a CER of $23.81\%$, while Demo Model B has a CER of only $9.52\%$. Unlike Accuracy, BLEU, or ROUGE, a lower CER score indicates better performance.

### 6. Advantages and Limitations

**Advantages:** \* Simple and easy to calculate. \* Measures character-level substitutions, deletions, and insertions. \* Useful for OCR, speech transcription, and text recognition. \* Can detect small spelling mistakes. \* Does not require a pretrained evaluation model.

**Limitations:** \* Does not understand the meaning of sentences. \* Treats all character errors equally, even when some mistakes are more important. \* Sensitive to spaces, punctuation, and text normalization. \* Character counting can be complicated for languages with combining marks, such as Khmer. \* Not suitable as the only metric for open-ended text generation.

### 7. Conclusion

Character Error Rate is useful for evaluating how accurately a model reproduces the characters in a reference text, especially in OCR and transcription tasks. It measures substitution, deletion, and insertion errors. A lower CER indicates fewer character errors, while a CER of $0\%$ means a perfect character-level match under the chosen preprocessing rules. However, CER does not understand semantic meaning or factual correctness. Therefore, it should be combined with other evaluation methods when evaluating general LLM-generated text.

------------------------------------------------------------------------

## Topic 10: Comparative Evaluation of LLM Metrics

### 1. Introduction

Comparative Evaluation of LLM Metrics is the process of comparing different evaluation methods used to measure the performance of Large Language Models (LLMs).

Different LLM tasks require different evaluation methods. For example, Accuracy is useful for classification, BLEU is useful for translation, ROUGE is useful for summarization, and WER or CER is useful for transcription. The main goal is to understand the strengths and limitations of each evaluation metric and choose the appropriate method for a specific task.

### 2. Theory

There are nine evaluation topics used to measure different aspects of LLM performance:

| Metric | What it measures | Common application |
|:-----------------------|:-----------------------|:-----------------------|
| **Exact Match (EM)** | Checks whether a prediction exactly matches the reference answer | Question Answering |
| **Accuracy** | Measures the percentage of correct predictions | Text Classification |
| **Precision, Recall, F1-score** | Measure positive prediction quality, coverage, and balance | Sentiment Classification |
| **BLEU** | Measures matching words and word sequences | Machine Translation |
| **ROUGE** | Measures word and sequence overlap | Text Summarization |
| **METEOR** | Considers word matches, stemming, synonyms, and word order | Text Generation |
| **BERTScore** | Measures contextual similarity using embeddings | Text Generation |
| **Word Error Rate (WER)** | Measures word-level errors | Speech Recognition |
| **Character Error Rate (CER)** | Measures character-level errors | OCR and Transcription |

**Important Formulas:** \* **Exact Match:** $EM = \frac{\text{Exact Matches}}{\text{Total Answers}} \times 100\%$ \* **Accuracy:** $Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100\%$ \* **Precision:** $Precision = \frac{TP}{TP + FP} \times 100\%$ \* **Recall:** $Recall = \frac{TP}{TP + FN} \times 100\%$ \* **F1-score:** $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$ \* **BLEU:** Combines modified n-gram precision with a brevity penalty. \* **ROUGE:** Uses overlap-based Precision, Recall, and F1-score. \* **METEOR:** $METEOR = F\text{-mean} \times (1 - \text{Fragmentation Penalty})$ \* **BERTScore:** Uses cosine similarity between contextual token embeddings, followed by Precision, Recall, and F1 calculation. \* **WER:** $WER = \frac{S + D + I}{\text{Total Reference Words}} \times 100\%$ \* **CER:** $CER = \frac{S + D + I}{\text{Total Reference Characters}} \times 100\%$

For EM, Accuracy, Precision, Recall, F1, BLEU, ROUGE, METEOR, and usually BERTScore, **higher scores** indicate better performance. For WER and CER, **lower scores** indicate better performance. Because these metrics measure different things, their percentages should not be directly compared across different tasks.

### 3. Practical Examples

-   **Example 1: Exact Match and Meaning** An LLM answers a question about Cambodia.
    -   **Question:** `"What is the capital of Cambodia?"`
    -   **Reference:** `"Phnom Penh"`
    -   **Prediction A:** `"Phnom Penh"`
    -   **Prediction B:** `"The capital is Phnom Penh."`
    -   **Exact Match A:** $100\%$
    -   **Exact Match B:** $0\%$
    -   **Result:** Both answers provide the correct location, but Exact Match only accepts the exact reference answer. This demonstrates that exact matching does not fully evaluate semantic meaning.
-   **Example 2: Word Error Rate vs. Character Error Rate** A transcription system makes one word-level error.
    -   **Reference:** `"cat"`
    -   **Prediction:** `"cut"`
    -   **WER:** $1/1 \times 100\% = 100\%$
    -   **CER:** $1/3 \times 100\% = 33.33\%$
    -   **Result:** WER gives $100\%$ because the entire word is different, while CER gives $33.33\%$ because only one character is incorrect.
-   **Example 3: Accuracy in Classification** An LLM classifies customer reviews as positive or negative.
    -   **Total reviews:** 10
    -   **Correct predictions:** 9
    -   **Incorrect predictions:** 1
    -   **Accuracy:** $9/10 \times 100\% = 90\%$
    -   **Result:** The model correctly classifies $90\%$ of the reviews. However, Precision and Recall may be needed to understand its classification errors in more detail.

### 4. Python Implementation

We can use Python to compare different evaluation metrics and analyze how two models perform on different NLP tasks.

The implementation is included in `code/Implementation_code.ipynb` (`topic10: comparative_evaluation`). The program uses separate example datasets for question answering, classification, text generation, and transcription.

It reuses the previous Python implementations for Exact Match, BLEU, ROUGE, METEOR, WER, and CER. It also calculates Accuracy, Precision, Recall, and F1-score. BERTScore requires an additional pretrained language model and can be calculated using the optional real BERTScore mode.

### 5. Experiment

We can compare two LLMs by giving them the same questions or tasks and evaluating their outputs using appropriate metrics. For a simple demonstration, two sets of example predictions were evaluated across four different NLP tasks:

| Metric       | Demo Model A | Demo Model B |
|:-------------|:------------:|:------------:|
| Exact Match  |  $60.00\%$   |  $100.00\%$  |
| Accuracy     |  $80.00\%$   |  $90.00\%$   |
| Precision    |  $80.00\%$   |  $83.33\%$   |
| Recall       |  $80.00\%$   |  $100.00\%$  |
| F1-score     |  $80.00\%$   |  $90.91\%$   |
| BLEU-4       |  $20.27\%$   |  $84.94\%$   |
| ROUGE-1 F1   |  $53.21\%$   |  $93.33\%$   |
| ROUGE-2 F1   |  $24.05\%$   |  $84.00\%$   |
| ROUGE-L F1   |  $53.21\%$   |  $93.33\%$   |
| METEOR       |  $39.06\%$   |  $95.97\%$   |
| BERTScore F1 | Not computed | Not computed |
| WER          |  $28.57\%$   |   $4.76\%$   |
| CER          |  $19.63\%$   |   $0.93\%$   |

*These results come from invented educational examples, not actual LLM outputs. BERTScore was not computed because it requires a pretrained evaluation model. The Python script includes an optional mode for calculating actual BERTScore.*

**Result:** Demo Model B performs better than Demo Model A on the evaluated examples. It achieves higher EM, Accuracy, F1, BLEU, ROUGE, and METEOR scores, along with lower WER and CER. However, these results only demonstrate the evaluation methods. A real experiment must use outputs collected from two actual LLMs.

### 6. Advantages and Limitations

**Advantages:** \* Helps select the right evaluation metric for different LLM tasks. \* Provides a more complete understanding of model performance. \* Makes it easier to compare models on the same evaluation task. \* Combines classification, text similarity, and error measurement. \* Helps identify weaknesses that one evaluation metric might miss.

**Limitations:** \* No single metric is suitable for every LLM task. \* Different evaluation metrics have different scoring methods. \* Some metrics do not fully understand semantic meaning. \* BERTScore requires more computational resources. \* Automatic metrics do not always detect factual errors. \* Results depend on the quality of the reference answers and evaluation datasets.

### 7. Conclusion

Comparative Evaluation of LLM Metrics is useful for understanding how different evaluation methods measure LLM performance. Exact Match is useful for precise answers, Accuracy and F1-score are useful for classification, BLEU and ROUGE are useful for generated text, METEOR and BERTScore help evaluate text similarity, and WER and CER measure transcription errors.

Each metric has its own strengths and limitations. Therefore, choosing the correct metric depends on the task, the expected output, and the type of errors we want to measure. Using multiple appropriate evaluation metrics can provide a more complete understanding of LLM performance.
