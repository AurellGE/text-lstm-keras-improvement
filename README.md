# Text Classification with LSTM (Keras) - Improved Version

## Requirements
- Python 3.x
- TensorFlow (for building and training the LSTM model)
- scikit-learn (for preprocessing and model evaluation)
- nltk (for text preprocessing and tokenization)
- pandas (for data handling)
- numpy (for numerical computations)
- tqdm (for progress bars)
- Matplotlib and Seaborn (for data visualization)

## Libraries Used
- **TensorFlow**: For building and training the LSTM model for text classification.
- **scikit-learn**: For data splitting, preprocessing, and evaluation metrics.
- **nltk**: For text preprocessing, including tokenization, stopword removal, and lemmatization.
- **pandas**: For efficient handling and manipulation of the dataset.
- **numpy**: For numerical operations, including array manipulation.
- **tqdm**: For displaying progress bars during long-running processes.
- **Matplotlib/Seaborn**: For visualizing results like confusion matrices and training curves.

## Changes and Improvements
- **Data Preprocessing:** Modularized the data cleaning process by creating a `cleansing()` function to handle missing values and unnecessary columns more efficiently.
- **Model Architecture:** Improved the LSTM model by adjusting layer configurations and adding dropout layers to reduce overfitting.
- **Evaluation Metrics:** Enhanced model evaluation by incorporating more detailed metrics like F1-score and precision-recall curves, in addition to accuracy.
- **Visualization:** Added more extensive visualizations using `seaborn` and `matplotlib` to better assess model performance and data insights.
- **Error Handling:** Improved error handling, particularly during the text preprocessing steps, to avoid potential issues with tokenization or missing data.

## Final Analysis

### Old Code 
The original implementation of the model showed severe limitations, achieving only **34% accuracy**, with the classifier predicting nearly all inputs as class 1. The classification report confirmed that **classes 0 and 2 received zero precision and recall**, indicating that the model failed to learn any meaningful distinctions between emotion categories. This underperformance can be traced to several critical shortcomings in the original code:
* The use of static Word2Vec embeddings without proper alignment to the tokenizer
* Lack of one-hot encoding for multi-class targets
* Minimal model architecture
* The absence of validation tracking or robust evaluation.

### New Code
In contrast, the improved version of the project introduced several key changes:
* **Implemented a cleaner and more consistent preprocessing pipeline**, which included:
  * Lowercase normalization
  * Stopword removal
  * Post-tokenization padding based on actual sequence length distribution.
* **Used a trainable embedding layer** instead of relying on pretrained embeddings, allowing the model to learn task-specific word representations directly.
* **Applied one-hot encoding to the target labels**, enabling the use of `categorical_crossentropy` loss and softmax activation for accurate multi-class classification.
* **Added a bidirectional LSTM and a tunable dense layer** to the model architecture to increase representational capacity.
* **Conducted a structured hyperparameter tuning** process using a grid search and received the best configuration:
  * `lstm_units=64`
  * `dropout_rate=0.3`
  * `learning_rate=0.001`
  * `dense_units=128`
| Model Version                 | Accuracy | Class 0 F1 | Class 1 F1 | Class 2 F1 | Macro F1 |
| ----------------------------- | -------- | ---------- | ---------- | ---------- | -------- |
| **Old Code**                  | 0.34     | 0.00       | 0.50       | 0.00       | 0.17     |
| **Baseline**                  | 0.91     | 0.91       | 0.90       | 0.92       | 0.91     |
| **Tuned Model (Best Config)** ✅ | 0.93     | 0.91       | 0.92       | 0.94       | 0.92     |

It achieved a final **accuracy of 93%**, with balanced F1-scores above 0.90 across all classes. This is a notable leap from the baseline model's 91% accuracy, and a substantial improvement over the original code with 0.34% of accuracy.

### Conclusion
The project demonstrates how thoughtful improvements in preprocessing, model design, and parameter tuning can greatly enhance classification performance. From a failing model that misclassified nearly all classes, the improved version became a high-performing, generalizable model. The results highlight the importance of not just building a model, but carefully refining each component—from input preparation to training strategy—to achieve optimal outcomes.
