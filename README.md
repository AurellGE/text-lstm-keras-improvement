# Text Classification with LSTM (Keras) - Improved Version
---

## Requirements
- Python 3.x
- TensorFlow (for building and training the LSTM model)
- scikit-learn (for preprocessing and model evaluation)
- nltk (for text preprocessing and tokenization)
- pandas (for data handling)
- numpy (for numerical computations)
- tqdm (for progress bars)
- Matplotlib and Seaborn (for data visualization)

---
## Libraries Used
- **TensorFlow**: For building and training the LSTM model for text classification.
- **scikit-learn**: For data splitting, preprocessing, and evaluation metrics.
- **nltk**: For text preprocessing, including tokenization, stopword removal, and lemmatization.
- **pandas**: For efficient handling and manipulation of the dataset.
- **numpy**: For numerical operations, including array manipulation.
- **tqdm**: For displaying progress bars during long-running processes.
- **Matplotlib/Seaborn**: For visualizing results like confusion matrices and training curves.

---
## Changes and Improvements
- **Data Preprocessing:** Modularized the data cleaning process by creating a `cleansing()` function to handle missing values and unnecessary columns more efficiently.
- **Model Architecture:** Improved the LSTM model by adjusting layer configurations and adding dropout layers to reduce overfitting.
- **Evaluation Metrics:** Enhanced model evaluation by incorporating more detailed metrics like F1-score and precision-recall curves, in addition to accuracy.
- **Visualization:** Added more extensive visualizations using `seaborn` and `matplotlib` to better assess model performance and data insights.
- **Error Handling:** Improved error handling, particularly during the text preprocessing steps, to avoid potential issues with tokenization or missing data.
