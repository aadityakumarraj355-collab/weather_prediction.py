
"""
Spam Email Classifier
Classifies emails as spam or ham (not spam) using machine learning techniques.
Uses Naive Bayes and Support Vector Machine (SVM) classifiers.
"""
import sys
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError as e:
    missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"✗ Missing required module: {missing_module}")
    print("Install missing packages with: pip install pandas numpy matplotlib seaborn scikit-learn")
    sys.exit(1)
import warnings
warnings.filterwarnings('ignore')

class SpamEmailClassifier:
    """A class to classify emails as spam or ham."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.nb_model = MultinomialNB()
        self.svm_model = SVC(kernel='linear', random_state=42)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath='sample_emails.csv'):
        """Load email data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {len(df)} emails")
            print(f"  - Spam: {sum(df['label'] == 'spam')}")
            print(f"  - Ham: {sum(df['label'] == 'ham')}")
            return df
        except FileNotFoundError:
            print(f"✗ Error: {filepath} not found!")
            return None
    
    def preprocess_data(self, df):
        """Preprocess and split the data."""
        # Convert labels to binary (spam=1, ham=0)
        df['label_binary'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Split data
        X = df['text']
        y = df['label_binary']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Transform text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print("\n✓ Data preprocessed successfully")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Testing samples: {len(self.X_test)}")
        print(f"  - Features extracted: {X_train_tfidf.shape[1]}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_naive_bayes(self, X_train_tfidf):
        """Train Naive Bayes classifier."""
        print("\n🔄 Training Naive Bayes classifier...")
        self.nb_model.fit(X_train_tfidf, self.y_train)
        print("✓ Naive Bayes model trained successfully")
    
    def train_svm(self, X_train_tfidf):
        """Train SVM classifier."""
        print("\n🔄 Training SVM classifier...")
        self.svm_model.fit(X_train_tfidf, self.y_train)
        print("✓ SVM model trained successfully")
    
    def evaluate_model(self, model, X_test_tfidf, model_name):
        """Evaluate model performance."""
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"{model_name} Performance")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Ham', 'Spam']))
        
        return y_pred, accuracy
    
    def plot_confusion_matrix(self, y_pred, model_name):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        print(f"✓ Confusion matrix saved as '{model_name.lower().replace(' ', '_')}_confusion_matrix.png'")
        plt.close()
    
    def predict_custom_email(self, email_text, use_svm=False):
        """Predict if a custom email is spam or ham."""
        email_tfidf = self.vectorizer.transform([email_text])
        model = self.svm_model if use_svm else self.nb_model
        prediction = model.predict(email_tfidf)[0]
        model_name = "SVM" if use_svm else "Naive Bayes"
        
        result = "SPAM" if prediction == 1 else "HAM"
        print(f"\n📧 Email Classification ({model_name})")
        print(f"Text: {email_text[:100]}...")
        print(f"Prediction: {result}")
        return result

def main():
    """Main function to run the spam classifier."""
    print("="*60)
    print("      SPAM EMAIL CLASSIFIER      ")
    print("="*60)
    
    # Initialize classifier
    classifier = SpamEmailClassifier()
    
    # Load data
    df = classifier.load_data('sample_emails.csv')
    if df is None:
        return
    
    # Preprocess data
    X_train_tfidf, X_test_tfidf = classifier.preprocess_data(df)
    
    # Train models
    classifier.train_naive_bayes(X_train_tfidf)
    classifier.train_svm(X_train_tfidf)
    
    # Evaluate Naive Bayes
    nb_pred, nb_accuracy = classifier.evaluate_model(
        classifier.nb_model, X_test_tfidf, "Naive Bayes"
    )
    classifier.plot_confusion_matrix(nb_pred, "Naive Bayes")
    
    # Evaluate SVM
    svm_pred, svm_accuracy = classifier.evaluate_model(
        classifier.svm_model, X_test_tfidf, "SVM"
    )
    classifier.plot_confusion_matrix(svm_pred, "SVM")
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
    print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    best_model = "SVM" if svm_accuracy > nb_accuracy else "Naive Bayes"
    print(f"\n🏆 Best Model: {best_model}")
    
    # Test with custom emails
    print("\n" + "="*50)
    print("TESTING WITH CUSTOM EMAILS")
    print("="*50)
    
    test_emails = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hi, can we schedule a meeting for tomorrow at 3pm?",
        "URGENT: Your account will be closed. Verify your identity immediately!",
        "Thanks for your presentation today. The team really appreciated it."
    ]
    
    for email in test_emails:
        classifier.predict_custom_email(email, use_svm=True)
    
    print("\n" + "="*60)
    print("✓ Classification completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
