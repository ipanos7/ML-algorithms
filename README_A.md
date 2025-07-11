

# Περιγραφή Εργασίας: Κατάταξη Κειμένου με χρήση 3 αλγορίθμων μηχανικής μάθησης

## Σε αυτό το μέρος υλοποιούμε 3 αλγόριθμους μηχανικής μάθησης για binary text classifiacation.Αυτοί είναι οι ακόλουθοι:
1) Naive Bayes
2) Logistic Regression with Stohastic Gradient Descent and regularization
3) Random Forest using ID3 trees

## Απαιτήσεις προγράμματος 

Για την εκτέλεση των 3 επιμέρους αλγορίθμων χρειάζονται οι ακόλουθες βιβλιοθήκες(με τη χρήση pip)
-> pip install numpy pandas scikit-learn matplotlib beautifulsoup4

## Dataset

Για το project χρησιμοποιήσαμε το IMDB Large Movie Review Dataset και το σχετικό link ,δίνεται στην περιγραφή της εργασίας

## Τρόπος μεταγλώτισσης και εκτέλεσης προγράμματος

Για τη μεταγλώτισση και εκτέλεση του προγράμματος , επειδή τον κώδικα τον υλοποιήσαμε σε python notebooks σε vs code , θα πρέπει για κάθε ipynb file να πατήσετε σε κάθε cell το κουμπί του run code για τα 3 αρχεία (NaiveBayes.ipynb, LogisticRegression.ipynb, RandomForest.ipynb)

## Outputs

Στο output περιλαμβάνονται -> 1) καμπύλες μάθησης με precision,recall και F1 Scores για training και development data (έγινε κατάλληλο split του αρχικού dataset σε δεδομένα εκπαίδευσης και ανάπτυξης,η διαδικασία εξηγείται στο έγγραφο αναφοράς) και ->2) αξιολόγηση πάνω στο test set με αποτελέσματα precision,recall και micro-macro averaged F1 scores. Λεπτομέρειες παρατίθενται στο σχετικό report