# **Disease Prediction & Kidney X-ray Classification**  
### *AI-Powered Healthcare Solution for Early Disease Detection*  

## 📌 **Problem Statement**  
Healthcare is evolving, but early disease detection remains a challenge. Many people struggle to identify illnesses based on symptoms, and diagnosing kidney-related conditions from X-ray images requires specialized medical expertise.  

### **Solution**  
This project provides a **dual AI-powered approach**:  
1. **Disease Prediction Model:** A machine learning model that predicts potential diseases based on user-reported symptoms.  
2. **CNN-Based Kidney Disease Classification:** A deep learning model that analyzes X-ray images and classifies kidney conditions into four categories: *Normal, Cyst, Stone, Tumor*.  

---

## 🚀 **How the Models Work**  

### **1️⃣ Disease Prediction Model (Machine Learning)**  
#### **🛠 What’s Used?**
- **Dataset:** A Kaggle dataset containing symptoms of multiple diseases.  
- **Techniques:** Decision Tree Classifier, Random Forest, SVM, Logistic Regression.  
- **Preprocessing:**  
  ✅ Converted categorical symptoms into numerical values.  
  ✅ Handled missing data by filling NaN values.  
  ✅ Dropped unnecessary symptom columns to optimize the model.  
  ✅ Randomized dataset to improve model generalization.  
- **Model Training:**  
  ✅ Used `DecisionTreeClassifier` to classify diseases.  
  ✅ Applied `RandomForestClassifier` and `SVM` to improve accuracy.  
  ✅ Evaluated with **accuracy score and cross-validation**.  
- **Prediction:**  
  ✅ Users enter symptoms in **plain text** (e.g., *fever, cough, headache*).  
  ✅ The model maps these symptoms to potential diseases and returns the best match.  

#### **📌 How to Use?**  
Run the script:  
```bash
python disease_prediction.py
```
Then, enter symptoms as a **comma-separated** list:  
```
Enter symptoms: fever, cough, headache
```
The model will predict the most likely disease.  

---

### **2️⃣ CNN-Based Kidney Disease Classification (Deep Learning)**  
#### **🛠 What’s Used?**  
- **Dataset:** Kidney X-ray images with 4 categories:  
  ✅ **Normal**  
  ✅ **Cyst**  
  ✅ **Stone**  
  ✅ **Tumor**  
- **Model Architecture (Convolutional Neural Network - CNN)**  
  ✅ Input Layer (Image processing)  
  ✅ 3 Convolutional Layers with ReLU Activation  
  ✅ Max Pooling for Feature Extraction  
  ✅ Fully Connected Layers for Classification  
  ✅ Softmax Activation for Final Prediction  
- **Training Strategy**  
  ✅ Data Augmentation to handle overfitting  
  ✅ Image Rescaling for uniform input  
  ✅ Categorical Cross-Entropy Loss for multi-class classification  
  ✅ Adam Optimizer for faster convergence  

#### **📌 How to Use?**  
Run the script and provide an X-ray image path:  
```bash
python kidney_classification.py --image "path/to/image.jpg"
```
The system will analyze the image and predict:  
✅ **Normal**  
⚠️ **Cyst**  
⚠️ **Stone**  
⚠️ **Tumor**  

---

## 📂 **Project Structure**  
```
📦 disease-kidney-classification
 ┣ 📂 dataset/                # CSV and X-ray image datasets
 ┣ 📂 models/                 # Trained ML and CNN models
 ┣ 📜 disease_prediction.py    # Symptom-based disease prediction
 ┣ 📜 kidney_classification.py # CNN-based kidney X-ray classification
 ┣ 📜 requirements.txt         # Required dependencies
 ┣ 📜 README.md                # Documentation
```

---

## 💡 **Technologies & Libraries Used**  
✔ **Machine Learning** - Decision Trees, Random Forest, SVM, Logistic Regression  
✔ **Deep Learning** - Convolutional Neural Networks (CNN)  
✔ **Python Libraries** - TensorFlow, Keras, Pandas, Scikit-learn, OpenCV, NumPy  
✔ **Metrics Used** - Accuracy Score, Cross-Validation, Precision, Recall  

---

## 🎯 **Future Enhancements**  
🚀 Deploy as a **Web App** (Flask / FastAPI) for real-time disease prediction.  
🚀 Integrate a **Chatbot** for an interactive AI healthcare assistant.  
🚀 Improve CNN performance with **transfer learning (ResNet / VGG16)**.  

---

## 🤝 **Contributing**  
Contributions are welcome! Fork this repository, make improvements, and submit a pull request.  
